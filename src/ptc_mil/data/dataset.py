"""Slide-level dataset: one item is (tile embeddings of a slide, its label)."""

import logging

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset

from ptc_mil.data.embedding_index import EmbeddingIndex
from ptc_mil.data.labels import Task

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def load_tile_embeddings(path: str) -> torch.Tensor:
    """Read the UNI2-h tile embeddings of one slide, shape (n_tiles, embed_dim)."""
    with h5py.File(path, "r") as f:
        return torch.tensor(f["features"][:], dtype=torch.float32)


def task_rows(splits: pd.DataFrame, task: Task) -> pd.DataFrame:
    """Rows of the splits table that belong to a task, with an integer ``label`` column.

    For the multiclass task only patients labelled BRAF_V600E, RAS or Other are kept;
    the RET task uses every patient.
    """
    rows = splits[splits[task.label_column].isin(task.label_map)].copy()
    dropped = len(splits) - len(rows)
    if dropped:
        logger.info("%s: %d patient(s) have labels outside the task", task.name, dropped)
    rows["label"] = task.encode(rows[task.label_column])
    return rows


class SlideDataset(Dataset[tuple[torch.Tensor, int]]):
    """Slides of one split for one task.

    Patients without an embedding file are excluded when the dataset is built, and
    logged, rather than skipped silently during training.

    Args:
        splits: Splits table with ``patient``, ``split`` and the task's label column.
        split: ``"train"``, ``"val"`` or ``"test"``.
        task: The prediction task.
        index: Where each patient's embedding file is.
    """

    def __init__(self, splits: pd.DataFrame, split: str, task: Task, index: EmbeddingIndex) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
        rows = task_rows(splits[splits["split"] == split], task)

        has_embedding = rows["patient"].map(lambda p: p in index)
        if not has_embedding.all():
            missing = sorted(rows.loc[~has_embedding, "patient"])
            logger.warning(
                "%s/%s: %d patient(s) have no embedding file and are excluded: %s",
                task.name,
                split,
                len(missing),
                missing,
            )
            rows = rows[has_embedding]

        self.task = task
        self.split = split
        self.patients: list[str] = rows["patient"].tolist()
        self.labels: list[int] = rows["label"].tolist()
        self.paths = [str(index.slide_for(p)) for p in self.patients]
        logger.info("%s/%s: %d slides", task.name, split, len(self))

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return load_tile_embeddings(self.paths[i]), self.labels[i]

    def class_counts(self) -> list[int]:
        """Number of slides per class, in class order."""
        return [self.labels.count(c) for c in range(self.task.num_classes)]
