"""Fusion data preparation, training, hyperparameter search and test prediction.

The ABMIL model is frozen; only the fusion head is trained (dissertation §3.13). The
slide embedding is z-normalised with training-set statistics and concatenated with the
clinical features. Patients need both an embedding and clinical data to be included.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ptc_mil.config import FusionSettings
from ptc_mil.data.dataset import SPLITS, task_rows
from ptc_mil.data.embedding_index import EmbeddingIndex
from ptc_mil.data.labels import Task
from ptc_mil.evaluation.metrics import Metrics, compute_metrics
from ptc_mil.evaluation.predictions import Predictions
from ptc_mil.fusion.model import FusionClassifier, ZNormalizer, slide_embeddings
from ptc_mil.models.abmil import ABMIL
from ptc_mil.training.abmil import inverse_frequency_weights

logger = logging.getLogger(__name__)


@dataclass
class FusionSplit:
    patients: list[str]
    x: np.ndarray
    labels: np.ndarray

    def tensors(self) -> TensorDataset:
        return TensorDataset(
            torch.tensor(self.x, dtype=torch.float32), torch.tensor(self.labels, dtype=torch.long)
        )


@dataclass
class FusionData:
    """Fusion inputs for one task, prepared once and reused across configurations."""

    task: Task
    splits: dict[str, FusionSplit]
    normalizer: ZNormalizer
    clinical_columns: list[str]

    @property
    def input_dim(self) -> int:
        return self.splits["train"].x.shape[1]

    def class_weights(self) -> list[float] | None:
        """Inverse-frequency weights from the fusion training labels (RET only)."""
        if self.task.num_classes != 2:
            return None
        return inverse_frequency_weights(
            np.bincount(self.splits["train"].labels, minlength=2).tolist()
        )


def prepare_fusion_data(
    task: Task,
    abmil: ABMIL,
    splits: pd.DataFrame,
    index: EmbeddingIndex,
    clinical: dict[str, np.ndarray],
    clinical_columns: list[str],
    device: torch.device,
) -> FusionData:
    """Extract frozen slide embeddings, fit the z-normaliser on training patients, and
    build the fused feature matrix for each split."""
    rows = task_rows(splits, task)
    z = slide_embeddings(abmil, index, rows["patient"].tolist(), device)

    train_patients = rows.loc[rows["split"] == "train", "patient"]
    normalizer = ZNormalizer.fit(np.stack([z[p] for p in train_patients if p in z]))

    out = {}
    for split in SPLITS:
        part = rows[rows["split"] == split]
        keep = [
            (p, y)
            for p, y in zip(part["patient"], part["label"], strict=True)
            if p in z and p in clinical
        ]
        patients = [p for p, _ in keep]
        x = np.stack([np.concatenate([normalizer(z[p]), clinical[p]]) for p in patients])
        out[split] = FusionSplit(patients, x, np.array([y for _, y in keep]))
        logger.info(
            "%s/%s: %d patients (%d without embedding or clinical data)",
            task.name,
            split,
            len(patients),
            len(part) - len(patients),
        )
    return FusionData(task, out, normalizer, clinical_columns)


@torch.no_grad()
def predict_fusion(
    model: FusionClassifier, split: FusionSplit, batch_size: int, device: torch.device
) -> np.ndarray:
    model.eval()
    probs = []
    for x, _ in DataLoader(split.tensors(), batch_size=batch_size, shuffle=False):
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(probs)


@dataclass
class FusionRun:
    fc_dim: int
    dropout: float
    lr: float
    best_val: Metrics | None
    epochs_trained: int


def train_fusion(
    data: FusionData,
    fc_dim: int,
    dropout: float,
    lr: float,
    settings: FusionSettings,
    checkpoint_path: Path,
    device: torch.device,
) -> FusionRun:
    """Train one fusion head with early stopping on validation AUROC.

    The schedule matches ABMIL training (Adam, ReduceLROnPlateau with factor 0.5 and
    patience // 2), with mini-batches of fixed-length fused vectors.
    """
    model = FusionClassifier(data.input_dim, fc_dim, data.task.num_classes, dropout).to(device)
    weights = data.class_weights()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device) if weights else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=settings.patience // 2
    )
    loader = DataLoader(
        data.splits["train"].tensors(), batch_size=settings.batch_size, shuffle=True
    )
    val = data.splits["val"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_auc, best_val, stale, epochs = 0.0, None, 0, 0
    for _ in range(settings.num_epochs):
        epochs += 1
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        metrics = compute_metrics(
            predict_fusion(model, val, settings.eval_batch_size, device), val.labels
        )
        val_auc = 0.0 if np.isnan(metrics.auc) else metrics.auc
        scheduler.step(val_auc)
        if val_auc > best_auc:
            best_auc, best_val, stale = val_auc, metrics, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale += 1
        if stale >= settings.patience:
            break
    return FusionRun(fc_dim, dropout, lr, best_val, epochs)


def search_fusion(
    data: FusionData, settings: FusionSettings, work_dir: Path, device: torch.device, seed: int
) -> pd.DataFrame:
    """Two-stage search ranked by validation AUROC (dissertation §3.13.2).

    Stage 1: every learning rate x hidden size, dropout 0. Stage 2: the Stage 1
    winner with each dropout in the extension. Returns the ranked table.
    """
    runs: list[tuple[int, FusionRun]] = []
    checkpoint = work_dir / f"_search_{data.task.name}.pt"
    for lr in settings.lr_grid:
        for fc_dim in settings.fc_dim_grid:
            torch.manual_seed(seed)
            runs.append((1, train_fusion(data, fc_dim, 0.0, lr, settings, checkpoint, device)))
            logger.info("Stage 1 lr=%g fc=%d: val AUROC %.4f", lr, fc_dim, _val_auc(runs[-1][1]))
    best = max((r for s, r in runs if s == 1), key=_val_auc)
    for dropout in settings.dropout_extension:
        torch.manual_seed(seed)
        runs.append(
            (2, train_fusion(data, best.fc_dim, dropout, best.lr, settings, checkpoint, device))
        )
        logger.info("Stage 2 dropout=%.1f: val AUROC %.4f", dropout, _val_auc(runs[-1][1]))
    checkpoint.unlink(missing_ok=True)

    table = pd.DataFrame(
        [
            {
                "stage": stage,
                "lr": r.lr,
                "fc_dim": r.fc_dim,
                "dropout": r.dropout,
                "val_auc": _val_auc(r),
                "val_bal_acc": r.best_val.bal_acc if r.best_val else float("nan"),
                "val_f1": r.best_val.f1 if r.best_val else float("nan"),
                "n_epochs": r.epochs_trained,
            }
            for stage, r in runs
        ]
    )
    table = table.sort_values("val_auc", ascending=False, kind="stable").reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    return table


def _val_auc(run: FusionRun) -> float:
    return run.best_val.auc if run.best_val else 0.0


def fusion_predictions(
    model: FusionClassifier, data: FusionData, split: str, batch_size: int, device: torch.device
) -> Predictions:
    part = data.splits[split]
    return Predictions(part.patients, part.labels, predict_fusion(model, part, batch_size, device))
