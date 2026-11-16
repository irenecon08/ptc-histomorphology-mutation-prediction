"""One object giving analysis scripts their paths, models and per-patient files."""

from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from ptc_mil.config import ExperimentConfig, Paths
from ptc_mil.data.labels import get_task
from ptc_mil.data.slide_files import SlideFiles
from ptc_mil.models.abmil import ABMIL
from ptc_mil.utils.runtime import default_device

PATHS_ENV = "PTC_MIL_PATHS"
EXPERIMENT_ENV = "PTC_MIL_EXPERIMENT"


class AnalysisContext:
    """Paths, settings, trained models and slide files for the downstream analyses.

    Per-patient lookups return None when a file is missing, as the dissertation's
    analysis scripts expect. For multi-slide patients they all refer to the same slide
    (the alphabetically first), so attention, tiles and pixels always match.
    """

    def __init__(self, paths: Paths, config: ExperimentConfig, device: torch.device) -> None:
        self.paths = paths
        self.config = config
        self.device = device

    @cached_property
    def files(self) -> SlideFiles:
        return SlideFiles(self.paths)

    @cached_property
    def splits(self) -> pd.DataFrame:
        return pd.read_csv(self.paths.splits_csv)

    def output_dir(self, name: str) -> Path:
        """``results/analyses/<name>/``, created if needed."""
        return self.paths.analysis_dir(name)

    # --- Models ---------------------------------------------------------------------

    def checkpoint(self, task_name: str) -> Path:
        return self.paths.results_dir / "checkpoints" / self.config.task(task_name).checkpoint

    def load_abmil(self, task_name: str, device: torch.device | None = None) -> tuple[ABMIL, Path]:
        """The final trained ABMIL model for a task (strict state-dict loading)."""
        device = device or self.device
        task = get_task(task_name)
        settings = self.config.task(task_name)
        model = ABMIL(
            self.files.embeddings.embed_dim, settings.attn_dim, settings.fc_dim, task.num_classes
        )
        path = self.checkpoint(task_name)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return model.to(device).eval(), path

    # --- Per-patient files -----------------------------------------------------------

    def embedding_file(self, patient: str) -> Path | None:
        return self.files.embeddings.get(patient)

    def tile_file(self, patient: str) -> Path | None:
        try:
            return self.files.tiles(patient)
        except (KeyError, FileNotFoundError):
            return None

    def slide_file(self, patient: str) -> Path | None:
        try:
            return self.files.svs(patient)
        except (KeyError, FileNotFoundError):
            return None

    def tile_info(self, patient: str) -> tuple[np.ndarray | None, int | None, float | None]:
        """Tile coordinates, tile size and level downsample, or (None, None, None)."""
        path = self.tile_file(patient)
        if path is None:
            return None, None, None
        with h5py.File(path, "r") as f:
            return f["coords"][:], int(f.attrs["tile_size"]), float(f.attrs["level_downsample"])

    @torch.no_grad()
    def tile_attention(
        self, model: ABMIL, patient: str, device: torch.device | None = None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Attention weight of every tile and the tile coordinates stored with the embeddings."""
        device = device or self.device
        path = self.embedding_file(patient)
        if path is None:
            return None, None
        with h5py.File(path, "r") as f:
            features = torch.tensor(f["features"][:], dtype=torch.float32).to(device)
            coords = f["coords"][:]
        _, attention = model(features)
        return attention.squeeze().cpu().numpy(), coords


def load_context(device: torch.device | None = None) -> AnalysisContext:
    """Build the context from the YAML files named by the environment (see module docs)."""
    paths = Paths.from_yaml(os.environ.get(PATHS_ENV, "configs/paths.yaml"))
    config = ExperimentConfig.from_yaml(os.environ.get(EXPERIMENT_ENV, "configs/experiment.yaml"))
    return AnalysisContext(paths, config, device or default_device())
