"""Fusion classifier head and the frozen-ABMIL slide embeddings it takes as input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ptc_mil.data.dataset import load_tile_embeddings
from ptc_mil.data.embedding_index import EmbeddingIndex
from ptc_mil.models.abmil import ABMIL


class FusionClassifier(nn.Module):
    """Two-layer head on [normalised ABMIL slide embedding ++ clinical features].

    Args:
        input_dim: Slide-embedding size plus the number of clinical features.
        fc_dim: Hidden layer size.
        num_classes: Number of output classes.
        dropout: Dropout before the output layer.
    """

    def __init__(self, input_dim: int, fc_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def slide_embeddings(
    model: ABMIL, index: EmbeddingIndex, patients: list[str], device: torch.device
) -> dict[str, np.ndarray]:
    """Attention-pooled slide embedding of each patient from a frozen ABMIL model.

    Patients without an embedding file are left out.
    """
    model.eval()
    out = {}
    for patient in patients:
        if patient not in index:
            continue
        features = load_tile_embeddings(str(index.slide_for(patient))).to(device)
        _, _, z = model.forward_with_embedding(features)
        out[patient] = z.squeeze(0).cpu().numpy()
    return out


@dataclass
class ZNormalizer:
    """Per-dimension z-scoring of slide embeddings, fitted on training patients only."""

    mean: np.ndarray
    std: np.ndarray

    EPSILON = 1e-8

    @classmethod
    def fit(cls, vectors: np.ndarray) -> ZNormalizer:
        return cls(vectors.mean(axis=0), vectors.std(axis=0) + cls.EPSILON)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return (z - self.mean) / self.std

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> ZNormalizer:
        stats = np.load(path)
        return cls(stats["mean"], stats["std"])
