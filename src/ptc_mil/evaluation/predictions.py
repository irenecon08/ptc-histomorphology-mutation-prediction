"""Per-patient prediction tables: the single record of a model's output on a split.

Every evaluation step (metrics, bootstrap, paired comparison, figures) reads these
CSV files, so each model's test predictions are computed exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ptc_mil.data.labels import Task


@dataclass
class Predictions:
    patients: list[str]
    labels: np.ndarray
    probs: np.ndarray

    def to_csv(self, path: Path, task: Task) -> None:
        df = pd.DataFrame({"patient": self.patients, "true_label": self.labels})
        for i, name in enumerate(task.class_names):
            df[f"prob_{name}"] = self.probs[:, i]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: Path) -> Predictions:
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        df = pd.read_csv(path)
        prob_columns = [c for c in df.columns if c.startswith("prob_")]
        return cls(
            patients=df["patient"].tolist(),
            labels=df["true_label"].to_numpy(),
            probs=df[prob_columns].to_numpy(),
        )

    def align(self, other: Predictions) -> tuple[Predictions, Predictions]:
        """Restrict both prediction sets to their shared patients, in sorted order.

        Raises:
            ValueError: If a shared patient has different labels in the two sets.
        """
        shared = sorted(set(self.patients) & set(other.patients))
        a, b = self._subset(shared), other._subset(shared)
        mismatched = int((a.labels != b.labels).sum())
        if mismatched:
            raise ValueError(f"{mismatched} shared patient(s) have different labels")
        return a, b

    def _subset(self, patients: list[str]) -> Predictions:
        position = {p: i for i, p in enumerate(self.patients)}
        idx = [position[p] for p in patients]
        return Predictions(list(patients), self.labels[idx], self.probs[idx])


def predictions_path(results_dir: Path, model: str, task: Task, split: str = "test") -> Path:
    """Where a prediction table lives, e.g. results/predictions/abmil_ret_binary_test.csv."""
    return results_dir / "predictions" / f"{model}_{task.name}_{split}.csv"
