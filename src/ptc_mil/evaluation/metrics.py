"""Classification metrics used throughout: balanced accuracy, AUROC, AUPRC and macro F1.

For the multiclass task, AUROC and AUPRC are one-vs-rest and macro-averaged. For the
binary RET task they use the predicted RET-positive probability, and class predictions
use a decision threshold on that probability.
"""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

METRIC_NAMES = ("bal_acc", "auc", "auprc", "f1")


@dataclass(frozen=True)
class Metrics:
    bal_acc: float
    auc: float
    auprc: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @property
    def ranking_defined(self) -> bool:
        """False when AUROC/AUPRC could not be computed (e.g. only one class present)."""
        return not (np.isnan(self.auc) or np.isnan(self.auprc))


def predict_classes(probs: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Predicted class per slide: argmax, or ``P(positive) >= threshold`` for binary tasks."""
    if threshold is None:
        return np.argmax(probs, axis=1)
    if probs.shape[1] != 2:
        raise ValueError("A decision threshold only applies to binary tasks")
    return (probs[:, 1] >= threshold).astype(int)


def ranking_metrics(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """AUROC and AUPRC; NaN when they are undefined for these labels."""
    try:
        if probs.shape[1] == 2:
            return (
                float(roc_auc_score(labels, probs[:, 1])),
                float(average_precision_score(labels, probs[:, 1])),
            )
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        auprc = np.mean(
            [
                average_precision_score((labels == c).astype(int), probs[:, c])
                for c in range(probs.shape[1])
            ]
        )
        return float(auc), float(auprc)
    except ValueError:
        return float("nan"), float("nan")


def compute_metrics(
    probs: np.ndarray, labels: np.ndarray, threshold: float | None = None
) -> Metrics:
    """All four metrics for one set of predictions.

    Args:
        probs: Predicted class probabilities, shape (n_slides, n_classes).
        labels: True integer labels, shape (n_slides,).
        threshold: Decision threshold for binary tasks; ``None`` uses argmax
            (equivalent to 0.5 for two classes).
    """
    labels = np.asarray(labels)
    preds = predict_classes(probs, threshold)
    auc, auprc = ranking_metrics(probs, labels)
    return Metrics(
        bal_acc=float(balanced_accuracy_score(labels, preds)),
        auc=auc,
        auprc=auprc,
        f1=float(f1_score(labels, preds, average="macro")),
    )
