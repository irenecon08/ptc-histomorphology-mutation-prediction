"""Decision threshold for the binary RET task, chosen on the validation set only.

The threshold maximising balanced accuracy (equivalently Youden's J) on validation is
kept, then applied once to the test set. The first threshold reaching the maximum wins.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def threshold_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Candidate thresholds, ``np.arange(start, stop, step)`` as in the dissertation."""
    return np.arange(start, stop, step)


def select_threshold(
    labels: np.ndarray, positive_probs: np.ndarray, grid: np.ndarray
) -> tuple[float, float]:
    """Return (threshold, validation balanced accuracy) maximising balanced accuracy.

    If no threshold improves on a balanced accuracy of 0, the default of 0.5 is returned.
    """
    best_threshold, best_score = 0.5, 0.0
    for threshold in grid:
        preds = (positive_probs >= threshold).astype(int)
        score = balanced_accuracy_score(labels, preds)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score
