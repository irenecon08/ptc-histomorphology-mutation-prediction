"""Tests for stratified and paired bootstrap confidence intervals."""

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from ptc_mil.evaluation.bootstrap import bootstrap_ci, paired_bootstrap, stratified_resample

Preds = tuple[np.ndarray, np.ndarray]


@pytest.fixture
def binary_predictions() -> Preds:
    rng = np.random.default_rng(0)
    labels = np.array([0] * 40 + [1] * 6)
    pos = np.clip(rng.normal(0.2 + 0.3 * labels, 0.15), 0.0, 1.0)
    return np.column_stack([1 - pos, pos]), labels


def test_resample_preserves_class_sizes() -> None:
    labels = np.array([0] * 7 + [1] * 3 + [2] * 5)
    idx = stratified_resample(labels, 3, np.random.RandomState(1))
    assert np.bincount(labels[idx]).tolist() == [7, 3, 5]


def test_matches_original_dissertation_algorithm(binary_predictions: Preds) -> None:
    """Same draws and same CI as the dissertation's bootstrap_ci.py for one threshold."""
    probs, labels = binary_predictions
    np.random.seed(42)
    aucs, bal_accs = [], []
    for _ in range(200):
        idx = []
        for c in range(2):
            class_idx = np.where(labels == c)[0]
            idx.extend(np.random.choice(class_idx, size=len(class_idx), replace=True))
        idx = np.array(idx)
        aucs.append(roc_auc_score(labels[idx], probs[idx, 1]))
        bal_accs.append(balanced_accuracy_score(labels[idx], (probs[idx, 1] >= 0.3).astype(int)))

    result = bootstrap_ci(probs, labels, {"t": 0.3}, n_resamples=200, seed=42)["t"]
    assert result.samples["auc"] == pytest.approx(aucs)
    assert result.ci("bal_acc") == pytest.approx(
        (np.percentile(bal_accs, 2.5), np.percentile(bal_accs, 97.5))
    )


def test_thresholds_share_resamples(binary_predictions: Preds) -> None:
    probs, labels = binary_predictions
    results = bootstrap_ci(probs, labels, {"default": 0.5, "optimal": 0.3}, 100, seed=42)
    assert results["default"].samples["auc"] == results["optimal"].samples["auc"]
    assert results["default"].samples["bal_acc"] != results["optimal"].samples["bal_acc"]


def test_paired_difference_of_identical_models_is_zero(binary_predictions: Preds) -> None:
    probs, labels = binary_predictions
    result = paired_bootstrap(probs, probs, labels, 0.3, 0.3, 100, seed=42)
    assert result["auc"]["point_diff"] == 0.0
    assert result["auc"]["95_ci"] == [0.0, 0.0]
    assert not result["auc"]["significant"]


def test_paired_detects_clearly_better_model(binary_predictions: Preds) -> None:
    probs, labels = binary_predictions
    perfect = np.column_stack([1 - labels, labels]).astype(float)
    result = paired_bootstrap(probs, perfect, labels, 0.3, 0.5, 200, seed=42)
    assert result["auc"]["point_diff"] > 0
    assert result["bal_acc"]["significant"]
