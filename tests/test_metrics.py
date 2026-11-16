"""Tests for metrics and validation-threshold selection."""

import numpy as np
import pytest

from ptc_mil.evaluation.metrics import compute_metrics, predict_classes
from ptc_mil.evaluation.thresholds import select_threshold, threshold_grid


def test_perfect_multiclass_predictions() -> None:
    probs = np.eye(3)[[0, 1, 2, 0, 1, 2]] * 0.9 + 0.1 / 3
    m = compute_metrics(probs, np.array([0, 1, 2, 0, 1, 2]))
    assert m.bal_acc == m.auc == m.auprc == m.f1 == pytest.approx(1.0)


def test_binary_threshold_changes_class_predictions_not_ranking() -> None:
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.95, 0.05]])
    labels = np.array([0, 1, 1, 0])
    at_half = compute_metrics(probs, labels, threshold=0.5)
    at_low = compute_metrics(probs, labels, threshold=0.15)
    assert at_half.bal_acc == 0.5
    assert at_low.bal_acc == 1.0
    assert at_half.auc == at_low.auc == 1.0


def test_threshold_rejected_for_multiclass() -> None:
    with pytest.raises(ValueError, match="binary"):
        predict_classes(np.full((2, 3), 1 / 3), threshold=0.5)


def test_ranking_metrics_undefined_with_one_class() -> None:
    m = compute_metrics(np.array([[0.6, 0.4], [0.3, 0.7]]), np.array([1, 1]))
    assert not m.ranking_defined


def test_threshold_grid_matches_dissertation() -> None:
    grid = threshold_grid(0.01, 0.55, 0.01)
    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] == pytest.approx(0.54)
    assert len(grid) == 54


def test_select_threshold_picks_first_best() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    pos = np.array([0.02, 0.05, 0.08, 0.12, 0.30])
    threshold, score = select_threshold(labels, pos, threshold_grid(0.01, 0.55, 0.01))
    assert threshold == pytest.approx(0.09)
    assert score == 1.0
