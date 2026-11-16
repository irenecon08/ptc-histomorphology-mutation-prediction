"""Stratified bootstrap confidence intervals, for one model or for the difference of two.

Resampling is done within each class separately, at that class's original size and
with replacement, so every resample keeps the test set's class proportions (dissertation
§3.10). Predictions are computed once; each resample only re-indexes them.

If AUROC or AUPRC cannot be computed on a resample, the whole resample is dropped from
every metric. For a binary task evaluated at several thresholds, each resample is
shared across the thresholds, so the threshold-independent AUROC and AUPRC intervals
are identical for all of them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ptc_mil.evaluation.metrics import METRIC_NAMES, Metrics, compute_metrics

logger = logging.getLogger(__name__)


def stratified_resample(
    labels: np.ndarray, num_classes: int, rng: np.random.RandomState
) -> np.ndarray:
    """Indices of one stratified bootstrap resample."""
    indices: list[int] = []
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        if len(class_indices) == 0:
            continue
        indices.extend(rng.choice(class_indices, size=len(class_indices), replace=True))
    return np.array(indices)


def percentile_ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    """Two-sided 95% percentile interval."""
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


@dataclass
class BootstrapResult:
    """Point estimate, 95% CI and the raw resampled values of every metric."""

    point: Metrics
    samples: dict[str, list[float]]
    n_requested: int

    @property
    def n_valid(self) -> int:
        return len(self.samples["auc"])

    def ci(self, metric: str) -> tuple[float, float]:
        return percentile_ci(self.samples[metric])

    def as_dict(self) -> dict:
        return {
            "point_estimate": self.point.as_dict(),
            "95_ci": {m: list(self.ci(m)) for m in METRIC_NAMES},
            "n_bootstrap_valid": self.n_valid,
            "n_bootstrap_requested": self.n_requested,
        }


def bootstrap_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[str, float | None],
    n_resamples: int,
    seed: int,
) -> dict[str, BootstrapResult]:
    """Bootstrap CIs for one model's test predictions.

    Args:
        probs: Predicted probabilities, shape (n_slides, n_classes).
        labels: True integer labels.
        thresholds: Named decision rules to report, e.g. ``{"argmax": None}`` for the
            multiclass task or ``{"default": 0.5, "optimal": 0.13}`` for RET.
        n_resamples: Number of bootstrap resamples.
        seed: Seed for ``np.random.RandomState``.

    Returns:
        One result per entry of ``thresholds``.
    """
    labels = np.asarray(labels)
    num_classes = probs.shape[1]
    rng = np.random.RandomState(seed)
    samples: dict[str, dict[str, list[float]]] = {
        name: {m: [] for m in METRIC_NAMES} for name in thresholds
    }
    for _ in range(n_resamples):
        idx = stratified_resample(labels, num_classes, rng)
        per_threshold = {
            name: compute_metrics(probs[idx], labels[idx], t) for name, t in thresholds.items()
        }
        if not all(m.ranking_defined for m in per_threshold.values()):
            continue
        for name, metrics in per_threshold.items():
            for metric, value in metrics.as_dict().items():
                samples[name][metric].append(value)

    results = {
        name: BootstrapResult(compute_metrics(probs, labels, t), samples[name], n_resamples)
        for name, t in thresholds.items()
    }
    for name, result in results.items():
        logger.info(
            "%s: %s",
            name,
            ", ".join(
                f"{m} {getattr(result.point, m):.4f} [{result.ci(m)[0]:.4f}, {result.ci(m)[1]:.4f}]"
                for m in METRIC_NAMES
            ),
        )
    return results


def paired_bootstrap(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
    threshold_a: float | None,
    threshold_b: float | None,
    n_resamples: int,
    seed: int,
) -> dict:
    """Bootstrap CI of the difference (model B - model A) on the same patients.

    The same resample is applied to both models in every iteration, so the interval
    reflects the difference between models rather than two independent intervals. A
    difference is called significant when its 95% CI excludes zero.
    """
    labels = np.asarray(labels)
    num_classes = probs_a.shape[1]
    rng = np.random.RandomState(seed)
    point_a = compute_metrics(probs_a, labels, threshold_a)
    point_b = compute_metrics(probs_b, labels, threshold_b)

    diffs: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    for _ in range(n_resamples):
        idx = stratified_resample(labels, num_classes, rng)
        a = compute_metrics(probs_a[idx], labels[idx], threshold_a)
        b = compute_metrics(probs_b[idx], labels[idx], threshold_b)
        if not (a.ranking_defined and b.ranking_defined):
            continue
        for m in METRIC_NAMES:
            diffs[m].append(getattr(b, m) - getattr(a, m))

    results: dict = {}
    for m in METRIC_NAMES:
        low, high = percentile_ci(diffs[m])
        results[m] = {
            "point_diff": getattr(point_b, m) - getattr(point_a, m),
            "95_ci": [low, high],
            "significant": bool(low > 0 or high < 0),
        }
        logger.info(
            "%-8s diff %+.4f [%+.4f, %+.4f]%s",
            m,
            results[m]["point_diff"],
            low,
            high,
            " *" if results[m]["significant"] else "",
        )
    results["n_valid"] = len(diffs["auc"])
    results["point_a"] = point_a.as_dict()
    results["point_b"] = point_b.as_dict()
    return results
