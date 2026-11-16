"""The steps of the experiment, each reading and writing files under ``results_dir``.

Every command-line subcommand calls one function here. Steps communicate only through
files, so any step can be re-run on its own:

    results/
      checkpoints/          best ABMIL model per task
      tuning/               grid-search checkpoints and ranked validation tables
      predictions/          per-patient probabilities, one CSV per model, task and split
      thresholds/           validation-selected RET thresholds
      metrics/              point estimates and bootstrap CIs on the test set
      fusion/               fusion search tables, checkpoints and normalisation stats
      comparisons/          paired bootstrap differences between models
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from ptc_mil.config import ExperimentConfig, Paths
from ptc_mil.data.clinical import load_clinical_features
from ptc_mil.data.dataset import SPLITS, SlideDataset
from ptc_mil.data.embedding_index import EmbeddingIndex
from ptc_mil.data.labels import MULTICLASS, RET_BINARY, Task
from ptc_mil.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from ptc_mil.evaluation.metrics import compute_metrics
from ptc_mil.evaluation.predictions import Predictions, predictions_path
from ptc_mil.evaluation.thresholds import select_threshold, threshold_grid
from ptc_mil.fusion.model import FusionClassifier
from ptc_mil.fusion.train import (
    FusionData,
    fusion_predictions,
    prepare_fusion_data,
    search_fusion,
    train_fusion,
)
from ptc_mil.models.abmil import ABMIL
from ptc_mil.models.baseline import make_baseline, mean_pooled_features
from ptc_mil.training.abmil import inverse_frequency_weights, predict, train_abmil, tune_abmil
from ptc_mil.utils.runtime import seed_everything

logger = logging.getLogger(__name__)

MODELS = ("baseline", "abmil", "fusion")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %s", path)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run the step that creates it first")
    return json.loads(path.read_text())


class Experiment:
    """Paths, settings and shared lookups for one run of the pipeline."""

    def __init__(self, paths: Paths, config: ExperimentConfig, device: torch.device) -> None:
        self.paths = paths
        self.config = config
        self.device = device
        self._splits: pd.DataFrame | None = None
        self._index: EmbeddingIndex | None = None

    @property
    def results(self) -> Path:
        return self.paths.results_dir

    @property
    def splits(self) -> pd.DataFrame:
        if self._splits is None:
            self._splits = pd.read_csv(self.paths.splits_csv)
        return self._splits

    @property
    def index(self) -> EmbeddingIndex:
        if self._index is None:
            self._index = EmbeddingIndex(self.paths.embeddings_dir)
        return self._index

    def dataset(self, task: Task, split: str) -> SlideDataset:
        return SlideDataset(self.splits, split, task, self.index)

    def class_weights(self, task: Task) -> list[float] | None:
        """ABMIL loss weights: inverse frequency for RET, none for the multiclass task."""
        if task is RET_BINARY:
            return inverse_frequency_weights(self.config.ret_class_counts)
        return None

    def checkpoint(self, task: Task) -> Path:
        return self.results / "checkpoints" / self.config.task(task.name).checkpoint

    def load_abmil(self, task: Task) -> ABMIL:
        settings = self.config.task(task.name)
        model = ABMIL(self.index.embed_dim, settings.attn_dim, settings.fc_dim, task.num_classes)
        state = torch.load(self.checkpoint(task), map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        return model.to(self.device).eval()

    def threshold(self, model: str) -> float:
        """The RET decision threshold reported for a model."""
        if model.startswith("fusion"):
            if model == "fusion" and self.config.thresholds.fusion is not None:
                return self.config.thresholds.fusion
            return float(
                read_json(self.results / "thresholds" / f"{model}_ret_binary.json")["threshold"]
            )
        return float(getattr(self.config.thresholds, model))


# --- Baseline -------------------------------------------------------------------------


def run_baseline(exp: Experiment, task: Task) -> None:
    """Fit the mean-pooling baseline on training slides; save model and predictions."""
    data = {split: exp.dataset(task, split) for split in SPLITS}
    clf = make_baseline(task.num_classes, exp.config.baseline_max_iter)
    clf.fit(mean_pooled_features(data["train"]), data["train"].labels)
    model_path = exp.results / "checkpoints" / f"baseline_{task.name}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)

    for split in ("val", "test"):
        probs = clf.predict_proba(mean_pooled_features(data[split]))
        preds = Predictions(data[split].patients, np.array(data[split].labels), probs)
        preds.to_csv(predictions_path(exp.results, "baseline", task, split), task)
        logger.info("Baseline %s %s: %s", task.name, split, compute_metrics(probs, preds.labels))


# --- ABMIL ----------------------------------------------------------------------------


def run_tuning(exp: Experiment, task: Task, extended: bool = False) -> None:
    """Grid search over learning rate and model size, ranked by validation AUROC."""
    seed_everything(exp.config.seed)
    out_dir = exp.paths.tuning_dir / task.name
    ranked = tune_abmil(
        exp.dataset(task, "train"),
        exp.dataset(task, "val"),
        exp.config,
        out_dir,
        exp.device,
        exp.index.embed_dim,
        exp.class_weights(task),
        extended=extended,
    )
    best = ranked[0]
    write_json(
        out_dir / f"tuning_{task.name}.json",
        {"task": task.name, "selected_config": best["config"], "configs_ranked_by_val_auc": ranked},
    )
    selected = exp.config.task(task.name)
    if (best["lr"], best["attn_dim"], best["fc_dim"]) != (
        selected.lr,
        selected.attn_dim,
        selected.fc_dim,
    ):
        logger.warning(
            "Best configuration %s differs from configs/experiment.yaml; update it before training",
            best["config"],
        )


def run_training(exp: Experiment, task: Task) -> None:
    """Train the final ABMIL model with the selected hyperparameters."""
    seed_everything(exp.config.seed)
    settings = exp.config.task(task.name)
    model = ABMIL(exp.index.embed_dim, settings.attn_dim, settings.fc_dim, task.num_classes)
    result = train_abmil(
        model.to(exp.device),
        exp.dataset(task, "train"),
        exp.dataset(task, "val"),
        lr=settings.lr,
        settings=exp.config.training,
        checkpoint_path=exp.checkpoint(task),
        device=exp.device,
        class_weights=exp.class_weights(task),
    )
    write_json(
        exp.results / "checkpoints" / f"training_{task.name}.json",
        {"task": task.name, "hyperparameters": vars(settings), **vars(result)},
    )


def run_abmil_predictions(exp: Experiment, task: Task) -> None:
    """Predict validation and test slides with the trained ABMIL model, once."""
    model = exp.load_abmil(task)
    for split in ("val", "test"):
        data = exp.dataset(task, split)
        preds = Predictions(data.patients, np.array(data.labels), predict(model, data, exp.device))
        preds.to_csv(predictions_path(exp.results, "abmil", task, split), task)


# --- Thresholds and evaluation --------------------------------------------------------


def run_calibration(exp: Experiment, model: str) -> float:
    """Select the RET threshold maximising validation balanced accuracy."""
    val = Predictions.from_csv(predictions_path(exp.results, model, RET_BINARY, "val"))
    is_fusion = model.startswith("fusion")
    grid = exp.config.thresholds.fusion_grid if is_fusion else exp.config.thresholds.abmil_grid
    threshold, val_bal_acc = select_threshold(val.labels, val.probs[:, 1], threshold_grid(*grid))
    write_json(
        exp.results / "thresholds" / f"{model}_ret_binary.json",
        {"model": model, "threshold": threshold, "val_bal_acc": val_bal_acc, "grid": list(grid)},
    )
    if not is_fusion and not np.isclose(threshold, exp.threshold(model)):
        logger.warning(
            "Selected threshold %.2f differs from the configured %.2f in experiment.yaml",
            threshold,
            exp.threshold(model),
        )
    return threshold


def run_evaluation(exp: Experiment, model: str, task: Task) -> None:
    """Test-set point estimates and stratified bootstrap 95% CIs."""
    test = Predictions.from_csv(predictions_path(exp.results, model, task, "test"))
    if task is MULTICLASS:
        thresholds: dict[str, float | None] = {"argmax": None}
    else:
        thresholds = {"default_0.5": 0.5, "optimal": exp.threshold(model)}
    results = bootstrap_ci(
        test.probs,
        test.labels,
        thresholds,
        exp.config.bootstrap_resamples,
        exp.config.bootstrap_seed,
    )
    write_json(
        exp.results / "metrics" / f"{model}_{task.name}.json",
        {
            "model": model,
            "task": task.name,
            "n_test": len(test.patients),
            "thresholds": thresholds,
            "results": {name: r.as_dict() for name, r in results.items()},
        },
    )
    samples: dict[str, Any] = {
        f"{name}__{metric}": np.array(values)
        for name, r in results.items()
        for metric, values in r.samples.items()
    }
    np.savez(exp.results / "metrics" / f"{model}_{task.name}_bootstrap_samples.npz", **samples)


def run_comparison(exp: Experiment, task: Task, reference: str, other: str) -> dict:
    """Paired bootstrap of ``other - reference`` on the patients both models scored."""
    a = Predictions.from_csv(predictions_path(exp.results, reference, task, "test"))
    b = Predictions.from_csv(predictions_path(exp.results, other, task, "test"))
    a, b = a.align(b)
    logger.info("Comparing %s vs %s on %d shared test patients", other, reference, len(a.patients))
    ta = exp.threshold(reference) if task is RET_BINARY else None
    tb = exp.threshold(other) if task is RET_BINARY else None
    result = paired_bootstrap(
        a.probs,
        b.probs,
        a.labels,
        ta,
        tb,
        exp.config.bootstrap_resamples,
        exp.config.bootstrap_seed,
    )
    result.update(
        {
            "reference": reference,
            "other": other,
            "n_patients": len(a.patients),
            "threshold_reference": ta,
            "threshold_other": tb,
        }
    )
    write_json(exp.results / "comparisons" / f"{other}_vs_{reference}_{task.name}.json", result)
    return result


# --- Fusion ---------------------------------------------------------------------------


def _fusion_data(exp: Experiment, task: Task, exclude_columns: list[str] | None) -> FusionData:
    clinical, columns = load_clinical_features(
        exp.paths.require("clinical_features_csv"), exclude_columns
    )
    logger.info("Clinical feature columns (%d): %s", len(columns), columns)
    return prepare_fusion_data(
        task, exp.load_abmil(task), exp.splits, exp.index, clinical, columns, exp.device
    )


def _fusion_suffix(exclude_columns: list[str] | None) -> str:
    return "_excl_" + "_".join(exclude_columns) if exclude_columns else ""


def run_fusion_search(exp: Experiment, task: Task) -> None:
    data = _fusion_data(exp, task, None)
    table = search_fusion(
        data, exp.config.fusion, exp.paths.fusion_dir, exp.device, exp.config.seed
    )
    out = exp.paths.fusion_dir / f"fusion_search_{task.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    logger.info("Fusion search for %s:\n%s", task.name, table.to_string(index=False))


def selected_fusion_config(exp: Experiment, task: Task) -> dict[str, float]:
    """Fusion hyperparameters: from experiment.yaml if set, else the top row of the search."""
    if task.name in exp.config.fusion.selected:
        return exp.config.fusion.selected[task.name]
    table = pd.read_csv(exp.paths.fusion_dir / f"fusion_search_{task.name}.csv")
    best = table.iloc[0]
    return {
        "lr": float(best["lr"]),
        "fc_dim": int(best["fc_dim"]),
        "dropout": float(best["dropout"]),
    }


def run_fusion_training(
    exp: Experiment, task: Task, exclude_columns: list[str] | None = None
) -> None:
    """Train the final fusion head and save its validation and test predictions.

    With ``exclude_columns`` (the site-confound check), outputs get a matching suffix
    so the main run is never overwritten.
    """
    seed_everything(exp.config.seed)
    suffix = _fusion_suffix(exclude_columns)
    model_name = f"fusion{suffix}"
    hp = selected_fusion_config(exp, task)
    data = _fusion_data(exp, task, exclude_columns)
    data.normalizer.save(exp.paths.fusion_dir / f"z_norm_stats_{task.name}{suffix}.npz")
    checkpoint = exp.paths.fusion_dir / f"best_fusion_{task.name}{suffix}.pt"
    run = train_fusion(
        data,
        int(hp["fc_dim"]),
        float(hp["dropout"]),
        float(hp["lr"]),
        exp.config.fusion,
        checkpoint,
        exp.device,
    )
    model = FusionClassifier(
        data.input_dim, int(hp["fc_dim"]), task.num_classes, float(hp["dropout"])
    )
    model.load_state_dict(torch.load(checkpoint, map_location=exp.device, weights_only=True))
    model.to(exp.device)
    for split in ("val", "test"):
        preds = fusion_predictions(
            model, data, split, exp.config.fusion.eval_batch_size, exp.device
        )
        preds.to_csv(predictions_path(exp.results, model_name, task, split), task)
    write_json(
        exp.paths.fusion_dir / f"fusion_training_{task.name}{suffix}.json",
        {
            "task": task.name,
            "hyperparameters": hp,
            "clinical_columns": data.clinical_columns,
            "excluded_columns": exclude_columns,
            "best_val": run.best_val.as_dict() if run.best_val else None,
            "epochs_trained": run.epochs_trained,
            "n_patients": {split: len(s.patients) for split, s in data.splits.items()},
        },
    )
