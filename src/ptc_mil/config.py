"""Project configuration: where the data lives (paths) and the experiment settings.

Two YAML files are used:

* ``configs/paths.yaml`` (local, not committed): absolute paths on this machine.
  Copy ``configs/paths.example.yaml`` to create it.
* ``configs/experiment.yaml`` (committed): hyperparameters and evaluation settings,
  set to the values used in the MSc dissertation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Paths:
    """Locations of inputs and outputs.

    Relative paths in the YAML file are resolved against ``root``. Optional entries
    may be left out when the corresponding stage is not run.
    """

    root: Path
    slides_dir: Path
    tiles_dir: Path
    embeddings_dir: Path
    labels_csv: Path
    splits_csv: Path
    results_dir: Path
    clinical_tsv: Path | None = None
    clinical_features_csv: Path | None = None
    thorsson_csv: Path | None = None
    analyses_dir: Path | None = None
    hf_token_file: Path | None = None

    @property
    def fusion_dir(self) -> Path:
        return self.results_dir / "fusion"

    @property
    def tuning_dir(self) -> Path:
        return self.results_dir / "tuning"

    def analysis_dir(self, name: str) -> Path:
        """Output folder for one downstream analysis, created if needed."""
        base = self.analyses_dir or (self.results_dir / "analyses")
        path = base / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def require(self, name: str) -> Path:
        """Return an optional path, raising a clear error if it was not configured."""
        value = getattr(self, name)
        if value is None:
            raise ValueError(f"'{name}' is not set in paths.yaml but is needed for this step")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> Paths:
        data = _read_yaml(path)
        if "root" not in data:
            raise ValueError(f"{path}: 'root' is required")
        root = Path(data.pop("root")).expanduser()
        known = {f.name for f in fields(cls)} - {"root"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        resolved = {k: _resolve(root, v) for k, v in data.items() if v is not None}
        return cls(root=root, **resolved)


@dataclass(frozen=True)
class TaskSettings:
    """Selected ABMIL architecture and learning rate for one task."""

    lr: float
    attn_dim: int
    fc_dim: int
    checkpoint: str


@dataclass(frozen=True)
class TrainingSettings:
    num_epochs: int = 100
    patience: int = 10
    weight_decay: float = 1e-3
    dropout: float = 0.0
    lr_factor: float = 0.5


@dataclass(frozen=True)
class TuningStage:
    """One block of the ABMIL grid search: every lr crossed with every (attn, fc) size."""

    lrs: tuple[float, ...]
    sizes: tuple[tuple[int, int], ...]
    num_epochs: int
    patience: int


@dataclass(frozen=True)
class ThresholdSettings:
    """RET decision thresholds and the grids they are searched over on validation."""

    abmil: float
    baseline: float
    abmil_grid: tuple[float, float, float] = (0.01, 0.55, 0.01)
    fusion_grid: tuple[float, float, float] = (0.01, 1.0, 0.01)
    fusion: float | None = None


@dataclass(frozen=True)
class FusionSettings:
    lr_grid: tuple[float, ...]
    fc_dim_grid: tuple[int, ...]
    dropout_extension: tuple[float, ...]
    batch_size: int = 16
    eval_batch_size: int = 32
    weight_decay: float = 1e-3
    num_epochs: int = 100
    patience: int = 10
    selected: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    tasks: dict[str, TaskSettings]
    training: TrainingSettings
    tuning: tuple[TuningStage, ...]
    thresholds: ThresholdSettings
    fusion: FusionSettings
    ret_class_counts: tuple[int, int] = (319, 21)
    tuning_extension: tuple[TuningStage, ...] = ()
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 42
    baseline_max_iter: int = 2000

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        data = _read_yaml(path)
        return cls(
            seed=int(data["seed"]),
            tasks={name: TaskSettings(**t) for name, t in data["tasks"].items()},
            training=TrainingSettings(**data.get("training", {})),
            tuning=_stages(data["tuning"]),
            tuning_extension=_stages(data.get("tuning_extension") or []),
            ret_class_counts=tuple(data.get("ret_class_counts", (319, 21))),
            thresholds=ThresholdSettings(
                abmil=float(data["thresholds"]["abmil"]),
                baseline=float(data["thresholds"]["baseline"]),
                fusion=data["thresholds"].get("fusion"),
                abmil_grid=_grid(data["thresholds"].get("abmil_grid", (0.01, 0.55, 0.01))),
                fusion_grid=_grid(data["thresholds"].get("fusion_grid", (0.01, 1.0, 0.01))),
            ),
            fusion=FusionSettings(
                lr_grid=tuple(float(x) for x in data["fusion"]["lr_grid"]),
                fc_dim_grid=tuple(int(x) for x in data["fusion"]["fc_dim_grid"]),
                dropout_extension=tuple(float(x) for x in data["fusion"]["dropout_extension"]),
                **{
                    k: v
                    for k, v in data["fusion"].items()
                    if k not in ("lr_grid", "fc_dim_grid", "dropout_extension", "selected")
                },
                selected={
                    task: dict(values)
                    for task, values in (data["fusion"].get("selected") or {}).items()
                },
            ),
            bootstrap_resamples=int(data.get("bootstrap", {}).get("resamples", 2000)),
            bootstrap_seed=int(data.get("bootstrap", {}).get("seed", 42)),
            baseline_max_iter=int(data.get("baseline", {}).get("max_iter", 2000)),
        )

    def task(self, name: str) -> TaskSettings:
        if name not in self.tasks:
            raise ValueError(f"No settings for task '{name}' in the experiment config")
        return self.tasks[name]


def _grid(values: Any) -> tuple[float, float, float]:
    start, stop, step = (float(v) for v in values)
    return start, stop, step


def _stages(items: list[dict[str, Any]]) -> tuple[TuningStage, ...]:
    return tuple(
        TuningStage(
            lrs=tuple(float(x) for x in s["lrs"]),
            sizes=tuple((int(a), int(f)) for a, f in s["sizes"]),
            num_epochs=int(s["num_epochs"]),
            patience=int(s["patience"]),
        )
        for s in items
    )


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path
