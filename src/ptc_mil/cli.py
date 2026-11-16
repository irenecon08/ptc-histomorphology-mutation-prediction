"""Command-line entry point: ``ptc-mil <step> [options]``.

Run the steps in this order to reproduce the dissertation (see the README):

    split → tile → embed → baseline → tune → train → predict → calibrate → evaluate
    → clinical → fusion-search → fusion-train → calibrate --model fusion → evaluate → compare
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ptc_mil import __version__
from ptc_mil.config import ExperimentConfig, Paths
from ptc_mil.data.labels import TASKS, get_task
from ptc_mil.utils.logging import setup_logging
from ptc_mil.utils.runtime import default_device

if TYPE_CHECKING:
    from ptc_mil.pipeline import Experiment

logger = logging.getLogger("ptc_mil")

DEFAULT_PATHS = Path("configs/paths.yaml")
DEFAULT_EXPERIMENT = Path("configs/experiment.yaml")


def _tasks(args: argparse.Namespace) -> list[str]:
    return list(TASKS) if args.task == "all" else [args.task]


def _experiment(args: argparse.Namespace) -> Experiment:
    from ptc_mil.pipeline import Experiment

    return Experiment(
        Paths.from_yaml(args.paths), ExperimentConfig.from_yaml(args.experiment), default_device()
    )


def cmd_split(args: argparse.Namespace) -> None:
    from ptc_mil.data.splits import create_splits_file

    paths = Paths.from_yaml(args.paths)
    create_splits_file(paths.labels_csv, paths.tiles_dir, paths.splits_csv)


def cmd_tile(args: argparse.Namespace) -> None:
    from ptc_mil.preprocessing.tiling import tile_all

    paths = Paths.from_yaml(args.paths)
    tile_all(paths.slides_dir, paths.tiles_dir)


def cmd_embed(args: argparse.Namespace) -> None:
    from ptc_mil.preprocessing.embeddings import embed_all

    paths = Paths.from_yaml(args.paths)
    embed_all(
        paths.slides_dir,
        paths.tiles_dir,
        paths.embeddings_dir,
        default_device(),
        paths.hf_token_file,
    )


def cmd_clinical(args: argparse.Namespace) -> None:
    from ptc_mil.data.clinical import create_clinical_features_file

    paths = Paths.from_yaml(args.paths)
    create_clinical_features_file(
        paths.labels_csv,
        paths.require("clinical_tsv"),
        paths.splits_csv,
        paths.require("clinical_features_csv"),
    )


def cmd_baseline(args: argparse.Namespace) -> None:
    from ptc_mil.pipeline import run_baseline

    exp = _experiment(args)
    for name in _tasks(args):
        run_baseline(exp, get_task(name))


def cmd_tune(args: argparse.Namespace) -> None:
    from ptc_mil.pipeline import run_tuning

    exp = _experiment(args)
    for name in _tasks(args):
        run_tuning(exp, get_task(name), extended=args.extended)


def cmd_train(args: argparse.Namespace) -> None:
    from ptc_mil.pipeline import run_training

    exp = _experiment(args)
    for name in _tasks(args):
        run_training(exp, get_task(name))


def cmd_predict(args: argparse.Namespace) -> None:
    from ptc_mil.pipeline import run_abmil_predictions

    exp = _experiment(args)
    for name in _tasks(args):
        run_abmil_predictions(exp, get_task(name))


def cmd_calibrate(args: argparse.Namespace) -> None:
    from ptc_mil.pipeline import run_calibration

    threshold = run_calibration(_experiment(args), args.model)
    logger.info("%s RET threshold selected on validation: %.2f", args.model, threshold)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from ptc_mil.pipeline import run_evaluation

    exp = _experiment(args)
    for name in _tasks(args):
        run_evaluation(exp, args.model, get_task(name))


def cmd_compare(args: argparse.Namespace) -> None:
    from ptc_mil.pipeline import run_comparison

    exp = _experiment(args)
    for name in _tasks(args):
        run_comparison(exp, get_task(name), args.reference, args.other)


def cmd_fusion_search(args: argparse.Namespace) -> None:
    from ptc_mil.pipeline import run_fusion_search

    exp = _experiment(args)
    for name in _tasks(args):
        run_fusion_search(exp, get_task(name))


def cmd_fusion_train(args: argparse.Namespace) -> None:
    from ptc_mil.pipeline import run_fusion_training

    exp = _experiment(args)
    exclude = args.exclude_clinical.split(",") if args.exclude_clinical else None
    for name in _tasks(args):
        run_fusion_training(exp, get_task(name), exclude)


COMMANDS: dict[str, tuple[Callable[[argparse.Namespace], None], str]] = {
    "split": (cmd_split, "Create the institutional train/val/test split"),
    "tile": (cmd_tile, "Tile every slide and drop background tiles"),
    "embed": (cmd_embed, "Extract UNI2-h embeddings for every tile"),
    "clinical": (cmd_clinical, "Build the clinical feature matrix for fusion"),
    "baseline": (cmd_baseline, "Fit the mean-pooling logistic regression baseline"),
    "tune": (cmd_tune, "ABMIL grid search on validation AUROC"),
    "train": (cmd_train, "Train the final ABMIL model per task"),
    "predict": (cmd_predict, "Save ABMIL validation and test predictions"),
    "calibrate": (cmd_calibrate, "Select the RET threshold on validation"),
    "evaluate": (cmd_evaluate, "Test metrics with bootstrap 95%% CIs"),
    "compare": (cmd_compare, "Paired bootstrap difference between two models"),
    "fusion-search": (cmd_fusion_search, "Fusion head hyperparameter search"),
    "fusion-train": (cmd_fusion_train, "Train the final fusion head"),
}

TASK_STEPS = {
    "baseline",
    "tune",
    "train",
    "predict",
    "evaluate",
    "compare",
    "fusion-search",
    "fusion-train",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptc-mil",
        description="Driver-alteration prediction from H&E whole-slide images.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--paths", type=Path, default=DEFAULT_PATHS, help="paths YAML (default: %(default)s)"
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=DEFAULT_EXPERIMENT,
        help="experiment YAML (default: %(default)s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log per-epoch progress")
    sub = parser.add_subparsers(dest="command", required=True, metavar="STEP")

    for name, (func, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        if name in TASK_STEPS:
            p.add_argument("--task", choices=[*TASKS, "all"], default="all")

    sub.choices["tune"].add_argument(
        "--extended", action="store_true", help="also run the extra small-LR stage (13 configs)"
    )
    sub.choices["calibrate"].add_argument(
        "--model", default="abmil", help="baseline, abmil or fusion"
    )
    sub.choices["evaluate"].add_argument(
        "--model", default="abmil", help="baseline, abmil or fusion"
    )
    sub.choices["compare"].add_argument("--reference", default="abmil")
    sub.choices["compare"].add_argument("--other", required=True, help="e.g. fusion or baseline")
    sub.choices["fusion-train"].add_argument(
        "--exclude-clinical",
        help="comma-separated clinical columns to drop (site-confound check), "
        "e.g. race_missing,ethnicity_missing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
