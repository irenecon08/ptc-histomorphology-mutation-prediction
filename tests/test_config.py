"""Tests for loading the paths and experiment configuration files."""

from pathlib import Path

import pytest

from ptc_mil.config import ExperimentConfig, Paths

REPO = Path(__file__).resolve().parents[1]


def test_experiment_config_holds_dissertation_values() -> None:
    config = ExperimentConfig.from_yaml(REPO / "configs" / "experiment.yaml")
    assert config.task("multiclass").lr == 1e-4
    assert (config.task("ret_binary").attn_dim, config.task("ret_binary").fc_dim) == (512, 256)
    assert config.thresholds.abmil == 0.13
    assert config.thresholds.baseline == 0.11
    assert sum(len(s.lrs) * len(s.sizes) for s in config.tuning) == 11
    assert config.bootstrap_resamples == 2000


def test_paths_resolve_relative_to_root(tmp_path: Path) -> None:
    yaml_file = tmp_path / "paths.yaml"
    yaml_file.write_text(
        f"root: {tmp_path}\nslides_dir: slides\ntiles_dir: /abs/tiles\nembeddings_dir: emb\n"
        "labels_csv: labels.csv\nsplits_csv: splits.csv\nresults_dir: results\n"
    )
    paths = Paths.from_yaml(yaml_file)
    assert paths.slides_dir == tmp_path / "slides"
    assert paths.tiles_dir == Path("/abs/tiles")
    assert paths.clinical_tsv is None
    with pytest.raises(ValueError, match="clinical_tsv"):
        paths.require("clinical_tsv")


def test_unknown_path_key_is_an_error(tmp_path: Path) -> None:
    yaml_file = tmp_path / "paths.yaml"
    yaml_file.write_text("root: /x\nslide_dir: typo\n")
    with pytest.raises(ValueError, match="slide_dir"):
        Paths.from_yaml(yaml_file)
