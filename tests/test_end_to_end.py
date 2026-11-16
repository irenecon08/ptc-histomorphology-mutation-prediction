"""Run the whole pipeline on a small synthetic dataset, from split to paired comparison.

The ``project`` fixture (tests/conftest.py) runs every step after tiling on fake
slides whose tile embeddings are random with a class-dependent shift. These tests
check that the steps fit together and write the files the next step reads.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_split_holds_out_site_el(project: Path) -> None:
    splits = pd.read_csv(project.parent / "splits.csv")
    assert set(splits.loc[splits["split"] == "test", "site"]) == {"EL"}


@pytest.mark.parametrize("model", ["baseline", "abmil", "fusion"])
@pytest.mark.parametrize("task", ["multiclass", "ret_binary"])
def test_every_model_has_test_predictions_and_metrics(project: Path, model: str, task: str) -> None:
    preds = pd.read_csv(project / "predictions" / f"{model}_{task}_test.csv")
    prob_columns = [c for c in preds.columns if c.startswith("prob_")]
    np.testing.assert_allclose(preds[prob_columns].sum(axis=1), 1.0, rtol=1e-5)
    assert (project / "metrics" / f"{model}_{task}.json").is_file()


def test_comparisons_written(project: Path) -> None:
    for other in ("fusion", "baseline"):
        for task in ("multiclass", "ret_binary"):
            assert (project / "comparisons" / f"{other}_vs_abmil_{task}.json").is_file()


def test_tuning_table_and_checkpoints(project: Path) -> None:
    assert (project / "tuning" / "multiclass" / "tuning_multiclass.json").is_file()
    assert (project / "checkpoints" / "best_ret_binary.pt").is_file()
