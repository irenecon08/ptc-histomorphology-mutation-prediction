"""Tests for the prediction task definitions."""

import dataclasses

import pandas as pd
import pytest

from ptc_mil.data.labels import MULTICLASS, RET_BINARY, get_task


def test_multiclass_encoding_matches_dissertation() -> None:
    encoded = MULTICLASS.encode(pd.Series(["BRAF_V600E", "RAS", "Other"]))
    assert encoded.tolist() == [0, 1, 2]


def test_ret_encoding() -> None:
    assert RET_BINARY.encode(pd.Series([0, 1, 1])).tolist() == [0, 1, 1]


def test_num_classes() -> None:
    assert MULTICLASS.num_classes == 3
    assert RET_BINARY.num_classes == 2


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError, match="unknown labels"):
        MULTICLASS.encode(pd.Series(["BRAF_V600E", "BRAF"]))


def test_get_task_returns_task_by_name() -> None:
    assert get_task("multiclass") is MULTICLASS
    assert get_task("ret_binary") is RET_BINARY


def test_get_task_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        get_task("braf")


def test_tasks_cannot_be_modified() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        MULTICLASS.label_column = "RET"  # type: ignore[misc]
        