"""Tests for the institutional train/validation/test split."""

import pandas as pd

from ptc_mil.data.splits import UNLABELLED_PATIENTS, make_splits, site_code


def _labels(patients: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"patient": patients, "BRAF_V600E": 1, "RAS": 0, "RET": 0, "mutation_label": "BRAF_V600E"}
    )


def test_site_code() -> None:
    assert site_code("TCGA-EL-A3H2") == "EL"


def test_site_el_is_test_and_rest_split_80_20() -> None:
    patients = [f"TCGA-EL-{i:04d}" for i in range(10)] + [f"TCGA-BJ-{i:04d}" for i in range(50)]
    splits = make_splits(_labels(patients), set(patients))
    assert set(splits.loc[splits["split"] == "test", "site"]) == {"EL"}
    assert (splits["split"] == "test").sum() == 10
    assert (splits["split"] == "val").sum() == 10
    assert (splits["split"] == "train").sum() == 40


def test_split_is_reproducible() -> None:
    patients = [f"TCGA-BJ-{i:04d}" for i in range(30)]
    a = make_splits(_labels(patients), set(patients))
    b = make_splits(_labels(patients), set(patients))
    pd.testing.assert_frame_equal(a, b)


def test_untiled_patients_dropped_and_unlabelled_added_as_other() -> None:
    patients = ["TCGA-BJ-0001", "TCGA-BJ-0002"]
    tiled = {"TCGA-BJ-0001", UNLABELLED_PATIENTS[0]}
    splits = make_splits(_labels(patients), tiled)
    assert set(splits["patient"]) == tiled
    assert splits.set_index("patient").loc[UNLABELLED_PATIENTS[0], "mutation_label"] == "Other"
