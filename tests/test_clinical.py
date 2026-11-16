"""Tests for clinical feature encoding."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from ptc_mil.data import clinical
from ptc_mil.data.clinical import build_clinical_features


@pytest.mark.parametrize(
    ("mapper", "value", "expected"),
    [
        (clinical.map_overall_stage, "Stage IVA", 4),
        (clinical.map_t_stage, "T3b", 3),
        (clinical.map_t_stage, "TX", np.nan),
        (clinical.map_n_stage, "N1a", 1),
        (clinical.map_m_stage, "M0", 0),
        (clinical.map_m_stage, None, np.nan),
    ],
)
def test_stage_maps(mapper: Callable[[object], float], value: object, expected: float) -> None:
    result = mapper(value)
    assert (np.isnan(result) and np.isnan(expected)) or result == expected


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient": ["P1", "P2", "P3", "P4"],
            "split": ["train", "train", "train", "test"],
            "Diagnosis Age": [30, 50, None, 90],
            clinical.OVERALL_STAGE_COLUMN: ["Stage I", "Stage I", "Stage III", None],
            clinical.T_STAGE_COLUMN: ["T1", "T2", "T2", "T4"],
            clinical.N_STAGE_COLUMN: ["N0", "N1", "NX", "N0"],
            clinical.M_STAGE_COLUMN: ["M0", "M0", "M0", "M1"],
            "Sex": ["Female", "Female", "Male", None],
            "Race Category": ["White", None, "White", "Asian"],
            "Ethnicity Category": ["Not Hispanic Or Latino"] * 4,
            "Tumor Type": ["Classical", "Classical", "Follicular", "Tall Cell"],
        }
    )


def test_imputation_and_scaling_use_training_statistics_only() -> None:
    out = build_clinical_features(_frame()).set_index("patient")
    # Age: train mean 40 and std of [30, 50]; P3 is missing and imputed to the mean
    assert out.loc["P3", "age_z"] == 0.0
    assert out.loc["P3", "age_missing"] == 1
    assert out.loc["P4", "age_z"] == pytest.approx((90 - 40) / np.std([30, 50], ddof=1))
    # Missing overall stage on a test patient is imputed with the training mode (1)
    assert out.loc["P4", "overall_stage"] == 1
    assert out.loc["P4", "sex_Female"]
    assert out.loc["P4", "sex_missing"] == 1
