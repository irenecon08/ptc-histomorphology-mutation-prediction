"""Clinical feature matrix for the multimodal fusion model.

Nine clinical variables are used for both tasks: Diagnosis Age, Sex, Race Category,
Ethnicity Category, Overall Disease Stage, T Stage, N Stage, M Stage and Histological
Subtype. Molecular-burden, hypoxia, survival, treatment and last-follow-up variables
are excluded (dissertation §3.13.1).

Every statistic used for imputation or scaling (Age mean and standard deviation, the
mode of each other variable) comes from the training split only, and a missingness
indicator is added for every variable.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OVERALL_STAGE_COLUMN = "Neoplasm Disease Stage American Joint Committee on Cancer Code"
T_STAGE_COLUMN = "American Joint Committee on Cancer Tumor Stage Code"
N_STAGE_COLUMN = "Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code"
M_STAGE_COLUMN = "American Joint Committee on Cancer Metastasis Stage Code"

_OVERALL_STAGES = {
    "STAGE I": 1,
    "STAGE II": 2,
    "STAGE III": 3,
    "STAGE IVA": 4,
    "STAGE IVB": 4,
    "STAGE IVC": 4,
    "STAGE IV": 4,
}


def map_overall_stage(value: object) -> float:
    if pd.isna(value):
        return np.nan
    return _OVERALL_STAGES.get(str(value).upper(), np.nan)


def _prefix_rank(value: object, unknown: str, ranks: dict[str, int]) -> float:
    """Rank an AJCC code by its prefix (sub-stage letters collapse to the parent stage)."""
    if pd.isna(value):
        return np.nan
    code = str(value).upper()
    if code.startswith(unknown):
        return np.nan
    for prefix, rank in ranks.items():
        if code.startswith(prefix):
            return rank
    return np.nan


def map_t_stage(value: object) -> float:
    return _prefix_rank(value, "TX", {"T1": 1, "T2": 2, "T3": 3, "T4": 4})


def map_n_stage(value: object) -> float:
    return _prefix_rank(value, "NX", {"N0": 0, "N1": 1})


def map_m_stage(value: object) -> float:
    return _prefix_rank(value, "MX", {"M0": 0, "M1": 1})


ORDINAL_VARIABLES: tuple[tuple[str, str, Callable[[object], float]], ...] = (
    ("overall_stage", OVERALL_STAGE_COLUMN, map_overall_stage),
    ("t_stage", T_STAGE_COLUMN, map_t_stage),
    ("n_stage", N_STAGE_COLUMN, map_n_stage),
    ("m_stage", M_STAGE_COLUMN, map_m_stage),
)
NOMINAL_VARIABLES = (
    ("sex", "Sex"),
    ("race", "Race Category"),
    ("ethnicity", "Ethnicity Category"),
    ("subtype", "Tumor Type"),
)


def load_and_merge(labels_csv: Path, clinical_tsv: Path, splits_csv: Path) -> pd.DataFrame:
    """Join the label table with cBioPortal clinical data and the split assignment."""
    labels = pd.read_csv(labels_csv)
    clinical = pd.read_csv(clinical_tsv, sep="\t").replace("NA", np.nan)
    if "Sample Type" in clinical.columns:
        clinical = clinical.sort_values("Sample Type", key=lambda s: s != "Primary")
    clinical = clinical.drop_duplicates(subset="Patient ID", keep="first")
    splits = pd.read_csv(splits_csv)
    merged = labels.merge(clinical, left_on="patient", right_on="Patient ID", how="left")
    return merged.merge(splits[["patient", "split"]], on="patient", how="left")


def build_clinical_features(df: pd.DataFrame, train_split: str = "train") -> pd.DataFrame:
    """Encode the nine clinical variables, using training-split statistics only."""
    train = df["split"] == train_split
    out = pd.DataFrame({"patient": df["patient"], "split": df["split"]})

    age = pd.to_numeric(df["Diagnosis Age"], errors="coerce")
    age_mean, age_std = age[train].mean(), age[train].std()
    out["age_missing"] = age.isna().astype(int)
    out["age_z"] = (age.fillna(age_mean) - age_mean) / age_std

    for name, column, mapper in ORDINAL_VARIABLES:
        mapped = df[column].map(mapper)
        mode = mapped[train].mode().iloc[0]
        out[f"{name}_missing"] = mapped.isna().astype(int)
        out[name] = mapped.fillna(mode)

    for prefix, column in NOMINAL_VARIABLES:
        values = df[column]
        mode = values[train].mode().iloc[0]
        out[f"{prefix}_missing"] = values.isna().astype(int)
        out = pd.concat([out, pd.get_dummies(values.fillna(mode), prefix=prefix)], axis=1)

    return out


def create_clinical_features_file(
    labels_csv: Path, clinical_tsv: Path, splits_csv: Path, output_csv: Path
) -> pd.DataFrame:
    merged = load_and_merge(labels_csv, clinical_tsv, splits_csv)
    logger.info("Merged clinical data for %d patients", len(merged))
    features = build_clinical_features(merged)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False)
    feature_columns = [c for c in features.columns if c not in ("patient", "split")]
    logger.info("Saved %d clinical feature columns to %s", len(feature_columns), output_csv)
    for column in (c for c in features.columns if c.endswith("_missing")):
        logger.info("  %s: %.1f%% missing", column, 100 * features[column].mean())
    return features


def load_clinical_features(
    path: Path, exclude_columns: list[str] | None = None
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Read the clinical feature matrix as {patient: feature vector} plus the column names.

    Args:
        path: CSV written by :func:`create_clinical_features_file`.
        exclude_columns: Columns to leave out, e.g. ``["race_missing", "ethnicity_missing"]``
            for the site-confound check (those flags track the institutional split).
    """
    df = pd.read_csv(path)
    columns = [c for c in df.columns if c not in ("patient", "split")]
    if exclude_columns:
        unknown = set(exclude_columns) - set(columns)
        if unknown:
            logger.warning("Columns to exclude not found, ignoring: %s", sorted(unknown))
        columns = [c for c in columns if c not in exclude_columns]
    lookup = {row["patient"]: row[columns].to_numpy(dtype=np.float32) for _, row in df.iterrows()}
    return lookup, columns
