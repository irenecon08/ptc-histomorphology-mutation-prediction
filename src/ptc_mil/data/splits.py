"""Institutional train/validation/test split.

All patients from tissue source site EL (MD Anderson) form the held-out test set, so
test performance reflects transfer to an institution not seen in training. The
remaining patients are split 80:20 into training and validation at random.
"""

import logging
from pathlib import Path

import pandas as pd

from ptc_mil.data.embedding_index import PatientFileIndex

logger = logging.getLogger(__name__)

TEST_SITE = "EL"
VAL_FRACTION = 0.2
SPLIT_SEED = 42

UNLABELLED_PATIENTS = (
    "TCGA-BJ-A0Z5",
    "TCGA-BJ-A190",
    "TCGA-BJ-A28W",
    "TCGA-BJ-A291",
    "TCGA-BJ-A45C",
    "TCGA-EL-A3ZM",
    "TCGA-ET-A25L",
    "TCGA-ET-A25M",
    "TCGA-ET-A25N",
    "TCGA-ET-A25P",
    "TCGA-FE-A238",
    "TCGA-FE-A239",
    "TCGA-FY-A4B0",
    "TCGA-IM-A41Z",
)
"""Patients with no definitive molecular call; assigned to Other (see the dissertation, §3.2)."""


def site_code(patient_id: str) -> str:
    """Tissue source site code from a TCGA barcode, e.g. 'EL' from 'TCGA-EL-A3H2'."""
    return patient_id[5:7]


def make_splits(
    labels: pd.DataFrame,
    tiled_patients: set[str],
    test_site: str = TEST_SITE,
    val_fraction: float = VAL_FRACTION,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """Build the splits table from the patient label table.

    Args:
        labels: One row per patient with ``patient``, ``BRAF_V600E``, ``RAS``, ``RET``
            and ``mutation_label`` columns.
        tiled_patients: Patients with at least one tiled slide; others are dropped.
        test_site: Site code held out as the test set.
        val_fraction: Fraction of the remaining patients used for validation.
        seed: Random seed for the train/validation split.

    Returns:
        The label table restricted to tiled patients, with ``site`` and ``split`` columns.
    """
    unlabelled = pd.DataFrame(
        {
            "patient": list(UNLABELLED_PATIENTS),
            "BRAF_V600E": 0,
            "RAS": 0,
            "RET": 0,
            "mutation_label": "Other",
        }
    )
    table = pd.concat([labels, unlabelled], ignore_index=True)
    table = table.drop_duplicates(subset=["patient"])
    table = table[table["patient"].isin(tiled_patients)].copy()

    table["site"] = table["patient"].map(site_code)
    table["split"] = "train"
    table.loc[table["site"] == test_site, "split"] = "test"

    train_val = table[table["split"] == "train"]
    val_index = train_val.sample(frac=val_fraction, random_state=seed).index
    table.loc[val_index, "split"] = "val"

    logger.info("Split sizes: %s", table["split"].value_counts().to_dict())
    return table


def create_splits_file(labels_csv: Path, tiles_dir: Path, output_csv: Path) -> pd.DataFrame:
    """Read the labels, find which patients were tiled, and write the splits CSV."""
    labels = pd.read_csv(labels_csv)
    tiled = set(PatientFileIndex(tiles_dir, ".h5").patients)
    splits = make_splits(labels, tiled)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(output_csv, index=False)
    logger.info("Saved %s", output_csv)
    logger.info(
        "Mutation labels per split:\n%s",
        splits.groupby(["split", "mutation_label"]).size().unstack(fill_value=0),
    )
    logger.info("RET-positive per split: %s", splits.groupby("split")["RET"].sum().to_dict())
    return splits
