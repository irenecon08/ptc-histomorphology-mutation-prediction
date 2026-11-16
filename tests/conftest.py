"""Shared test fixtures: small synthetic datasets shaped like the real one.

No TCGA data is needed. Each fake slide is an .h5 file of random tile embeddings,
named like a real TCGA slide, so the patient ID is its first 12 characters.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from ptc_mil.cli import main

EMBED_DIM = 16  # small stand-in for UNI2-h's 1536, so tests run fast


def write_fake_slide(folder: Path, slide_name: str, n_tiles: int, seed: int) -> Path:
    """Write an .h5 file with a 'features' dataset of shape (n_tiles, EMBED_DIM)."""
    rng = np.random.default_rng(seed)
    path = folder / f"{slide_name}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=rng.normal(size=(n_tiles, EMBED_DIM)).astype("float32"))
        f.create_dataset("coords", data=np.arange(n_tiles * 2).reshape(n_tiles, 2) * 256)
    return path


@pytest.fixture
def embeddings_dir(tmp_path: Path) -> Path:
    """Four patients with embeddings; TCGA-AA-0002 has two slides."""
    folder = tmp_path / "embeddings"
    folder.mkdir()
    slides = [
        "TCGA-AA-0001-01Z-00-DX1",
        "TCGA-AA-0002-01Z-00-DX2",
        "TCGA-AA-0002-01Z-00-DX1",
        "TCGA-AA-0003-01Z-00-DX1",
        "TCGA-AA-0004-01Z-00-DX1",
    ]
    for seed, name in enumerate(slides):
        write_fake_slide(folder, name, n_tiles=10 + seed, seed=seed)
    return folder


# --- Full pipeline on synthetic data (used by the end-to-end and analysis tests) ---

PIPELINE_EMBED_DIM = 16
LABELS = ("BRAF_V600E", "RAS", "Other")


def _make_project(root: Path) -> Path:
    rng = np.random.default_rng(0)
    (root / "embeddings").mkdir()
    (root / "tiles").mkdir()
    patients = [f"TCGA-EL-{i:04d}" for i in range(24)] + [f"TCGA-BJ-{i:04d}" for i in range(60)]
    rows, clinical = [], []
    for i, patient in enumerate(patients):
        label = LABELS[i % 3]
        ret = int(label == "Other" and i % 2 == 0)
        shift = np.zeros(PIPELINE_EMBED_DIM)
        shift[LABELS.index(label)] = 1.5
        shift[5] = 2.0 * ret
        features = rng.normal(size=(20, PIPELINE_EMBED_DIM)) + shift
        slide = f"{patient}-01Z-00-DX1.fake"
        with h5py.File(root / "embeddings" / f"{slide}.h5", "w") as f:
            f.create_dataset("features", data=features.astype("float32"))
        h5py.File(root / "tiles" / f"{slide}.h5", "w").close()
        rows.append(
            {
                "patient": patient,
                "BRAF_V600E": int(label == "BRAF_V600E"),
                "RAS": int(label == "RAS"),
                "RET": ret,
                "mutation_label": label,
            }
        )
        clinical.append(
            {
                "Patient ID": patient,
                "Sample Type": "Primary",
                "Diagnosis Age": 30 + i % 40,
                "Neoplasm Disease Stage American Joint Committee on Cancer Code": "Stage I",
                "American Joint Committee on Cancer Tumor Stage Code": ["T1", "T2", "T3"][i % 3],
                "Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code": "N0",
                "American Joint Committee on Cancer Metastasis Stage Code": "M0",
                "Sex": ["Female", "Male"][i % 2],
                "Race Category": "White",
                "Ethnicity Category": "Not Hispanic Or Latino",
                "Tumor Type": "Classical",
            }
        )
    pd.DataFrame(rows).to_csv(root / "labels.csv", index=False)
    pd.DataFrame(clinical).to_csv(root / "clinical.tsv", sep="\t", index=False)

    paths = {
        "root": str(root),
        "slides_dir": "slides",
        "tiles_dir": "tiles",
        "embeddings_dir": "embeddings",
        "labels_csv": "labels.csv",
        "splits_csv": "splits.csv",
        "results_dir": "results",
        "clinical_tsv": "clinical.tsv",
        "clinical_features_csv": "results/clinical_features.csv",
    }
    (root / "paths.yaml").write_text(yaml.safe_dump(paths))

    repo = Path(__file__).resolve().parents[1]
    experiment = yaml.safe_load((repo / "configs" / "experiment.yaml").read_text())
    experiment["training"].update(num_epochs=4, patience=2)
    experiment["tuning"] = [{"lrs": [1e-3], "sizes": [[32, 16]], "num_epochs": 3, "patience": 2}]
    for task in experiment["tasks"].values():
        task.update(lr=1e-3, attn_dim=32, fc_dim=16)
    experiment["fusion"].update(
        lr_grid=[1e-3], fc_dim_grid=[16], dropout_extension=[0.2], num_epochs=4, patience=2
    )
    experiment["bootstrap"]["resamples"] = 50
    (root / "experiment.yaml").write_text(yaml.safe_dump(experiment))
    return root


@pytest.fixture(scope="session")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Results folder of a full pipeline run on a small synthetic dataset."""
    root = _make_project(tmp_path_factory.mktemp("project"))
    steps = [
        ["split"],
        ["baseline"],
        ["tune"],
        ["train"],
        ["predict"],
        ["calibrate", "--model", "abmil"],
        ["calibrate", "--model", "baseline"],
        ["evaluate", "--model", "abmil"],
        ["evaluate", "--model", "baseline"],
        ["clinical"],
        ["fusion-search"],
        ["fusion-train"],
        ["calibrate", "--model", "fusion"],
        ["evaluate", "--model", "fusion"],
        ["compare", "--other", "fusion"],
        ["compare", "--other", "baseline"],
    ]
    common = ["--paths", str(root / "paths.yaml"), "--experiment", str(root / "experiment.yaml")]
    for step in steps:
        assert main(common + step) == 0, step
    return root / "results"
