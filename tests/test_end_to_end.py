"""Run the whole pipeline on a small synthetic dataset, from split to paired comparison.

This checks that the steps fit together and write the files the next step reads.
Slides are fake: tile embeddings are random with a class-dependent shift, so the
models have something to learn. Tiling and UNI2-h extraction are not run (they need
real slides); empty tile files stand in for the tiling output.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from ptc_mil.cli import main

EMBED_DIM = 16
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
        shift = np.zeros(EMBED_DIM)
        shift[LABELS.index(label)] = 1.5
        shift[5] = 2.0 * ret
        features = rng.normal(size=(20, EMBED_DIM)) + shift
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


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
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
