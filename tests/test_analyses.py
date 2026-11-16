"""Smoke-run the analysis scripts that need neither OpenSlide nor HoVer-Net.

Each script runs in a subprocess, as a user would run it, against the outputs of the
synthetic pipeline run (the ``project`` fixture). This checks that the scripts find
their inputs through the shared configuration and write their outputs.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

SCRIPTS = {
    "analyses/evaluation/plot_paired_forest.py": "figures/paired_bootstrap_forest.png",
    "analyses/evaluation/plot_abmil_vs_fusion.py": "figures/fusion_vs_abmil_bootstrap.png",
    "analyses/interpretability/embedding_analysis.py": "embeddings/embedding_coords_ret_binary.csv",
    "analyses/tiic/genotype_stratified_slide_selection.py": (
        "slide_selection/genotype_stratified_slides.csv"
    ),
    "analyses/clinical/eda_fusion_features.py": "clinical_eda/eda_summary.csv",
}


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_analysis_script_runs(project: Path, script: str) -> None:
    pytest.importorskip("matplotlib")
    if "embedding_analysis" in script:
        pytest.importorskip("umap")
    root = project.parent
    env = {
        **os.environ,
        "PTC_MIL_PATHS": str(root / "paths.yaml"),
        "PTC_MIL_EXPERIMENT": str(root / "experiment.yaml"),
        "MPLBACKEND": "Agg",
    }
    result = subprocess.run(
        [sys.executable, str(REPO / script)], cwd=REPO, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert (project / "analyses" / SCRIPTS[script]).is_file()
