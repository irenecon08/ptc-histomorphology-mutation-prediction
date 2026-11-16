"""Shared test fixtures: a small synthetic dataset shaped like the real one.

No TCGA data is needed. Each fake slide is an .h5 file of random tile embeddings,
named like a real TCGA slide, so the patient ID is its first 12 characters.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

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
