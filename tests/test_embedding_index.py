"""Tests for the patient-to-embedding-file index."""

import logging
from pathlib import Path

import pytest

from ptc_mil.data.embedding_index import EmbeddingIndex, patient_id_from_filename


def test_patient_id_from_filename() -> None:
    assert patient_id_from_filename("TCGA-EL-A3H2-01Z-00-DX1.abc123.h5") == "TCGA-EL-A3H2"


def test_index_groups_slides_by_patient(embeddings_dir: Path) -> None:
    index = EmbeddingIndex(embeddings_dir)
    assert len(index) == 4
    assert "TCGA-AA-0001" in index
    assert index.patients_with_multiple_slides == ["TCGA-AA-0002"]


def test_multi_slide_patient_uses_alphabetically_first_slide(embeddings_dir: Path) -> None:
    chosen = EmbeddingIndex(embeddings_dir).slide_for("TCGA-AA-0002")
    assert chosen.name == "TCGA-AA-0002-01Z-00-DX1.h5"


def test_multi_slide_patients_are_logged(
    embeddings_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        EmbeddingIndex(embeddings_dir)
    assert "TCGA-AA-0002" in caplog.text


def test_missing_patient_raises(embeddings_dir: Path) -> None:
    with pytest.raises(KeyError, match="TCGA-AA-0099"):
        EmbeddingIndex(embeddings_dir).slide_for("TCGA-AA-0099")


def test_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        EmbeddingIndex(tmp_path / "does_not_exist")
