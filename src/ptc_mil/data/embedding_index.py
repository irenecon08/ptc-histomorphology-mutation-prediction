"""Find each patient's embedding file once, with a fixed rule for multi-slide patients.

Embedding files are named after the TCGA slide, e.g. TCGA-EL-A3H2-01Z-00-DX1.<id>.h5.
The first 12 characters are the patient ID (TCGA-EL-A3H2). One slide is used per
patient; for patients with more than one slide, the alphabetically first is used.
"""

import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

TCGA_PATIENT_ID_LENGTH = 12


def patient_id_from_filename(filename: str) -> str:
    """Return the TCGA patient ID at the start of a slide or embedding file name."""
    return filename[:TCGA_PATIENT_ID_LENGTH]


class EmbeddingIndex:
    """Maps each patient to the embedding file of their slide.

    The folder is scanned once, when the index is created.

    Args:
        embeddings_dir: Folder with one .h5 embedding file per slide.

    Raises:
        FileNotFoundError: If embeddings_dir does not exist.
    """

    def __init__(self, embeddings_dir: str | Path) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        if not self.embeddings_dir.is_dir():
            raise FileNotFoundError(f"Embeddings folder not found: {self.embeddings_dir}")

        slides: dict[str, list[Path]] = defaultdict(list)
        for path in sorted(self.embeddings_dir.glob("*.h5")):
            slides[patient_id_from_filename(path.name)].append(path)
        self._slides = dict(slides)

        if self.patients_with_multiple_slides:
            logger.warning(
                "%d patient(s) have more than one slide; using the alphabetically first: %s",
                len(self.patients_with_multiple_slides),
                self.patients_with_multiple_slides,
            )

    def __len__(self) -> int:
        """Number of patients with at least one embedding file."""
        return len(self._slides)

    def __contains__(self, patient_id: str) -> bool:
        return patient_id in self._slides

    @property
    def patients_with_multiple_slides(self) -> list[str]:
        return sorted(p for p, paths in self._slides.items() if len(paths) > 1)

    def slide_for(self, patient_id: str) -> Path:
        """Return the embedding file to use for a patient.

        Raises:
            KeyError: If the patient has no embedding file.
        """
        if patient_id not in self._slides:
            raise KeyError(f"No embedding file for patient {patient_id} in {self.embeddings_dir}")
        return self._slides[patient_id][0]
