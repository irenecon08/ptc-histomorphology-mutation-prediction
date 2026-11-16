"""Find each patient's slide files once, with a fixed rule for multi-slide patients.

Slide, tile and embedding files are named after the TCGA slide, e.g.
TCGA-EL-A3H2-01Z-00-DX1.<id>.h5. The first 12 characters are the patient ID
(TCGA-EL-A3H2). One slide is used per patient; for patients with more than one
slide, the alphabetically first is used. Because the same slide name is used for
the .svs, tile and embedding files, choosing by name also keeps all three consistent.
"""

import logging
from collections import defaultdict
from pathlib import Path

import h5py

logger = logging.getLogger(__name__)

TCGA_PATIENT_ID_LENGTH = 12


def patient_id_from_filename(filename: str) -> str:
    """Return the TCGA patient ID at the start of a slide or embedding file name."""
    return filename[:TCGA_PATIENT_ID_LENGTH]


class PatientFileIndex:
    """Maps each patient to the file(s) of their slide(s) in one folder.

    The folder is scanned once, when the index is created.

    Args:
        folder: Folder to scan.
        suffix: File extension to index, e.g. ".h5" or ".svs".
        recursive: Also scan subfolders (the GDC download stores each .svs in its own folder).

    Raises:
        FileNotFoundError: If the folder does not exist.
    """

    def __init__(self, folder: str | Path, suffix: str, recursive: bool = False) -> None:
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {self.folder}")

        pattern = f"*{suffix}"
        found = self.folder.rglob(pattern) if recursive else self.folder.glob(pattern)
        slides: dict[str, list[Path]] = defaultdict(list)
        for path in sorted(found, key=lambda p: p.name):
            slides[patient_id_from_filename(path.name)].append(path)
        self._slides = dict(slides)

        if self.patients_with_multiple_slides:
            logger.warning(
                "%s: %d patient(s) have more than one slide; using the alphabetically first: %s",
                self.folder,
                len(self.patients_with_multiple_slides),
                self.patients_with_multiple_slides,
            )

    def __len__(self) -> int:
        """Number of patients with at least one file."""
        return len(self._slides)

    def __contains__(self, patient_id: object) -> bool:
        return patient_id in self._slides

    @property
    def patients(self) -> list[str]:
        return sorted(self._slides)

    @property
    def patients_with_multiple_slides(self) -> list[str]:
        return sorted(p for p, paths in self._slides.items() if len(paths) > 1)

    def slide_for(self, patient_id: str) -> Path:
        """Return the file to use for a patient.

        Raises:
            KeyError: If the patient has no file.
        """
        if patient_id not in self._slides:
            raise KeyError(f"No file for patient {patient_id} in {self.folder}")
        return self._slides[patient_id][0]

    def all_files(self, patient_id: str) -> list[Path]:
        """Every file for a patient, sorted by name (empty if there are none)."""
        return list(self._slides.get(patient_id, []))

    def get(self, patient_id: str) -> Path | None:
        """Like :meth:`slide_for`, but return None for a patient with no file."""
        return self.slide_for(patient_id) if patient_id in self else None


class EmbeddingIndex(PatientFileIndex):
    """Maps each patient to the UNI2-h embedding file (.h5) of their slide.

    Args:
        embeddings_dir: Folder with one .h5 embedding file per slide.
    """

    def __init__(self, embeddings_dir: str | Path) -> None:
        super().__init__(embeddings_dir, suffix=".h5")
        self.embeddings_dir = self.folder

    @property
    def embed_dim(self) -> int:
        """Size of one tile embedding, read from the first file (1536 for UNI2-h)."""
        if not self._slides:
            raise ValueError(f"No embedding files in {self.folder}")
        first = self.slide_for(self.patients[0])
        with h5py.File(first, "r") as f:
            return int(f["features"].shape[1])


def slide_name(path: Path) -> str:
    """The slide's name without extension, shared by its .svs, tile and embedding files."""
    return path.name.rsplit(".", 1)[0]
