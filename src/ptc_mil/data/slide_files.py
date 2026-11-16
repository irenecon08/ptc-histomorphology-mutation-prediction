"""Look up the whole-slide image, tile coordinates and embeddings of one patient together.

The embedding index decides which slide a patient uses; the tile file and .svs file
are then found by that slide's name, so all three always refer to the same slide.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from ptc_mil.config import Paths
from ptc_mil.data.embedding_index import EmbeddingIndex, PatientFileIndex, slide_name


@dataclass(frozen=True)
class TileGrid:
    """Tile coordinates of one slide, as written by the tiling step.

    Attributes:
        coords: (n_tiles, 2) top-left (x, y) of each tile, in level-``level`` pixels.
        level: OpenSlide pyramid level the tiles were read from.
        tile_size: Tile width and height in pixels at that level.
        level_downsample: The slide's actual downsample factor at that level.
    """

    coords: np.ndarray
    level: int
    tile_size: int
    level_downsample: float

    def level0_origin(self, i: int) -> tuple[int, int]:
        """Top-left corner of tile ``i`` in level-0 (full resolution) pixels."""
        x, y = self.coords[i]
        return int(round(x * self.level_downsample)), int(round(y * self.level_downsample))

    @property
    def level0_tile_size(self) -> int:
        """Side length of one tile in level-0 pixels."""
        return int(round(self.tile_size * self.level_downsample))


def read_tile_grid(path: str | Path) -> TileGrid:
    with h5py.File(path, "r") as f:
        return TileGrid(
            coords=f["coords"][:],
            level=int(f.attrs["level"]),
            tile_size=int(f.attrs["tile_size"]),
            level_downsample=float(f.attrs["level_downsample"]),
        )


class SlideFiles:
    """Finds a patient's embedding, tile and .svs files, consistently for the same slide.

    Args:
        paths: Project paths (embeddings, tiles and slides folders are used).
    """

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.embeddings = EmbeddingIndex(paths.embeddings_dir)
        self._svs: PatientFileIndex | None = None

    def embedding(self, patient_id: str) -> Path:
        return self.embeddings.slide_for(patient_id)

    def name(self, patient_id: str) -> str:
        """Name of the slide used for this patient."""
        return slide_name(self.embedding(patient_id))

    def tiles(self, patient_id: str) -> Path:
        path = self.paths.tiles_dir / f"{self.name(patient_id)}.h5"
        if not path.is_file():
            raise FileNotFoundError(f"No tile file for {patient_id}: {path}")
        return path

    def tile_grid(self, patient_id: str) -> TileGrid:
        return read_tile_grid(self.tiles(patient_id))

    def svs(self, patient_id: str) -> Path:
        if self._svs is None:
            self._svs = PatientFileIndex(self.paths.slides_dir, ".svs", recursive=True)
        wanted = f"{self.name(patient_id)}.svs"
        for path in self._svs.all_files(patient_id):
            if path.name == wanted:
                return path
        raise FileNotFoundError(f"No .svs file named {wanted} under {self.paths.slides_dir}")
