"""Tile whole-slide images into non-overlapping 256x256 tiles and drop background.

Tiles are read at pyramid level 1 (about 10x magnification). A tile is background
when more than 85% of its pixels have all RGB channels above 220. Only coordinates are
stored; pixels are read again during embedding extraction.

Level coordinates are converted to level-0 coordinates (which OpenSlide's
``read_region`` expects) with the slide's actual downsample factor at that level,
read from the slide itself rather than assumed to be 2**level (dissertation §3.4).
"""

import logging
from pathlib import Path

import h5py
import numpy as np

from ptc_mil.data.embedding_index import slide_name

logger = logging.getLogger(__name__)

TILE_SIZE = 256
LEVEL = 1
WHITE_THRESHOLD = 220
BACKGROUND_FRACTION = 0.85


def is_background(
    tile_rgb: np.ndarray,
    white_threshold: int = WHITE_THRESHOLD,
    background_fraction: float = BACKGROUND_FRACTION,
) -> bool:
    """True if more than ``background_fraction`` of pixels are white in every channel."""
    white = np.all(tile_rgb > white_threshold, axis=-1)
    return bool(white.mean() > background_fraction)


def level0_location(x: int, y: int, downsample: float) -> tuple[int, int]:
    """Convert a level-``k`` pixel position to level 0, given that level's downsample."""
    return int(round(x * downsample)), int(round(y * downsample))


def tile_slide(
    slide_path: Path, output_dir: Path, tile_size: int = TILE_SIZE, level: int = LEVEL
) -> Path:
    """Write the foreground tile coordinates of one slide to ``output_dir/<slide>.h5``.

    Slides that already have an output file are skipped, so an interrupted run can resume.
    """
    import openslide  # optional dependency: pip install "ptc-mil[wsi]"

    output = output_dir / f"{slide_name(slide_path)}.h5"
    if output.exists():
        return output

    slide = openslide.OpenSlide(str(slide_path))
    try:
        downsample = float(slide.level_downsamples[level])
        width, height = slide.level_dimensions[level]
        coords = []
        for y in range(0, height - tile_size + 1, tile_size):
            for x in range(0, width - tile_size + 1, tile_size):
                tile = slide.read_region(
                    level0_location(x, y, downsample), level, (tile_size, tile_size)
                )
                if not is_background(np.array(tile.convert("RGB"))):
                    coords.append([x, y])
    finally:
        slide.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as f:
        f.create_dataset("coords", data=np.array(coords))
        f.attrs["slide_path"] = str(slide_path)
        f.attrs["tile_size"] = tile_size
        f.attrs["level"] = level
        f.attrs["level_downsample"] = downsample
        f.attrs["n_tiles"] = len(coords)
    logger.info("%s: %d tiles (downsample %.4f)", output.stem, len(coords), downsample)
    return output


def tile_all(slides_dir: Path, output_dir: Path) -> None:
    """Tile every .svs file under ``slides_dir``. Failures are logged and skipped."""
    slides = sorted(slides_dir.rglob("*.svs"), key=lambda p: p.name)
    logger.info("Tiling %d slides into %s", len(slides), output_dir)
    for path in slides:
        try:
            tile_slide(path, output_dir)
        except Exception:
            logger.exception("Failed to tile %s", path.name)
