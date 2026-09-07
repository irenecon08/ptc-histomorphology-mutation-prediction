"""
Extract figure assets from one representative slide:
  - a whole-slide thumbnail
  - a small number of tile crops at the tiling resolution

Run with venv310 on a GPU machine (needs openslide, h5py, numpy, Pillow).

Usage:
    python3 extract_figure_assets.py
Output (written to figure_assets/):
    wsi_thumbnail.png
    tile_00.png ... tile_03.png
"""

import os
import h5py
import numpy as np
from PIL import Image
import openslide

PROJ = "/cs/student/project_msc/2025/aibh/iconstan/"
SLIDES_DIR = PROJ + "slides/"
TILES_DIR = PROJ + "tiles_v2/"
OUT_DIR = PROJ + "figure_assets/"

# Pick a slide. Any patient works; this one is in the test set and was used
# in the shape analysis, so it is already a slide you have looked at.
PATIENT = "TCGA-EL-A3GV"

N_TILES = 4          # how many tile crops to save
THUMB_MAX = 900      # longest edge of the thumbnail, in pixels

os.makedirs(OUT_DIR, exist_ok=True)

# ---- locate the tile file for this patient ----
tile_files = [f for f in os.listdir(TILES_DIR)
              if f.startswith(PATIENT) and f.endswith(".h5")]
if not tile_files:
    raise SystemExit(f"No tile file found for {PATIENT} in {TILES_DIR}")
tile_path = os.path.join(TILES_DIR, sorted(tile_files)[0])
print("Tile file:", tile_path)

with h5py.File(tile_path, "r") as h5:
    coords = h5["coords"][:]
    tile_size = int(h5.attrs["tile_size"])
    level_downsample = float(h5.attrs["level_downsample"])
print(f"{len(coords)} tiles, tile_size={tile_size}, level_downsample={level_downsample}")

# ---- locate the SVS. slides/ is one UUID directory per file ----
svs_path = None
for uuid_dir in os.listdir(SLIDES_DIR):
    d = os.path.join(SLIDES_DIR, uuid_dir)
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        if f.startswith(PATIENT) and f.endswith(".svs"):
            svs_path = os.path.join(d, f)
            break
    if svs_path:
        break
if svs_path is None:
    raise SystemExit(f"No SVS found for {PATIENT} under {SLIDES_DIR}")
print("Slide:", svs_path)

slide = openslide.OpenSlide(svs_path)

# ---- thumbnail ----
thumb = slide.get_thumbnail((THUMB_MAX, THUMB_MAX))
thumb.save(os.path.join(OUT_DIR, "wsi_thumbnail.png"))
print("Saved wsi_thumbnail.png", thumb.size)

# ---- tile crops, spread across the slide rather than adjacent ----
rng = np.random.default_rng(0)
idx = rng.choice(len(coords), size=min(N_TILES, len(coords)), replace=False)

for n, i in enumerate(idx):
    x, y = coords[i]
    # coords are stored at tiling-level resolution; convert to level 0
    x0 = int(x * level_downsample)
    y0 = int(y * level_downsample)
    size_l0 = int(tile_size * level_downsample)
    region = slide.read_region((x0, y0), 0, (size_l0, size_l0)).convert("RGB")
    region = region.resize((512, 512), Image.LANCZOS)
    out = os.path.join(OUT_DIR, f"tile_{n:02d}.png")
    region.save(out)
    print("Saved", out, f"(slide coords {x0},{y0})")

slide.close()
print("\nAll assets in:", OUT_DIR)
