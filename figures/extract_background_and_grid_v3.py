"""
Assets for the tiling and preprocessing figure:
  - a background patch adjacent to retained tissue that the white-pixel
    filter would reject, chosen so it still shows a sliver of tissue
  - the WSI thumbnail with the retained tile grid drawn on it

Run with venv310 on a GPU machine (needs openslide, h5py, numpy, Pillow).

Usage:
    python3 extract_background_and_grid_v3.py
Output (written to figure_assets/):
    tile_background.png
    wsi_grid_outline.png
"""

import os
import h5py
import numpy as np
from PIL import Image, ImageDraw
import openslide

PROJ = "/cs/student/project_msc/2025/aibh/iconstan/"
SLIDES_DIR = PROJ + "slides/"
TILES_DIR = PROJ + "tiles_v2/"
OUT_DIR = PROJ + "figure_assets/"

PATIENT = "TCGA-EL-A3GV"

# --- appearance ---
THUMB_MAX = 1600            # smaller than v2, so the grid is proportionally lighter
GRID_RGB = (0, 0, 0)
GRID_WIDTH = 2

# --- background detection, should match the tiling pipeline ---
WHITE_THRESHOLD = 220       # a pixel above this in all channels counts as white
BACKGROUND_MIN_FRAC = 0.85  # the pipeline's discard rule
BACKGROUND_MAX_FRAC = 0.94  # upper bound, so the chosen patch is not blank

os.makedirs(OUT_DIR, exist_ok=True)

# ---- tile coordinates ----
tile_files = [f for f in os.listdir(TILES_DIR)
              if f.startswith(PATIENT) and f.endswith(".h5")]
if not tile_files:
    raise SystemExit(f"No tile file found for {PATIENT}")
with h5py.File(os.path.join(TILES_DIR, sorted(tile_files)[0]), "r") as h5:
    coords = h5["coords"][:]
    tile_size = int(h5.attrs["tile_size"])
    level_downsample = float(h5.attrs["level_downsample"])
print(f"{len(coords)} retained tiles, tile_size={tile_size}, ds={level_downsample}")

# ---- locate the SVS ----
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
    raise SystemExit(f"No SVS found for {PATIENT}")

slide = openslide.OpenSlide(svs_path)
w0, h0 = slide.level_dimensions[0]
size_l0 = int(tile_size * level_downsample)
print(f"slide level-0: {w0} x {h0}, tile footprint at level 0: {size_l0}px")

retained = set((int(x), int(y)) for x, y in coords)

# ---- background patch: scan neighbours of retained tiles ----
# Tissue-edge neighbours are the genuinely borderline cases, so they show a
# sliver of tissue rather than a blank square.
NEIGHBOUR_OFFSETS = [(tile_size, 0), (-tile_size, 0), (0, tile_size), (0, -tile_size)]

candidates = []
for x, y in coords:
    for dx, dy in NEIGHBOUR_OFFSETS:
        nx, ny = int(x) + dx, int(y) + dy
        if (nx, ny) in retained or nx < 0 or ny < 0:
            continue
        px, py = int(nx * level_downsample), int(ny * level_downsample)
        if px + size_l0 > w0 or py + size_l0 > h0:
            continue
        candidates.append((nx, ny, px, py))

# de-duplicate, then walk them until one lands in the target band
seen = set()
unique = []
for c in candidates:
    if (c[0], c[1]) in seen:
        continue
    seen.add((c[0], c[1]))
    unique.append(c)
print(f"{len(unique)} candidate background positions adjacent to retained tissue")

rng = np.random.default_rng(0)
rng.shuffle(unique)

found = None
checked = 0
for nx, ny, px, py in unique:
    if checked >= 600:
        break
    checked += 1
    region = slide.read_region((px, py), 0, (size_l0, size_l0)).convert("RGB")
    arr = np.asarray(region)
    white_frac = float((arr > WHITE_THRESHOLD).all(axis=2).mean())
    if BACKGROUND_MIN_FRAC <= white_frac <= BACKGROUND_MAX_FRAC:
        found = (region, white_frac, px, py)
        break

if found is None:
    print(f"No patch in the {BACKGROUND_MIN_FRAC}-{BACKGROUND_MAX_FRAC} band "
          f"after {checked} checks. Widen BACKGROUND_MAX_FRAC and rerun.")
else:
    region, white_frac, px, py = found
    region.resize((512, 512), Image.LANCZOS).save(
        os.path.join(OUT_DIR, "tile_background.png"))
    print(f"Saved tile_background.png  white fraction {white_frac:.3f} at ({px},{py})")

# ---- thumbnail with the retained grid ----
thumb = slide.get_thumbnail((THUMB_MAX, THUMB_MAX)).convert("RGB")
tw, th = thumb.size
sx, sy = tw / w0, th / h0
print(f"thumbnail: {tw} x {th}  (one tile is ~{size_l0 * sx:.1f}px wide here)")

outline_img = thumb.copy()
draw = ImageDraw.Draw(outline_img)
for x, y in coords:
    x0 = int(x * level_downsample * sx)
    y0 = int(y * level_downsample * sy)
    x1 = int((x * level_downsample + size_l0) * sx)
    y1 = int((y * level_downsample + size_l0) * sy)
    draw.rectangle([x0, y0, x1, y1], outline=GRID_RGB, width=GRID_WIDTH)
outline_img.save(os.path.join(OUT_DIR, "wsi_grid_outline.png"))
print("Saved wsi_grid_outline.png", outline_img.size)

slide.close()
print("\nAssets in:", OUT_DIR)
