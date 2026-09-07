"""
Assets for the tiling and preprocessing figure. All panels are drawn from the
same zoom block, so the individual tiles in panel C are visibly present in
panel B.

Outputs (in figure_assets/):
    wsi_grid_overview.png   whole slide, cropped to tissue, grid + zoom marker
    wsi_grid_zoom.png       the zoom block at full detail with the same grid
    zoom_tile_0.png ...     retained tiles taken from inside the zoom block
    zoom_tile_bg.png        a rejected tile taken from inside the zoom block

Run with venv310 on a GPU machine (needs openslide, h5py, numpy, Pillow).

Usage:
    python3 extract_tiling_assets_v5.py
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

# --- zoom block selection ---
ZOOM_TILES = 8
TARGET_MIN = 45          # want mostly tissue...
TARGET_MAX = 58          # ...but with a few rejected cells visible
N_TISSUE_TILES = 3       # how many retained tiles to export for panel C

# --- appearance ---
OVERVIEW_MAX = 1600
GRID_RGB = (0, 0, 0)
OVERVIEW_GRID_WIDTH = 1
ZOOM_GRID_WIDTH = 3
ZOOM_OUT = 1400
MARGIN_FRAC = 0.02
ZOOM_BOX_WIDTH = 5
TILE_OUT = 512

# --- background definition, should match the tiling pipeline ---
WHITE_THRESHOLD = 220
BACKGROUND_MIN_FRAC = 0.85

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

# =====================================================================
# tissue bounding box
# =====================================================================
xs = coords[:, 0].astype(float) * level_downsample
ys = coords[:, 1].astype(float) * level_downsample
bx0, bx1 = xs.min(), xs.max() + size_l0
by0, by1 = ys.min(), ys.max() + size_l0
pad_x, pad_y = (bx1 - bx0) * MARGIN_FRAC, (by1 - by0) * MARGIN_FRAC
bx0, by0 = max(0, bx0 - pad_x), max(0, by0 - pad_y)
bx1, by1 = min(w0, bx1 + pad_x), min(h0, by1 + pad_y)
bw, bh = bx1 - bx0, by1 - by0
print(f"tissue bounding box at level 0: ({bx0:.0f},{by0:.0f}) {bw:.0f} x {bh:.0f}")

# =====================================================================
# zoom block: prefer a block whose retained count falls in the target
# band, so both kept and discarded tiles are visible in the zoom
# =====================================================================
step = tile_size
gxs = sorted(set(int(x) for x, y in coords))
gys = sorted(set(int(y) for x, y in coords))

in_band = []
best_any, best_any_count = None, -1
for gx in gxs[::2]:
    for gy in gys[::2]:
        count = 0
        for i in range(ZOOM_TILES):
            for j in range(ZOOM_TILES):
                if (gx + i * step, gy + j * step) in retained:
                    count += 1
        if TARGET_MIN <= count <= TARGET_MAX:
            in_band.append((gx, gy, count))
        if count > best_any_count:
            best_any_count, best_any = count, (gx, gy)

if in_band:
    # of the blocks in band, take the densest, so the tissue still dominates
    in_band.sort(key=lambda t: -t[2])
    zx_lvl, zy_lvl, zcount = in_band[0]
    print(f"{len(in_band)} blocks in the {TARGET_MIN}-{TARGET_MAX} band; "
          f"using ({zx_lvl},{zy_lvl}) with {zcount}/{ZOOM_TILES**2} retained")
else:
    zx_lvl, zy_lvl = best_any
    zcount = best_any_count
    print(f"No block in band. Falling back to densest: ({zx_lvl},{zy_lvl}) "
          f"with {zcount}/{ZOOM_TILES**2} retained. Widen TARGET_MIN/MAX.")

zx0, zy0 = int(zx_lvl * level_downsample), int(zy_lvl * level_downsample)
zside = min(int(ZOOM_TILES * size_l0), w0 - zx0, h0 - zy0)

# =====================================================================
# overview
# =====================================================================
scale = OVERVIEW_MAX / max(bw, bh)
ow, oh = int(bw * scale), int(bh * scale)
overview = slide.get_thumbnail((int(w0 * scale), int(h0 * scale))).convert("RGB")
overview = overview.crop((int(bx0 * scale), int(by0 * scale),
                          int(bx0 * scale) + ow, int(by0 * scale) + oh))
odraw = ImageDraw.Draw(overview)
for x, y in coords:
    x0 = int((x * level_downsample - bx0) * scale)
    y0 = int((y * level_downsample - by0) * scale)
    x1 = int((x * level_downsample + size_l0 - bx0) * scale)
    y1 = int((y * level_downsample + size_l0 - by0) * scale)
    odraw.rectangle([x0, y0, x1, y1], outline=GRID_RGB, width=OVERVIEW_GRID_WIDTH)
odraw.rectangle(
    [int((zx0 - bx0) * scale), int((zy0 - by0) * scale),
     int((zx0 + zside - bx0) * scale), int((zy0 + zside - by0) * scale)],
    outline=GRID_RGB, width=ZOOM_BOX_WIDTH)
overview.save(os.path.join(OUT_DIR, "wsi_grid_overview.png"))
print("Saved wsi_grid_overview.png", overview.size)

# =====================================================================
# zoom, with the grid drawn only on retained cells
# =====================================================================
zoom = slide.read_region((zx0, zy0), 0, (zside, zside)).convert("RGB")
zoom = zoom.resize((ZOOM_OUT, ZOOM_OUT), Image.LANCZOS)
zscale = ZOOM_OUT / zside
zdraw = ImageDraw.Draw(zoom)
for x, y in coords:
    px, py = x * level_downsample, y * level_downsample
    if zx0 <= px < zx0 + zside and zy0 <= py < zy0 + zside:
        x0, y0 = int((px - zx0) * zscale), int((py - zy0) * zscale)
        x1 = int((px + size_l0 - zx0) * zscale)
        y1 = int((py + size_l0 - zy0) * zscale)
        zdraw.rectangle([x0, y0, x1, y1], outline=GRID_RGB, width=ZOOM_GRID_WIDTH)
zoom.save(os.path.join(OUT_DIR, "wsi_grid_zoom.png"))
print("Saved wsi_grid_zoom.png", zoom.size)

# =====================================================================
# individual tiles from inside the zoom block, clean (no grid drawn)
# =====================================================================
kept_cells, rejected_cells = [], []
for i in range(ZOOM_TILES):
    for j in range(ZOOM_TILES):
        cx, cy = zx_lvl + i * step, zy_lvl + j * step
        px, py = int(cx * level_downsample), int(cy * level_downsample)
        if px + size_l0 > w0 or py + size_l0 > h0:
            continue
        (kept_cells if (cx, cy) in retained else rejected_cells).append((i, j, px, py))

print(f"zoom block contains {len(kept_cells)} retained and "
      f"{len(rejected_cells)} rejected cells")

rng = np.random.default_rng(0)
rng.shuffle(kept_cells)
for n, (i, j, px, py) in enumerate(kept_cells[:N_TISSUE_TILES]):
    region = slide.read_region((px, py), 0, (size_l0, size_l0)).convert("RGB")
    region.resize((TILE_OUT, TILE_OUT), Image.LANCZOS).save(
        os.path.join(OUT_DIR, f"zoom_tile_{n}.png"))
    print(f"Saved zoom_tile_{n}.png  from block cell ({i},{j})")

# the rejected cell whose white fraction is closest to the threshold, so the
# example is a borderline case rather than blank glass
best_bg = None
for i, j, px, py in rejected_cells:
    region = slide.read_region((px, py), 0, (size_l0, size_l0)).convert("RGB")
    wf = float((np.asarray(region) > WHITE_THRESHOLD).all(axis=2).mean())
    if wf < BACKGROUND_MIN_FRAC:
        continue                      # rejected for some other reason
    if best_bg is None or wf < best_bg[1]:
        best_bg = (region, wf, i, j)

if best_bg is None:
    print("No rejected cell in the zoom block met the white-fraction rule.")
else:
    region, wf, i, j = best_bg
    region.resize((TILE_OUT, TILE_OUT), Image.LANCZOS).save(
        os.path.join(OUT_DIR, "zoom_tile_bg.png"))
    print(f"Saved zoom_tile_bg.png  white fraction {wf:.3f} from block cell ({i},{j})")

slide.close()
print("\nAssets in:", OUT_DIR)
