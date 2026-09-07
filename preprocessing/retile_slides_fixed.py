import openslide
import numpy as np
import os
import h5py
from tqdm import tqdm

# === PATHS ===
SLIDES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/slides"
TILES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/tiles_v2"   # NEW folder, does not touch original

# === TILING PARAMETERS ===
TILE_SIZE = 256
LEVEL = 1
BACKGROUND_THRESHOLD = 0.85
WHITE_THRESHOLD = 220

os.makedirs(TILES_DIR, exist_ok=True)


def is_background(tile, white_thresh=WHITE_THRESHOLD, bg_thresh=BACKGROUND_THRESHOLD):
    tile_np = np.array(tile.convert("RGB"))
    white_pixels = np.all(tile_np > white_thresh, axis=-1)
    return white_pixels.mean() > bg_thresh


def tile_slide(slide_path, output_dir, tile_size=TILE_SIZE, level=LEVEL):
    slide_name = os.path.basename(slide_path).replace(".svs", "")
    h5_path = os.path.join(output_dir, f"{slide_name}.h5")

    if os.path.exists(h5_path):
        return  # already done, allows resuming

    try:
        slide = openslide.OpenSlide(slide_path)

        # *** THE FIX: use the slide's ACTUAL downsample factor, not an assumed 2**level ***
        level_downsample = slide.level_downsamples[level]

        w, h = slide.level_dimensions[level]

        coords = []
        for y in range(0, h - tile_size + 1, tile_size):
            for x in range(0, w - tile_size + 1, tile_size):
                # Correct conversion to level-0 coordinates for read_region
                loc_x = int(round(x * level_downsample))
                loc_y = int(round(y * level_downsample))
                tile = slide.read_region((loc_x, loc_y), level, (tile_size, tile_size))
                if not is_background(tile):
                    coords.append([x, y])

        coords = np.array(coords)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("coords", data=coords)
            f.attrs["slide_path"] = slide_path
            f.attrs["tile_size"] = tile_size
            f.attrs["level"] = level
            f.attrs["level_downsample"] = level_downsample  # NEW: store the real downsample used
            f.attrs["n_tiles"] = len(coords)

        slide.close()
        print(f"  {slide_name}: {len(coords)} tiles (downsample={level_downsample:.4f})")

    except Exception as e:
        print(f"  ERROR on {slide_name}: {e}")


if __name__ == "__main__":
    svs_files = []
    for root, dirs, files in os.walk(SLIDES_DIR):
        for f in files:
            if f.endswith(".svs"):
                svs_files.append(os.path.join(root, f))

    print(f"Re-tiling {len(svs_files)} slides with CORRECTED coordinate conversion...")
    print(f"Output directory: {TILES_DIR}\n")

    for slide_path in tqdm(svs_files):
        tile_slide(slide_path, TILES_DIR)

    print("\nRe-tiling complete!")
