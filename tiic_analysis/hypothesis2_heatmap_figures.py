"""
hypothesis2_heatmap_figures.py

Generates a three-panel figure per slide: attention, immune density, and
epithelial density, each rendered as a heatmap over the tissue thumbnail,
using the same tile grid. Because Hypothesis 2's slides have FULL tile
coverage (every tissue tile, not a 250-tile sample), these heatmaps are
gap-free reconstructions, not scattered points, unlike heatmaps built from
the sampled 250-tile data used elsewhere in this chapter.

Intended to accompany the Hypothesis 2 spatial exclusion finding: seeing
attention, immune presence, and epithelial presence side by side makes the
distance-based result (Section 5.4.2.2) visually intuitive.

Uses the *_with_attn.csv files already produced by
hypothesis2_spatial_analysis.py --stage distance (patient, x, y,
immune_density, epithelial_density, attn_multiclass, attn_ret_binary).

No GPU required - this only reads already-computed data and slide
thumbnails.

Usage:
    python hypothesis2_heatmap_figures.py --patients TCGA-EL-A3GY TCGA-EL-A3ZR
    python hypothesis2_heatmap_figures.py --patients TCGA-EL-A3GY --task multiclass
"""

import os
import argparse
import numpy as np
import pandas as pd
import openslide
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

PROJ = "/cs/student/project_msc/2025/aibh/iconstan/"
SLIDES_DIR = PROJ + "slides"
TILES_DIR = PROJ + "tiles_v2"
DATA_DIR = PROJ + "hypothesis2_analysis/per_slide/"
OUTPUT_DIR = PROJ + "hypothesis2_analysis/figures/"

THUMBNAIL_MAX_DIM = 1200
ALPHA = 0.45  # matches the alpha used in the existing attention heatmap convention

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_slide_file(patient):
    for root, dirs, files in os.walk(SLIDES_DIR):
        for f in files:
            if f.startswith(patient) and f.endswith(".svs"):
                return os.path.join(root, f)
    return None


def get_tile_geometry(patient):
    """Returns tile_size (at level 1) and level_downsample, needed to place
    each tile correctly on the level-0 thumbnail."""
    for f in os.listdir(TILES_DIR):
        if f.startswith(patient) and f.endswith(".h5"):
            with h5py.File(os.path.join(TILES_DIR, f), "r") as h5:
                tile_size = int(h5.attrs["tile_size"])
                level_downsample = float(h5.attrs["level_downsample"])
            return tile_size, level_downsample
    return None, None


def percentile_normalise(values):
    """Percentile-rank normalisation, matching the convention already used
    for attention heatmaps elsewhere in this project (CLAM-style), so all
    three panels are visually comparable on the same 0-1 scale regardless
    of each value's raw distribution."""
    ranks = pd.Series(values).rank(pct=True).values
    return ranks


def build_heatmap_array(df, value_col, tile_size, level_downsample, thumb_scale):
    """
    Places each tile's (percentile-normalised) value into a 2D array at its
    correct grid position, producing a gap-free heatmap since coverage is
    complete. thumb_scale converts level-0 pixel coordinates down to
    thumbnail pixel coordinates.
    """
    x0 = df["x"].values * level_downsample
    y0 = df["y"].values * level_downsample
    tile_size_l0 = tile_size * level_downsample

    thumb_x = (x0 * thumb_scale).astype(int)
    thumb_y = (y0 * thumb_scale).astype(int)
    thumb_tile_size = max(1, int(round(tile_size_l0 * thumb_scale)))

    width = int(df["x"].max() * level_downsample * thumb_scale) + thumb_tile_size + 1
    height = int(df["y"].max() * level_downsample * thumb_scale) + thumb_tile_size + 1

    heat = np.full((height, width), np.nan)
    values = percentile_normalise(df[value_col].values)

    for xi, yi, v in zip(thumb_x, thumb_y, values):
        heat[yi:yi + thumb_tile_size, xi:xi + thumb_tile_size] = v

    return heat


def make_figure(patient, task):
    csv_path = os.path.join(DATA_DIR, f"{patient}_with_attn.csv")
    if not os.path.exists(csv_path):
        print(f"  {patient}: missing {csv_path}. Run hypothesis2_spatial_analysis.py "
              f"--stage distance first.")
        return

    df = pd.read_csv(csv_path)
    attn_col = f"attn_{task}"
    if attn_col not in df.columns:
        print(f"  {patient}: no '{attn_col}' column found, skipping.")
        return
    df = df.dropna(subset=[attn_col, "immune_density", "epithelial_density"])

    slide_path = get_slide_file(patient)
    tile_size, level_downsample = get_tile_geometry(patient)
    if slide_path is None or tile_size is None:
        print(f"  {patient}: missing slide file or tile geometry, skipping.")
        return

    slide = openslide.OpenSlide(slide_path)
    w0, h0 = slide.dimensions
    thumb_scale = min(THUMBNAIL_MAX_DIM / w0, THUMBNAIL_MAX_DIM / h0)
    thumb = slide.get_thumbnail((int(w0 * thumb_scale), int(h0 * thumb_scale)))
    thumb = np.array(thumb.convert("RGB"))
    slide.close()

    panels = [
        ("immune_density", "Immune density"),
        (attn_col, f"Attention ({task})"),
        ("epithelial_density", "Epithelial density"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im = None
    for ax, (col, title) in zip(axes, panels):
        heat = build_heatmap_array(df, col, tile_size, level_downsample, thumb_scale)
        th, tw = thumb.shape[:2]
        hh, hw = heat.shape
        padded = np.full((max(th, hh), max(tw, hw)), np.nan)
        padded[:hh, :hw] = heat
        heat = padded[:th, :tw]

        ax.imshow(thumb)
        im = ax.imshow(heat, cmap="jet", alpha=ALPHA, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle(f"{patient}", fontsize=14)
    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.03, pad=0.02,
                 label="Percentile rank (within slide)")

    out_path = os.path.join(OUTPUT_DIR, f"{patient}_three_panel_{task}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main(patients, task):
    print(f"Generating three-panel figures ({task} attention) for {len(patients)} slide(s)")
    for patient in patients:
        print(f"\n{patient}:")
        make_figure(patient, task)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="+", required=True,
                     help="Patient IDs, e.g. TCGA-EL-A3GY TCGA-EL-A3ZR")
    ap.add_argument("--task", choices=["multiclass", "ret_binary"], default="ret_binary")
    args = ap.parse_args()
    main(args.patients, args.task)
