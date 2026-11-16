"""Check how much the TIIC correlations depend on how much of each tile is segmented.

The main TIIC analysis (``tiic_correlation.py``) counts nuclei over a 4x4 grid of
256x256 level-0 crops, covering each tile's 1024x1024 footprint completely. An earlier
design read only a 2x2 grid with a 512 px stride, i.e. 25% of the tile. This script
re-runs nucleus counting on the same tiles of a subset of slides with that 2x2 design
and compares the per-slide correlations with the main results (a paired comparison:
the tile coordinates are read back from ``tiic_per_tile_{task}.csv``).

In the dissertation this comparison was run the other way round (2x2 main results
re-run at 4x4), and showed that the 25% subsample biased the partial correlation
towards more negative values, which is why the final analysis uses full coverage.

Usage:

    python analyses/tiic/tiic_subcrop_validation.py --task ret_binary --n-slides 10
"""

import argparse
import os
import sys

import numpy as np
import openslide
import pandas as pd
import tiic_correlation as tiic
from scipy.stats import spearmanr

OUTPUT_DIR = tiic.OUTPUT_DIR
SUBSAMPLE_GRID = 2  # 2x2 crops = 25% of the tile


def select_slides(summary_df, n):
    """Pick n slides spanning the range of partial correlations (where an artefact would show)."""
    d = summary_df.sort_values("partial_r_immune_given_epithelial")
    idx = np.linspace(0, len(d) - 1, min(n, len(d))).astype(int)
    return d.iloc[idx]["patient"].tolist()


def reanalyse_slide(patient, tiles_for_slide, segmentor):
    """Re-run nucleus counting on exactly the tiles used in the main analysis."""
    slide_path = tiic.ctx.slide_file(patient)
    if slide_path is None:
        return None
    _, tile_size, level_downsample = tiic.ctx.tile_info(patient)
    slide = openslide.OpenSlide(str(slide_path))

    rows = []
    n_tiles = len(tiles_for_slide)
    for i, (_, row) in enumerate(tiles_for_slide.iterrows()):
        x, y = int(row["x"]), int(row["y"])
        counts = tiic.compute_nuclei_counts_for_tile(
            slide, x, y, level_downsample, tile_size, segmentor
        )
        if counts is None:
            continue
        total = counts["total_nuclei"]
        if total <= 0:
            continue
        rows.append(
            {
                "attn": float(row["attn"]),
                "immune_density": counts["immune_nuclei"] / total,
                "epithelial_density": counts["epithelial_nuclei"] / total,
                "immune_nuclei": counts["immune_nuclei"],
            }
        )
        if (i + 1) % 50 == 0:
            print(f"      {i + 1}/{n_tiles} tiles", flush=True)

    slide.close()
    if len(rows) < 10:
        return None

    d = pd.DataFrame(rows)
    r_imm, _ = spearmanr(d["attn"], d["immune_density"])
    r_epi, _ = spearmanr(d["attn"], d["epithelial_density"])
    pr, pp = tiic.partial_spearman(
        d["attn"].values, d["immune_density"].values, d["epithelial_density"].values
    )
    return {
        "patient": patient,
        "n_tiles_subsample": len(d),
        "r_immune_subsample": r_imm,
        "r_epithelial_subsample": r_epi,
        "partial_r_subsample": pr,
        "partial_p_subsample": pp,
    }


def main(task, n_slides):
    summary_path = os.path.join(OUTPUT_DIR, f"tiic_summary_{task}.csv")
    tile_path = os.path.join(OUTPUT_DIR, f"tiic_per_tile_{task}.csv")
    for p in (summary_path, tile_path):
        if not os.path.exists(p):
            sys.exit(f"Missing {p}; run tiic_correlation.py first")

    summary = pd.read_csv(summary_path)
    tiles = pd.read_csv(tile_path)
    slides = select_slides(summary, n_slides)
    print(
        f"Re-running {len(slides)} slides for '{task}' with a {SUBSAMPLE_GRID}x{SUBSAMPLE_GRID} grid"
    )
    print(f"Slides: {slides}\n")

    tiic.SUBCROPS_PER_TILE = SUBSAMPLE_GRID
    segmentor = tiic.load_segmentor()

    results = []
    for i, patient in enumerate(slides):
        sub = tiles[tiles["patient"] == patient]
        if len(sub) == 0:
            print(f"[{i + 1}/{len(slides)}] {patient}: no tiles in CSV, skipped")
            continue
        print(f"[{i + 1}/{len(slides)}] {patient} ({len(sub)} tiles)")
        res = reanalyse_slide(patient, sub, segmentor)
        if res:
            results.append(res)
    if not results:
        sys.exit("No slides completed.")

    full = summary[
        [
            "patient",
            "spearman_r_immune_density",
            "spearman_r_epithelial_density",
            "partial_r_immune_given_epithelial",
        ]
    ].rename(
        columns={
            "spearman_r_immune_density": "r_immune_full",
            "spearman_r_epithelial_density": "r_epithelial_full",
            "partial_r_immune_given_epithelial": "partial_r_full",
        }
    )
    comp = pd.DataFrame(results).merge(full, on="patient", how="left")
    comp["d_immune"] = comp["r_immune_subsample"] - comp["r_immune_full"]
    comp["d_epithelial"] = comp["r_epithelial_subsample"] - comp["r_epithelial_full"]
    comp["d_partial"] = comp["partial_r_subsample"] - comp["partial_r_full"]

    print("\n" + "=" * 70)
    print(f"PAIRED COMPARISON - {task}")
    print("=" * 70)
    print(comp.round(3).to_string(index=False))

    print("\nAgreement (25% subsample vs full coverage):")
    for lab, full_col, sub_col in [
        ("attn~immune", "r_immune_full", "r_immune_subsample"),
        ("attn~epithelial", "r_epithelial_full", "r_epithelial_subsample"),
        ("partial", "partial_r_full", "partial_r_subsample"),
    ]:
        if comp[full_col].notna().sum() >= 3:
            rho, _ = spearmanr(comp[full_col], comp[sub_col])
            diff = comp[sub_col] - comp[full_col]
            print(
                f"  {lab:18s} rho={rho:+.3f}  mean|diff|={diff.abs().mean():.3f}  mean diff={diff.mean():+.3f}"
            )

    out = os.path.join(OUTPUT_DIR, f"subcrop_validation_{task}.csv")
    comp.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", choices=["multiclass", "ret_binary"], default="ret_binary")
    ap.add_argument("--n-slides", type=int, default=10)
    args = ap.parse_args()
    main(args.task, args.n_slides)
