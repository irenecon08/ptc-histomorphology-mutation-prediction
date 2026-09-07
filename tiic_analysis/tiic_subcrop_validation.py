"""
tiic_subcrop_validation.py

Validates the 2x2 sub-crop sampling design used in tiic_correlation_v4.py.

BACKGROUND
----------
The main analysis reads four 256x256 sub-crops per tile at level 0, placed on a
2x2 grid with a stride of 512 px within the tile's 1024x1024 level-0 footprint.
Because each crop is 256 px wide and the stride is 512 px, these four crops
sample 25% of the tile area as four spatially separated windows - not the full
footprint.

That is a systematic spatial subsample and should be unbiased, but it is an
assumption rather than a demonstrated fact. This script tests it empirically by
re-running nucleus detection on a subset of slides with SUBCROPS_PER_TILE = 4
(a 4x4 grid, stride 256 px, 16 crops per tile = exhaustive coverage) and
comparing the resulting per-slide correlations against the values already in
tiic_summary_{task}.csv.

The comparison is PAIRED: the exact same tile coordinates are re-analysed,
read back from tiic_per_tile_{task}.csv, so the only thing that differs between
old and new is sub-crop coverage.

USAGE
-----
    python tiic_subcrop_validation.py --probe
        Prints what get_tile_info() returns and what columns the per-tile CSV
        has, without running any HoVer-Net. Run this first.

    python tiic_subcrop_validation.py --task ret_binary --n-slides 10
        Runs the validation on 10 slides for one task.

RUNTIME
-------
16 crops per tile instead of 4, so roughly 4x the original cost: about 160 s
per slide, i.e. ~30 min for 10 slides. Run one task first.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import openslide
from scipy.stats import spearmanr

import tiic_correlation_v4 as t4
from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

OUTPUT_DIR = t4.OUTPUT_DIR


# ==========================================================================
# Adapters - isolate assumptions about the v4 module's internals here
# ==========================================================================
def unpack_tile_info(info):
    """
    get_tile_info(patient) returns the per-slide tiling metadata. Its exact
    shape is not assumed: handle dict, or a tuple/list containing the
    coordinates array, the level downsample, and the tile size in some order.
    Returns (coords, level_downsample, tile_size).
    """
    if isinstance(info, dict):
        coords = info.get("coords")
        ld = info.get("level_downsample", info.get("downsample"))
        ts = info.get("tile_size", info.get("patch_size", 256))
        return coords, ld, ts

    if isinstance(info, (tuple, list)):
        coords, ld, ts = None, None, None
        for item in info:
            if hasattr(item, "shape") and getattr(item, "ndim", 0) == 2:
                coords = item
            elif isinstance(item, (int, np.integer)) and item > 16:
                ts = int(item)
            elif isinstance(item, (float, np.floating)):
                ld = float(item)
        if coords is not None and ld is not None:
            return coords, ld, (ts if ts else 256)

    raise RuntimeError(
        "Could not interpret get_tile_info() output. Run with --probe and send "
        "me the printed structure so the adapter can be corrected."
    )


def find_coord_columns(df):
    """Locate the tile coordinate columns in the per-tile CSV."""
    candidates = [("x", "y"), ("tile_x", "tile_y"), ("coord_x", "coord_y"),
                  ("X", "Y"), ("x_coord", "y_coord")]
    for cx, cy in candidates:
        if cx in df.columns and cy in df.columns:
            return cx, cy
    return None, None


# ==========================================================================
# Probe
# ==========================================================================
def probe(task):
    print("=" * 70)
    print("PROBE - no HoVer-Net will be run")
    print("=" * 70)

    tile_csv = os.path.join(OUTPUT_DIR, f"tiic_per_tile_{task}.csv")
    if not os.path.exists(tile_csv):
        sys.exit(f"Missing {tile_csv}")

    df = pd.read_csv(tile_csv, nrows=5)
    print(f"\nper-tile CSV columns ({task}):")
    for c in df.columns:
        print(f"  {c}")

    cx, cy = find_coord_columns(df)
    if cx:
        print(f"\n  -> coordinate columns found: '{cx}', '{cy}'  (paired "
              f"comparison possible)")
    else:
        print("\n  -> NO coordinate columns found. A paired re-run is not "
              "possible from this file; send me the column list and I will "
              "adapt the script.")

    patient = pd.read_csv(tile_csv, usecols=["patient"], nrows=1)["patient"].iloc[0]
    print(f"\nget_tile_info('{patient}') returns:")
    info = t4.get_tile_info(patient)
    print(f"  type: {type(info)}")
    if isinstance(info, (tuple, list)):
        for i, item in enumerate(info):
            desc = getattr(item, "shape", None)
            print(f"  [{i}] {type(item).__name__}"
                  f"{f' shape={desc}' if desc is not None else f' = {item!r}'}")
    elif isinstance(info, dict):
        for k, v in info.items():
            desc = getattr(v, "shape", None)
            print(f"  '{k}': {type(v).__name__}"
                  f"{f' shape={desc}' if desc is not None else f' = {v!r}'}")
    else:
        print(f"  {info!r}")

    try:
        coords, ld, ts = unpack_tile_info(info)
        print(f"\n  -> adapter OK: {len(coords)} coords, "
              f"level_downsample={ld}, tile_size={ts}")
        print(f"  -> level-0 footprint per tile: {int(ts*ld)} x {int(ts*ld)} px")
        print(f"  -> current stride (2x2): {int(ts*ld/2)} px, crop 256 px "
              f"-> {100*4*256**2/(ts*ld)**2:.0f}% coverage")
        print(f"  -> validation stride (4x4): {int(ts*ld/4)} px, crop 256 px "
              f"-> {100*16*256**2/(ts*ld)**2:.0f}% coverage")
    except RuntimeError as e:
        print(f"\n  -> adapter FAILED: {e}")


# ==========================================================================
# Validation run
# ==========================================================================
def select_slides(summary_df, n):
    """
    Pick n slides spanning the range of partial correlation values, so the
    validation covers slides where the residual was near zero and slides where
    it was strongest - i.e. exactly where a coverage artefact would show up.
    """
    d = summary_df.sort_values("partial_r_immune_given_epithelial")
    idx = np.linspace(0, len(d) - 1, min(n, len(d))).astype(int)
    return d.iloc[idx]["patient"].tolist()


def reanalyse_slide(patient, tiles_for_slide, cx, cy, model, task, segmentor):
    """Re-run nucleus counting on exactly the tiles used originally."""
    slide_path = t4.get_slide_file(patient)
    if slide_path is None:
        return None
    slide = openslide.OpenSlide(slide_path)

    info = t4.get_tile_info(patient)
    _, level_downsample, tile_size = unpack_tile_info(info)

    rows = []
    n_tiles = len(tiles_for_slide)
    for i, (_, row) in enumerate(tiles_for_slide.iterrows()):
        x, y = int(row[cx]), int(row[cy])
        counts = t4.compute_nuclei_counts_for_tile(
            slide, x, y, level_downsample, tile_size, segmentor
        )
        if counts is None:
            continue
        total = counts["total_nuclei"]
        immune = counts["immune_nuclei"]
        epithelial = counts["epithelial_nuclei"]
        if total <= 0:
            continue
        rows.append({
            "attn": float(row["attn"]),
            "immune_density": immune / total,
            "epithelial_density": epithelial / total,
            "immune_nuclei": immune,
        })
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{n_tiles} tiles", flush=True)

    slide.close()
    if len(rows) < 10:
        return None

    d = pd.DataFrame(rows)
    r_imm, p_imm = spearmanr(d["attn"], d["immune_density"])
    r_epi, _ = spearmanr(d["attn"], d["epithelial_density"])
    pr, pp = t4.partial_spearman(
        d["attn"].values, d["immune_density"].values, d["epithelial_density"].values
    )
    return {
        "patient": patient,
        "n_tiles_new": len(d),
        "r_immune_new": r_imm,
        "r_epithelial_new": r_epi,
        "partial_r_new": pr,
        "partial_p_new": pp,
    }


def main(task, n_slides):
    summary_path = os.path.join(OUTPUT_DIR, f"tiic_summary_{task}.csv")
    tile_path = os.path.join(OUTPUT_DIR, f"tiic_per_tile_{task}.csv")
    for p in (summary_path, tile_path):
        if not os.path.exists(p):
            sys.exit(f"Missing {p}")

    summary = pd.read_csv(summary_path)
    tiles = pd.read_csv(tile_path)

    cx, cy = find_coord_columns(tiles)
    if cx is None:
        sys.exit("No coordinate columns in the per-tile CSV - run --probe and "
                 "send me the column list.")

    slides = select_slides(summary, n_slides)
    print(f"Validating {len(slides)} slides for task '{task}' with "
          f"SUBCROPS_PER_TILE=4 (16 crops/tile, exhaustive coverage)")
    print(f"Slides: {slides}\n")

    # Switch to exhaustive coverage
    t4.SUBCROPS_PER_TILE = 4

    print("Loading HoVer-Net...")
    segmentor = MultiTaskSegmentor(
        model="hovernet_fast-monusac", batch_size=4, num_workers=1,
        device="cuda" if t4.DEVICE == "cuda" else "cpu", verbose=False,
    )
    model = t4.load_abmil_model(task)
    print("  Loaded.\n")

    results = []
    for i, patient in enumerate(slides):
        sub = tiles[tiles["patient"] == patient]
        if len(sub) == 0:
            print(f"[{i+1}/{len(slides)}] {patient}: no tiles in CSV, skipped")
            continue
        print(f"[{i+1}/{len(slides)}] {patient} ({len(sub)} tiles)")
        res = reanalyse_slide(patient, sub, cx, cy, model, task, segmentor)
        if res:
            results.append(res)

    if not results:
        sys.exit("No slides completed.")

    new = pd.DataFrame(results)
    old = summary[["patient", "spearman_r_immune_density",
                   "spearman_r_epithelial_density",
                   "partial_r_immune_given_epithelial"]]
    comp = new.merge(old, on="patient", how="left").rename(columns={
        "spearman_r_immune_density": "r_immune_old",
        "spearman_r_epithelial_density": "r_epithelial_old",
        "partial_r_immune_given_epithelial": "partial_r_old",
    })

    comp["d_immune"] = comp["r_immune_new"] - comp["r_immune_old"]
    comp["d_epithelial"] = comp["r_epithelial_new"] - comp["r_epithelial_old"]
    comp["d_partial"] = comp["partial_r_new"] - comp["partial_r_old"]

    print("\n" + "=" * 70)
    print(f"PAIRED COMPARISON - {task}")
    print("=" * 70)
    print(comp[["patient", "r_immune_old", "r_immune_new",
                "r_epithelial_old", "r_epithelial_new",
                "partial_r_old", "partial_r_new"]].round(3).to_string(index=False))

    print("\nAgreement (25% subsample vs exhaustive):")
    for lab, o, n_ in [("attn~immune", "r_immune_old", "r_immune_new"),
                       ("attn~epithelial", "r_epithelial_old", "r_epithelial_new"),
                       ("partial", "partial_r_old", "partial_r_new")]:
        if comp[o].notna().sum() >= 3:
            rho, _ = spearmanr(comp[o], comp[n_])
            mad = (comp[n_] - comp[o]).abs().mean()
            bias = (comp[n_] - comp[o]).mean()
            print(f"  {lab:18s} rho={rho:+.3f}  mean|diff|={mad:.3f}  "
                  f"mean diff={bias:+.3f}")

    out = os.path.join(OUTPUT_DIR, f"subcrop_validation_{task}.csv")
    comp.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    print("\nHow to read this:")
    print("  High rho and small mean|diff| -> the 25% subsample reproduces the")
    print("  exhaustive result; report it as a validated design choice.")
    print("  A consistently negative 'mean diff' on the raw correlations would")
    print("  indicate the subsample attenuates them, as attenuation theory")
    print("  predicts; a 'partial' mean diff toward zero would indicate the")
    print("  small residual is partly measurement error rather than signal.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="Inspect data structures without running HoVer-Net.")
    ap.add_argument("--task", choices=["multiclass", "ret_binary"],
                    default="ret_binary")
    ap.add_argument("--n-slides", type=int, default=10)
    args = ap.parse_args()

    if args.probe:
        probe(args.task)
    else:
        main(args.task, args.n_slides)
