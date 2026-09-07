"""
hypothesis2_spatial_analysis.py

HYPOTHESIS 2: immune cells are spatially excluded from high-attention
(tumour-epithelium-dense) regions, rather than merely displaced into the
immediately adjacent tile (which is what Hypothesis 1's compositional
exclusion alone would predict).

SCOPE (documented, not a cohort-level claim): the 7 slides in
hypothesis2_selected_slides.csv, selected via a fixed, pre-existing
criterion (strongest residual partial correlation in the already-completed
v6 analysis), applied before this spatial question was examined. This is a
targeted case-study analysis, not a claim about the full 80-slide cohort.

METHOD, why it differs from the sampling used elsewhere in this chapter:
the correlation analyses (Section 3) use a random 250-tile sample per
slide, which is statistically appropriate for computing a correlation
coefficient. A distance/exclusion question cannot safely use a random
sample, because gaps in coverage would make it impossible to distinguish
"no immune-dense tile nearby" from "the nearest one happened to fall in a
tile we didn't examine." This check therefore uses EVERY tissue tile on
each of the 7 selected slides (the full tiles_v2/ grid), not a sample.

Two stages:
  1. Nucleus detection (GPU, expensive, resumable per-slide) on every
     tissue tile of each selected slide. Run once per slide; nucleus
     counts don't depend on which model's attention you look at.
  2. Attention + distance analysis (fast, CPU-feasible once stage 1 is
     done): attention scores obtained from BOTH final models per slide,
     tiles classified as high/low attention and immune-dense (relative to
     that slide's own distribution), nearest-neighbour distance computed
     from every high-attention tile and every low-attention tile to its
     nearest immune-dense tile, and the two distributions compared.

Usage:
    python hypothesis2_spatial_analysis.py --stage nucleus
    python hypothesis2_spatial_analysis.py --stage distance
    python hypothesis2_spatial_analysis.py --stage nucleus --limit 1   # smoke test
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
import openslide
from scipy.stats import mannwhitneyu, wilcoxon

from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

# ==========================================================================
# CONFIG (architecture and paths match tiic_correlation_v6.py / the manifest)
# ==========================================================================
PROJ = "/cs/student/project_msc/2025/aibh/iconstan/"
TILES_DIR = PROJ + "tiles_v2"
SLIDES_DIR = PROJ + "slides"
RESULTS_DIR = PROJ + "results_v2"
SELECTED_SLIDES_PATH = PROJ + "hypothesis2_selected_slides.csv"
OUTPUT_DIR = PROJ + "hypothesis2_analysis/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDINGS_DIR = PROJ + "embeddings_v2"
EMBED_DIM = 1536
HOVERNET_SUBCROP_SIZE = 256
SUBCROPS_PER_TILE = 4   # 4x4 grid = 16 crops = full coverage, matches v6

TASK_ARCH = {
    "multiclass": {"attn_dim": 512, "fc_dim": 512, "model_file": "best_multiclass_v2_retuned.pt"},
    "ret_binary": {"attn_dim": 512, "fc_dim": 256, "model_file": "best_ret_binary_v2_retuned.pt"},
}

IMMUNE_TYPE_IDS = {2, 3, 4}
EPITHELIAL_TYPE_ID = 1

HIGH_ATTN_PERCENTILE = 80   # top 20% of a slide's own attention distribution
LOW_ATTN_PERCENTILE = 20    # bottom 20%
IMMUNE_DENSE_PERCENTILE = 80  # top 20% of a slide's own immune density distribution

# Microns per pixel at level 1 (~10x). Confirm against your actual per-slide
# level_downsample values if you want exact per-slide conversion; this uses
# the documented approximate value (0.99 mpp) for reporting distances in
# physical units, matching the convention in the spatial-TIL literature.
APPROX_MPP_LEVEL1 = 0.99

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "per_slide"), exist_ok=True)


# ==========================================================================
# ABMIL model (matches manifest / train_abmil_v2_retrained.py)
# ==========================================================================
class GatedAttention(nn.Module):
    def __init__(self, embed_dim=1536, hidden_dim=512, dropout=0.0):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Sigmoid())
        self.w = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        A = self.w(self.drop(self.V(x) * self.U(x)))
        A_softmax = torch.softmax(A, dim=0)
        return torch.mm(A_softmax.T, x), A_softmax


class ABMIL(nn.Module):
    def __init__(self, embed_dim=1536, attn_dim=512, fc_dim=256, num_classes=3, dropout=0.0):
        super().__init__()
        self.attention = GatedAttention(embed_dim, attn_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, fc_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(fc_dim, num_classes)
        )

    def forward(self, x):
        z, A = self.attention(x)
        return self.classifier(z), A


def load_abmil_model(task):
    arch = TASK_ARCH[task]
    num_classes = 3 if task == "multiclass" else 2
    model = ABMIL(EMBED_DIM, arch["attn_dim"], arch["fc_dim"], num_classes, 0.0).to(DEVICE)
    path = os.path.join(RESULTS_DIR, arch["model_file"])
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def get_all_tile_coords(patient):
    for f in os.listdir(TILES_DIR):
        if f.startswith(patient) and f.endswith(".h5"):
            with h5py.File(os.path.join(TILES_DIR, f), "r") as h5:
                coords = h5["coords"][:]
                tile_size = int(h5.attrs["tile_size"])
                level_downsample = float(h5.attrs["level_downsample"])
            return coords, tile_size, level_downsample
    return None, None, None


def get_slide_file(patient):
    for root, dirs, files in os.walk(SLIDES_DIR):
        for f in files:
            if f.startswith(patient) and f.endswith(".svs"):
                return os.path.join(root, f)
    return None


def get_attention_for_all_tiles(model, patient):
    """Forward pass over ALL tile embeddings (not just the 250-sample),
    matched by coordinate to the full tile grid."""
    emb_file = None
    for f in os.listdir(EMBEDDINGS_DIR):
        if f.startswith(patient) and f.endswith(".h5"):
            emb_file = os.path.join(EMBEDDINGS_DIR, f)
            break
    if emb_file is None:
        return None
    with h5py.File(emb_file, "r") as f:
        features = torch.tensor(f["features"][:], dtype=torch.float32).to(DEVICE)
        emb_coords = f["coords"][:]
    with torch.no_grad():
        _, A = model(features)
    attn_scores = A.squeeze().cpu().numpy()
    return {(int(x), int(y)): float(a) for (x, y), a in zip(emb_coords, attn_scores)}


def compute_nuclei_counts_for_tile(slide, x, y, level_downsample, orig_tile_size, segmentor):
    orig_tile_size_l0 = orig_tile_size * level_downsample
    x0_l0 = x * level_downsample
    y0_l0 = y * level_downsample
    step = orig_tile_size_l0 / SUBCROPS_PER_TILE
    sub_crops = [(int(round(x0_l0 + i * step)), int(round(y0_l0 + j * step)))
                 for i in range(SUBCROPS_PER_TILE) for j in range(SUBCROPS_PER_TILE)]
    sub_images = np.array([
        np.array(slide.read_region((cx, cy), 0, (HOVERNET_SUBCROP_SIZE, HOVERNET_SUBCROP_SIZE)).convert("RGB"))
        for (cx, cy) in sub_crops
    ])
    try:
        output = segmentor.run(images=sub_images, patch_mode=True, save_dir=None, output_type="dict")
    except Exception:
        return None
    total_nuclei, immune_nuclei, epithelial_nuclei = 0, 0, 0
    for i in range(len(sub_images)):
        types_i = output["type"][i]
        n = len(types_i) if hasattr(types_i, "__len__") else 0
        total_nuclei += n
        if n > 0:
            immune_nuclei += sum(1 for t in types_i if t in IMMUNE_TYPE_IDS)
            epithelial_nuclei += sum(1 for t in types_i if t == EPITHELIAL_TYPE_ID)
    if total_nuclei == 0:
        return None
    return {"total_nuclei": total_nuclei, "immune_nuclei": immune_nuclei, "epithelial_nuclei": epithelial_nuclei}


def per_slide_path(patient):
    return os.path.join(OUTPUT_DIR, "per_slide", f"{patient}.csv")


# ==========================================================================
# Stage 1: nucleus detection (expensive, GPU, resumable)
# ==========================================================================
def run_nucleus_stage(limit=None):
    selected = pd.read_csv(SELECTED_SLIDES_PATH)
    patients = selected["patient"].tolist()
    if limit:
        patients = patients[:limit]

    todo = [p for p in patients if not os.path.exists(per_slide_path(p))]
    print(f"Selected slides: {len(patients)}. Already done: {len(patients)-len(todo)}. "
          f"Remaining: {len(todo)}")
    if not todo:
        print("Nothing to do for stage 1.")
        return

    print("Loading HoVer-Net (MoNuSAC)...")
    segmentor = MultiTaskSegmentor(
        model="hovernet_fast-monusac", batch_size=4, num_workers=1,
        device="cuda" if DEVICE == "cuda" else "cpu", verbose=False)
    print("  Loaded.")

    overall = time.time()
    for i, patient in enumerate(todo):
        coords, tile_size, level_downsample = get_all_tile_coords(patient)
        slide_path = get_slide_file(patient)
        if coords is None or slide_path is None:
            print(f"  [{i+1}/{len(todo)}] {patient}: missing tile info or slide, skipped")
            continue

        slide = openslide.OpenSlide(slide_path)
        n = len(coords)
        el = (time.time() - overall) / 60
        print(f"\n[{i+1}/{len(todo)}] {patient}: {n} tiles (full coverage), "
              f"elapsed {el:.1f} min")

        rows = []
        t0 = time.time()
        for j, (x, y) in enumerate(coords):
            x, y = int(x), int(y)
            counts = compute_nuclei_counts_for_tile(slide, x, y, level_downsample, tile_size, segmentor)
            if counts is None:
                continue
            total = counts["total_nuclei"]
            if total <= 0:
                continue
            rows.append({
                "patient": patient, "x": x, "y": y, "total_nuclei": total,
                "immune_nuclei": counts["immune_nuclei"],
                "epithelial_nuclei": counts["epithelial_nuclei"],
                "immune_density": counts["immune_nuclei"] / total,
                "epithelial_density": counts["epithelial_nuclei"] / total,
            })
            if (j + 1) % 200 == 0:
                elt = time.time() - t0
                print(f"    {j+1}/{n} tiles ({elt:.0f}s, {elt/(j+1)*n:.0f}s projected for this slide)",
                      flush=True)
        slide.close()

        if len(rows) < 10:
            print(f"    only {len(rows)} valid tiles, skipping save")
            continue
        pd.DataFrame(rows).to_csv(per_slide_path(patient), index=False)
        print(f"    saved {len(rows)} tiles in {time.time()-t0:.0f}s")

    print(f"\nStage 1 complete in {(time.time()-overall)/60:.1f} min")


# ==========================================================================
# Stage 2: attention + distance analysis (fast, no GPU needed if already
# have embeddings; model forward pass is cheap relative to HoVer-Net)
# ==========================================================================
def nearest_immune_distance(tile_x, tile_y, immune_dense_coords, mpp):
    """Euclidean distance in microns from one tile to the nearest immune-dense tile."""
    if len(immune_dense_coords) == 0:
        return np.nan
    dx = immune_dense_coords[:, 0] - tile_x
    dy = immune_dense_coords[:, 1] - tile_y
    dists_px = np.sqrt(dx**2 + dy**2)
    dists_px = dists_px[dists_px > 0]  # exclude the tile itself if present
    if len(dists_px) == 0:
        return np.nan
    return dists_px.min() * mpp


def run_distance_stage():
    selected = pd.read_csv(SELECTED_SLIDES_PATH)
    patients = selected["patient"].tolist()

    models = {task: load_abmil_model(task) for task in TASK_ARCH}

    per_slide_results = []
    for patient in patients:
        p = per_slide_path(patient)
        if not os.path.exists(p):
            print(f"  {patient}: no nucleus data yet (run --stage nucleus first), skipped")
            continue
        tiles = pd.read_csv(p)
        print(f"\n{patient}: {len(tiles)} tiles with nucleus data")

        for task, model in models.items():
            attn_map = get_attention_for_all_tiles(model, patient)
            if attn_map is None:
                print(f"    {task}: no embeddings found, skipped")
                continue
            tiles[f"attn_{task}"] = tiles.apply(
                lambda r: attn_map.get((int(r["x"]), int(r["y"])), np.nan), axis=1)

        tiles.to_csv(os.path.join(OUTPUT_DIR, "per_slide", f"{patient}_with_attn.csv"), index=False)

        immune_thresh = np.percentile(tiles["immune_density"].dropna(), IMMUNE_DENSE_PERCENTILE)
        immune_dense = tiles[tiles["immune_density"] >= immune_thresh]
        immune_coords = immune_dense[["x", "y"]].values.astype(float)

        for task in TASK_ARCH:
            col = f"attn_{task}"
            if col not in tiles.columns or tiles[col].isna().all():
                continue
            valid = tiles.dropna(subset=[col])
            high_thresh = np.percentile(valid[col], HIGH_ATTN_PERCENTILE)
            low_thresh = np.percentile(valid[col], LOW_ATTN_PERCENTILE)
            high_tiles = valid[valid[col] >= high_thresh]
            low_tiles = valid[valid[col] <= low_thresh]

            high_dists = [nearest_immune_distance(r["x"], r["y"], immune_coords, APPROX_MPP_LEVEL1)
                          for _, r in high_tiles.iterrows()]
            low_dists = [nearest_immune_distance(r["x"], r["y"], immune_coords, APPROX_MPP_LEVEL1)
                         for _, r in low_tiles.iterrows()]
            high_dists = [d for d in high_dists if not np.isnan(d)]
            low_dists = [d for d in low_dists if not np.isnan(d)]

            if len(high_dists) < 3 or len(low_dists) < 3:
                print(f"    {task}: too few high/low-attention tiles with valid distances, skipped")
                continue

            u, pval = mannwhitneyu(high_dists, low_dists, alternative="two-sided")
            print(f"    {task}: high-attn mean dist to nearest immune-dense tile = "
                  f"{np.mean(high_dists):.1f} um (n={len(high_dists)}), "
                  f"low-attn = {np.mean(low_dists):.1f} um (n={len(low_dists)}), "
                  f"Mann-Whitney p={pval:.3g}")

            per_slide_results.append({
                "patient": patient, "task": task,
                "n_high": len(high_dists), "n_low": len(low_dists),
                "mean_dist_high_um": np.mean(high_dists),
                "median_dist_high_um": np.median(high_dists),
                "mean_dist_low_um": np.mean(low_dists),
                "median_dist_low_um": np.median(low_dists),
                "mannwhitney_p": pval,
            })

    if not per_slide_results:
        print("\nNo results produced. Check stage 1 completed and embeddings exist.")
        return

    results_df = pd.DataFrame(per_slide_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "hypothesis2_distance_results.csv"), index=False)

    print(f"\n{'='*70}")
    print(f"SUMMARY ACROSS {results_df['patient'].nunique()} SLIDES")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))

    for task in TASK_ARCH:
        sub = results_df[results_df["task"] == task]
        if len(sub) < 3:
            continue
        print(f"\n{task}: paired comparison across {len(sub)} slides "
              f"(high-attn mean dist vs low-attn mean dist per slide)")
        try:
            stat, p = wilcoxon(sub["mean_dist_high_um"], sub["mean_dist_low_um"])
            print(f"  Wilcoxon signed-rank: stat={stat:.2f}, p={p:.3g}")
        except ValueError as e:
            print(f"  Wilcoxon test not computable ({e}) - report descriptively instead.")
        print(f"  mean(high-attn distance) across slides: {sub['mean_dist_high_um'].mean():.1f} um")
        print(f"  mean(low-attn distance) across slides:  {sub['mean_dist_low_um'].mean():.1f} um")
        print(f"  -> if high > low consistently, supports Hypothesis 2 (spatial exclusion).")
        print(f"  -> if similar, does not support Hypothesis 2 beyond Hypothesis 1 alone.")

    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'hypothesis2_distance_results.csv')}")
    print(f"\nREMINDER: this is a {results_df['patient'].nunique()}-slide targeted case study on "
          f"pre-selected outlier slides, not a cohort-level test. Report accordingly.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["nucleus", "distance"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="Limit slides (stage=nucleus only)")
    args = ap.parse_args()

    if args.stage == "nucleus":
        run_nucleus_stage(limit=args.limit)
    else:
        run_distance_stage()
