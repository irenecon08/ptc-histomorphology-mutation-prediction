"""
shape_descriptor_analysis.py

Supervisor's suggestion: take the tiles the model attends to, segment the
epithelial nuclei, and compare classic shape descriptors (area,
circularity, solidity, eccentricity) between cases.

FULLY SELF-CONTAINED: this script loads its own trained ABMIL models and
computes its own attention scores directly from embeddings. It has no
dependency on hypothesis2_spatial_analysis.py or any other analysis
script's output - only on the project's core data (tiles_v2/, embeddings_v2/,
results_v2/), so it can run on any slide regardless of what other analyses
have or haven't been run on it.

This produces ONE shared dataset (per-epithelial-nucleus shape descriptors,
tagged with tile-level attention and slide-level genotype), then runs TWO
separate, genuinely different comparisons from it:

  Option A: grouped by ATTENTION level (high-attention tiles vs low-attention
            tiles, within slide). Tests whether the model's attention relates
            to nuclear shape, not just density/count.

  Option B: grouped by GENOTYPE (BRAF vs RAS vs RET). Tests whether epithelial
            nuclear shape differs by driver mutation, independent of the
            model - grounded in established diagnostic pathology (BRAF V600E
            PTC is classically associated with distinctive nuclear features:
            grooves, irregular contours, "Orphan Annie eye" nuclei).

TWO SLIDE LISTS available via --slide-list, kept as SEPARATE samples with
different selection logic (do not pool their results):
    h2        7 slides, selected as strongest residual outliers from the
              Hypothesis 2 partial-correlation analysis.
    genotype  30 slides, genotype-stratified (Design A): all RAS, all RET,
              a random-seeded sample of BRAF and DriverNeg. See
              genotype_stratified_slide_selection.py.

Both comparisons should be read with the same case-study caveat: neither
slide list is a random or representative sample of the full test cohort.

STAGES (run in order):
    --stage probe               Inspect the HoVer-Net output structure for
                                 one sub-crop, to confirm the contour field
                                 name/shape before committing to a full run.
    --stage time-one             Time full shape-descriptor extraction for
                                 one tile.
    --stage extract              Full run: extract descriptors for every
                                 epithelial nucleus, every tile, every slide
                                 in the chosen list. Resumable per-slide.
    --stage analyze              Option A and Option B on pooled nuclei
                                 (pseudoreplicated - see analyze-aggregated).
    --stage analyze-aggregated   Corrected version: aggregates to one value
                                 per slide (Option A) / per patient (Option B)
                                 before testing.

Usage:
    python shape_descriptor_analysis.py --stage probe --slide-list h2
    python shape_descriptor_analysis.py --stage extract --slide-list genotype
    python shape_descriptor_analysis.py --stage analyze-aggregated --slide-list genotype
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import openslide
from scipy.stats import mannwhitneyu, kruskal

from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

try:
    import cv2
except ImportError:
    sys.exit("This script needs opencv. Install with:\n"
              "  pip install opencv-python-headless --break-system-packages")

# ==========================================================================
# CONFIG
# ==========================================================================
PROJ = "/cs/student/project_msc/2025/aibh/iconstan/"
SLIDES_DIR = PROJ + "slides"
TILES_DIR = PROJ + "tiles_v2"
EMBEDDINGS_DIR = PROJ + "embeddings_v2"
RESULTS_DIR = PROJ + "results_v2"
LABELS_PATH = PROJ + "final_labels.csv"
OUTPUT_DIR = PROJ + "shape_descriptor_analysis/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM = 1536
TASK_ARCH = {
    "multiclass": {"attn_dim": 512, "fc_dim": 512, "model_file": "best_multiclass_v2_retuned.pt"},
    "ret_binary": {"attn_dim": 512, "fc_dim": 256, "model_file": "best_ret_binary_v2_retuned.pt"},
}

# Two available slide-selection files, chosen via --slide-list at runtime.
# These are SEPARATE samples with different selection logic and should not
# be pooled or presented as one uniform sample (see genotype_stratified_
# slide_selection.py docstring for why).
SLIDE_LISTS = {
    "h2":       PROJ + "hypothesis2_selected_slides.csv",       # original 7, residual-outlier selected
    "genotype": PROJ + "genotype_stratified_slides.csv",         # new 30, genotype-stratified
}

HOVERNET_SUBCROP_SIZE = 256
SUBCROPS_PER_TILE = 4  # matches v6 full coverage
EPITHELIAL_TYPE_ID = 1
HIGH_ATTN_PERCENTILE = 80
LOW_ATTN_PERCENTILE = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "per_slide"), exist_ok=True)


# ==========================================================================
# ABMIL model - copied verbatim from tiic_correlation_v6.py (verified
# correct against results_v2_model_manifest.json). This script computes its
# own attention independently; it does NOT depend on
# hypothesis2_spatial_analysis.py or its output in any way.
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


_LOADED_MODELS = {}  # cache, so each task's model is only loaded once per run


def load_abmil_model(task):
    if task in _LOADED_MODELS:
        return _LOADED_MODELS[task]
    arch = TASK_ARCH[task]
    num_classes = 3 if task == "multiclass" else 2
    model = ABMIL(EMBED_DIM, arch["attn_dim"], arch["fc_dim"], num_classes, 0.0).to(DEVICE)
    path = os.path.join(RESULTS_DIR, arch["model_file"])
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)  # strict, matches v6 - any mismatch raises loudly
    model.eval()
    _LOADED_MODELS[task] = model
    return model


def compute_attention_for_slide(patient):
    """
    Self-contained attention computation: loads embeddings, runs both trained
    models, returns {(x, y): attn_multiclass} and {(x, y): attn_ret_binary}.
    No dependency on hypothesis2_spatial_analysis.py or any prior H2 run.
    """
    emb_file = None
    for f in os.listdir(EMBEDDINGS_DIR):
        if f.startswith(patient) and f.endswith(".h5"):
            emb_file = os.path.join(EMBEDDINGS_DIR, f)
            break
    if emb_file is None:
        return None, None

    import h5py
    with h5py.File(emb_file, "r") as f:
        features = torch.tensor(f["features"][:], dtype=torch.float32).to(DEVICE)
        emb_coords = f["coords"][:]

    attn_by_task = {}
    for task in TASK_ARCH:
        model = load_abmil_model(task)
        with torch.no_grad():
            _, A = model(features)
        attn_scores = A.squeeze().cpu().numpy()
        attn_by_task[task] = {(int(x), int(y)): float(a)
                               for (x, y), a in zip(emb_coords, attn_scores)}
    return attn_by_task.get("multiclass"), attn_by_task.get("ret_binary")


# ==========================================================================
# Data access (self-contained, no dependency on other analysis scripts)
# ==========================================================================
def get_slide_file(patient):
    for root, dirs, files in os.walk(SLIDES_DIR):
        for f in files:
            if f.startswith(patient) and f.endswith(".svs"):
                return os.path.join(root, f)
    return None


def get_tile_positions(patient):
    """
    Tile x/y positions and geometry, sourced directly from tiles_v2/ .h5
    files. No dependency on any other analysis script's output.
    """
    for f in os.listdir(TILES_DIR):
        if f.startswith(patient) and f.endswith(".h5"):
            import h5py
            with h5py.File(os.path.join(TILES_DIR, f), "r") as h5:
                coords = h5["coords"][:]
                tile_size = int(h5.attrs["tile_size"])
                level_downsample = float(h5.attrs["level_downsample"])
            df = pd.DataFrame(coords, columns=["x", "y"])
            return df, tile_size, level_downsample
    return None, None, None


def per_slide_shape_path(patient):
    return os.path.join(OUTPUT_DIR, "per_slide", f"{patient}_shapes.csv")


# ==========================================================================
# Shape descriptors from a contour (cv2 point array)
# ==========================================================================
def touches_boundary(cnt, crop_size=HOVERNET_SUBCROP_SIZE, edge_margin=2):
    """True if this contour comes within edge_margin pixels of the sub-crop
    boundary, meaning the nucleus is likely clipped by the sub-crop tiling
    and its shape cannot be trusted."""
    pts = cnt.reshape(-1, 2)
    return (pts[:, 0].min() <= edge_margin or pts[:, 0].max() >= crop_size - 1 - edge_margin or
            pts[:, 1].min() <= edge_margin or pts[:, 1].max() >= crop_size - 1 - edge_margin)


def descriptors_from_contour(cnt):
    """
    area, circularity, solidity, eccentricity from a single nucleus contour.
    Returns None if the contour is degenerate (too small, or too few points
    to fit an ellipse). Boundary-clipping is checked separately by the
    caller via touches_boundary(), so discard reasons can be counted
    independently.
    """
    area = cv2.contourArea(cnt)
    if area < 5:
        return None
    perimeter = cv2.arcLength(cnt, closed=True)
    if perimeter == 0:
        return None
    circularity = 4 * np.pi * area / (perimeter ** 2)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else np.nan

    if len(cnt) < 5:
        eccentricity = np.nan  # fitEllipse needs >= 5 points
    else:
        (_, _), (minor_axis, major_axis), _ = cv2.fitEllipse(cnt)
        if major_axis <= 0 or minor_axis > major_axis:
            eccentricity = np.nan
        else:
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)

    return {"area": area, "circularity": circularity,
            "solidity": solidity, "eccentricity": eccentricity}


# ==========================================================================
# Probe: confirm HoVer-Net's output structure before committing to it
# ==========================================================================
def probe(slide_list_path):
    print("Loading HoVer-Net (MoNuSAC)...")
    segmentor = MultiTaskSegmentor(
        model="hovernet_fast-monusac", batch_size=4, num_workers=1,
        device="cuda" if DEVICE == "cuda" else "cpu", verbose=False)
    print("  Loaded.")

    selected = pd.read_csv(slide_list_path)
    patient = selected["patient"].iloc[0]
    slide_path = get_slide_file(patient)
    if slide_path is None:
        sys.exit(f"No slide file found for {patient}")

    slide = openslide.OpenSlide(slide_path)
    # grab a real sub-crop from somewhere near the slide centre
    w0, h0 = slide.dimensions
    cx, cy = w0 // 2, h0 // 2
    img = np.array(slide.read_region((cx, cy), 0, (HOVERNET_SUBCROP_SIZE, HOVERNET_SUBCROP_SIZE)).convert("RGB"))
    slide.close()

    output = segmentor.run(images=np.array([img]), patch_mode=True, save_dir=None, output_type="dict")

    print("\n" + "=" * 70)
    print("OUTPUT STRUCTURE")
    print("=" * 70)
    print(f"Top-level keys: {list(output.keys())}")
    for k, v in output.items():
        print(f"\n  '{k}': type={type(v).__name__}", end="")
        if hasattr(v, "__len__"):
            print(f", len={len(v)}", end="")
            if len(v) > 0:
                v0 = v[0]
                print(f"\n    [0]: type={type(v0).__name__}", end="")
                if hasattr(v0, "__len__"):
                    print(f", len={len(v0)}", end="")
                    if len(v0) > 0:
                        print(f"\n      [0][0]: {v0[0]!r}"[:300])
        print()

    print("\n" + "=" * 70)
    print("LOOKING FOR A CONTOUR/COORD FIELD")
    print("=" * 70)
    candidates = [k for k in output.keys() if k.lower() in
                  ("contour", "contours", "coord", "coords", "cnt", "box", "boxes")]
    print(f"Candidate field names found: {candidates}")
    if not candidates:
        print("No obvious contour field found among top-level keys.")
        print("Full key list above - inspect manually and tell me what you see,")
        print("the extraction code will need updating to match the real structure.")


# ==========================================================================
# Extraction: full shape descriptors for epithelial nuclei, one tile
# ==========================================================================
def extract_tile_shapes(slide, x, y, level_downsample, orig_tile_size, segmentor,
                          contour_field):
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
        return [], 0, 0, 0

    descriptors = []
    n_epithelial_total = 0
    n_discarded_edge = 0
    n_discarded_degenerate = 0
    for i in range(len(sub_images)):
        types_i = output["type"][i]
        contours_i = output[contour_field][i]
        for t, cnt in zip(types_i, contours_i):
            if t != EPITHELIAL_TYPE_ID:
                continue
            n_epithelial_total += 1
            cnt_arr = np.array(cnt, dtype=np.int32).reshape(-1, 1, 2)
            if touches_boundary(cnt_arr):
                n_discarded_edge += 1
                continue
            d = descriptors_from_contour(cnt_arr)
            if d is not None:
                descriptors.append(d)
            else:
                n_discarded_degenerate += 1
    return descriptors, n_epithelial_total, n_discarded_edge, n_discarded_degenerate


def run_extraction(contour_field, slide_list_path, limit=None):
    selected = pd.read_csv(slide_list_path)
    patients = selected["patient"].tolist()
    if limit:
        patients = patients[:limit]

    todo = [p for p in patients if not os.path.exists(per_slide_shape_path(p))]
    print(f"Selected slides: {len(patients)}. Already done: {len(patients)-len(todo)}. "
          f"Remaining: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    print("Loading HoVer-Net (MoNuSAC)...")
    segmentor = MultiTaskSegmentor(
        model="hovernet_fast-monusac", batch_size=4, num_workers=1,
        device="cuda" if DEVICE == "cuda" else "cpu", verbose=False)
    print("  Loaded.")

    for patient in todo:
        tiles, tile_size, level_downsample = get_tile_positions(patient)
        if tiles is None:
            print(f"  {patient}: no tile data found in tiles_v2/, skipped")
            continue

        slide_path = get_slide_file(patient)

        slide = openslide.OpenSlide(slide_path)
        n = len(tiles)
        print(f"\n{patient}: {n} tiles")
        t0 = time.time()

        rows = []
        total_epithelial = 0
        total_edge_discarded = 0
        total_degenerate_discarded = 0
        for j, (_, r) in enumerate(tiles.iterrows()):
            x, y = int(r["x"]), int(r["y"])
            descs, n_epi, n_edge, n_degen = extract_tile_shapes(
                slide, x, y, level_downsample, tile_size, segmentor, contour_field)
            total_epithelial += n_epi
            total_edge_discarded += n_edge
            total_degenerate_discarded += n_degen
            for d in descs:
                d["patient"] = patient
                d["x"] = x
                d["y"] = y
                rows.append(d)
            if (j + 1) % 200 == 0:
                el = time.time() - t0
                print(f"    {j+1}/{n} tiles ({el:.0f}s, {el/(j+1)*n:.0f}s projected)", flush=True)

        slide.close()
        pd.DataFrame(rows).to_csv(per_slide_shape_path(patient), index=False)
        edge_pct = 100 * total_edge_discarded / total_epithelial if total_epithelial else 0
        print(f"    epithelial nuclei detected: {total_epithelial}")
        print(f"    discarded (boundary-clipped): {total_edge_discarded} ({edge_pct:.1f}%)")
        print(f"    discarded (degenerate): {total_degenerate_discarded}")
        print(f"    saved {len(rows)} usable epithelial nuclei in {time.time()-t0:.0f}s")


def time_one(contour_field, slide_list_path):
    selected = pd.read_csv(slide_list_path)
    n_slides_in_list = len(selected)
    patient = selected["patient"].iloc[0]
    tiles, tile_size, level_downsample = get_tile_positions(patient)
    if tiles is None:
        sys.exit(f"No tile data found for {patient} in tiles_v2/")

    print("Loading HoVer-Net (MoNuSAC)...")
    segmentor = MultiTaskSegmentor(
        model="hovernet_fast-monusac", batch_size=4, num_workers=1,
        device="cuda" if DEVICE == "cuda" else "cpu", verbose=False)
    print("  Loaded.")

    slide_path = get_slide_file(patient)
    slide = openslide.OpenSlide(slide_path)

    row = tiles.iloc[0]
    x, y = int(row["x"]), int(row["y"])

    t0 = time.time()
    descs, n_epi, n_edge, n_degen = extract_tile_shapes(
        slide, x, y, level_downsample, tile_size, segmentor, contour_field)
    elapsed = time.time() - t0
    slide.close()

    edge_pct = 100 * n_edge / n_epi if n_epi else 0
    print(f"\nONE TILE TIMING: {elapsed:.2f}s")
    print(f"  epithelial nuclei detected: {n_epi}")
    print(f"  discarded (boundary-clipped): {n_edge} ({edge_pct:.1f}%)")
    print(f"  discarded (degenerate): {n_degen}")
    print(f"  usable descriptors extracted: {len(descs)}")
    print(f"Projected for ~{len(tiles)} tiles x {n_slides_in_list} slides in this list: "
          f"{elapsed * len(tiles) * n_slides_in_list / 3600:.1f} hours "
          f"(rough estimate, actual per-slide tile counts vary)")


def run_aggregated_analysis(slide_list_path, output_dir):
    """
    Corrects the pseudoreplication problem in run_analysis(): pooling ~1.2M
    individual nuclei treats each nucleus as an independent observation,
    which it is not (nuclei within a slide are highly correlated), producing
    p-values that are effectively meaningless at that scale. This instead
    aggregates to one value per independent unit before testing:

      Option A: one median value per slide, per attention group (high/low),
                per task -> paired Wilcoxon signed-rank across slides,
                matching the approach already used for Hypothesis 2's
                distance analysis.

      Option B: one median value per PATIENT, per genotype -> Kruskal-Wallis
                across genotype groups with n>=3 patients, BH-corrected
                across descriptors. Groups with n<3 are excluded from the
                formal test and reported descriptively only.
    """
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    selected = pd.read_csv(slide_list_path)
    labels = pd.read_csv(LABELS_PATH)
    labels["_bc"] = labels["patient"].astype(str).str[:12]

    all_shapes = []
    missing_attn = []
    for patient in selected["patient"]:
        p = per_slide_shape_path(patient)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        attn_mc, attn_rb = compute_attention_for_slide(patient)
        if attn_mc is not None:
            df["attn_multiclass"] = df.apply(
                lambda r: attn_mc.get((int(r["x"]), int(r["y"])), np.nan), axis=1)
            df["attn_ret_binary"] = df.apply(
                lambda r: attn_rb.get((int(r["x"]), int(r["y"])), np.nan), axis=1)
        else:
            missing_attn.append(patient)
        all_shapes.append(df)

    if missing_attn:
        print(f"WARNING: {len(missing_attn)} slides have no embeddings found, "
              f"excluded from Option A: {missing_attn}")

    shapes = pd.concat(all_shapes, ignore_index=True)
    shapes["_bc"] = shapes["patient"].astype(str).str[:12]
    shapes = shapes.merge(labels[["_bc", "mutation_label", "RET"]], on="_bc", how="left")
    geno_map = {"BRAF_V600E": "BRAF", "RAS": "RAS", "Other": "Other"}
    shapes["genotype"] = shapes["mutation_label"].map(geno_map).fillna("Other")
    shapes.loc[shapes["RET"] == 1, "genotype"] = "RET"

    descriptors = ["area", "circularity", "solidity", "eccentricity"]

    # ---- Option A, per-slide aggregation ----
    print("=" * 70)
    print(f"OPTION A (AGGREGATED): per-slide median, high- vs low-attention")
    print(f"Unit of analysis: slide. Paired Wilcoxon signed-rank.")
    print("=" * 70)

    for task in ["multiclass", "ret_binary"]:
        col = f"attn_{task}"
        if col not in shapes.columns:
            continue
        print(f"\n--- {task} ---")
        sub = shapes.dropna(subset=[col])
        per_slide_hi, per_slide_lo = {}, {}
        for patient, grp in sub.groupby("patient"):
            hi_thresh = grp[col].quantile(HIGH_ATTN_PERCENTILE / 100)
            lo_thresh = grp[col].quantile(LOW_ATTN_PERCENTILE / 100)
            hi = grp[grp[col] >= hi_thresh]
            lo = grp[grp[col] <= lo_thresh]
            per_slide_hi[patient] = hi[descriptors].median()
            per_slide_lo[patient] = lo[descriptors].median()

        hi_df = pd.DataFrame(per_slide_hi).T
        lo_df = pd.DataFrame(per_slide_lo).T
        n_slides_this_task = len(hi_df)
        print(f"  Per-slide median values (n={n_slides_this_task} slides):")
        print(f"  {'patient':16s} " + "  ".join(f"{d:>12s}(hi/lo)" for d in descriptors))
        for patient in hi_df.index:
            vals = "  ".join(f"{hi_df.loc[patient,d]:.3f}/{lo_df.loc[patient,d]:.3f}"
                              for d in descriptors)
            print(f"  {patient:16s} {vals}")

        # Save the full per-slide table so these exact numbers can be verified
        # independently (e.g. opened directly in a spreadsheet), with no
        # transcription step anywhere between the computation and the record.
        combined = hi_df.add_suffix("_hi").join(lo_df.add_suffix("_lo"))
        combined_path = os.path.join(output_dir, f"option_a_per_slide_{task}.csv")
        combined.to_csv(combined_path)
        print(f"  Saved per-slide table: {combined_path}")

        # Exact summary statistics, computed directly by pandas from the
        # same hi_df/lo_df used for the test below - this is the authoritative
        # figure for any table quoting "median high vs low" values.
        print(f"\n  EXACT summary (median and mean of the {n_slides_this_task} "
              f"per-slide medians, computed directly - use these values, not "
              f"any hand-transcribed estimate):")
        for d in descriptors:
            print(f"    {d:12s} hi: median={hi_df[d].median():.4f} mean={hi_df[d].mean():.4f}"
                  f"   lo: median={lo_df[d].median():.4f} mean={lo_df[d].mean():.4f}")

        pvals = []
        for d in descriptors:
            try:
                stat, p = wilcoxon(hi_df[d], lo_df[d])
            except ValueError:
                p = float("nan")
            pvals.append(p)
        rej, p_adj, _, _ = multipletests([p for p in pvals if not np.isnan(p)],
                                          alpha=0.05, method="fdr_bh")
        adj_iter = iter(zip(rej, p_adj))
        print(f"\n  Paired Wilcoxon (n={n_slides_this_task} slides), BH-corrected "
              f"across 4 descriptors:")
        for d, p in zip(descriptors, pvals):
            if np.isnan(p):
                print(f"    {d:12s} p_raw=nan (could not compute, e.g. all differences zero)")
                continue
            sig, pa = next(adj_iter)
            flag = " *" if sig else ""
            print(f"    {d:12s} p_raw={p:.4f}  p_BH={pa:.4f}{flag}")

    # ---- Option B, per-patient aggregation ----
    print("\n" + "=" * 70)
    print("OPTION B (AGGREGATED): per-patient median, by genotype")
    print("Unit of analysis: patient.")
    print("=" * 70)

    per_patient = shapes.groupby(["patient", "genotype"])[descriptors].median().reset_index()
    print(f"\n  Per-patient median values (n={len(per_patient)} patients):")
    print(per_patient.to_string(index=False))

    counts = per_patient["genotype"].value_counts()
    print(f"\n  Genotype counts in this sample: {dict(counts)}")

    MIN_N_FOR_TEST = 3  # below this, a genotype group cannot support Kruskal-Wallis at all
    testable = [g for g in counts.index if counts[g] >= MIN_N_FOR_TEST]
    untestable = [g for g in counts.index if counts[g] < MIN_N_FOR_TEST]

    if untestable:
        print(f"\n  NOTE: {untestable} excluded from the formal test "
              f"(n<{MIN_N_FOR_TEST}), reported descriptively above only.")

    if len(testable) >= 2:
        print(f"\n  Kruskal-Wallis across {testable} "
              f"(n={[int(counts[g]) for g in testable]}), per descriptor, "
              f"BH-corrected:")
        b_pvals, b_meta = [], []
        for d in descriptors:
            groups = [per_patient.loc[per_patient["genotype"] == g, d].dropna()
                      for g in testable]
            h, p = kruskal(*groups)
            b_pvals.append(p)
            b_meta.append((d, h, groups))
        rej, p_adj, _, _ = multipletests(b_pvals, alpha=0.05, method="fdr_bh")
        for (d, h, groups), p_raw, pa, sig in zip(b_meta, b_pvals, p_adj, rej):
            flag = " *" if sig else ""
            medians = "  ".join(f"{g}={grp.median():.3f}" for g, grp in zip(testable, groups))
            print(f"    {d:12s} {medians}")
            print(f"      H={h:.3f}  p_raw={p_raw:.4f}  p_BH={pa:.4f}{flag}")
        print(f"\n  Unit of analysis: patient (n={sum(counts[g] for g in testable)} "
              f"across tested genotypes). Modest sample for a multi-group comparison; "
              f"case-study result, not a cohort-level claim.")
    else:
        print(f"\n  Fewer than 2 genotype groups have n>={MIN_N_FOR_TEST}; "
              f"no formal test possible, reported descriptively only.")

    per_patient.to_csv(os.path.join(output_dir, "option_b_per_patient.csv"), index=False)
    print(f"\nSaved: {os.path.join(output_dir, 'option_b_per_patient.csv')}")
    print("\nREMINDER: case-study sample, not a cohort-level claim.")


# ==========================================================================
# Analysis: Option A (by attention) and Option B (by genotype)
# ==========================================================================
def run_analysis(slide_list_path, output_dir):
    selected = pd.read_csv(slide_list_path)
    labels = pd.read_csv(LABELS_PATH)
    labels["_bc"] = labels["patient"].astype(str).str[:12]

    all_shapes = []
    missing_attn = []
    for patient in selected["patient"]:
        p = per_slide_shape_path(patient)
        if not os.path.exists(p):
            print(f"  {patient}: no shape data yet, skipped")
            continue
        df = pd.read_csv(p)
        attn_mc, attn_rb = compute_attention_for_slide(patient)
        if attn_mc is not None:
            df["attn_multiclass"] = df.apply(
                lambda r: attn_mc.get((int(r["x"]), int(r["y"])), np.nan), axis=1)
            df["attn_ret_binary"] = df.apply(
                lambda r: attn_rb.get((int(r["x"]), int(r["y"])), np.nan), axis=1)
        else:
            missing_attn.append(patient)
        all_shapes.append(df)

    if missing_attn:
        print(f"WARNING: {len(missing_attn)} slides have no embeddings found, "
              f"excluded from Option A: {missing_attn}")

    if not all_shapes:
        sys.exit("No shape data found. Run --stage extract first.")

    shapes = pd.concat(all_shapes, ignore_index=True)
    shapes["_bc"] = shapes["patient"].astype(str).str[:12]
    shapes = shapes.merge(labels[["_bc", "mutation_label", "RET"]], on="_bc", how="left")

    descriptors = ["area", "circularity", "solidity", "eccentricity"]

    print("=" * 70)
    print(f"SHAPE DESCRIPTOR DATASET: {len(shapes)} epithelial nuclei, "
          f"{shapes['patient'].nunique()} slides")
    print("=" * 70)

    # ---- Option A: by attention level, per task ----
    print("\n" + "=" * 70)
    print("OPTION A: shape descriptors by attention level (within-slide)")
    print("=" * 70)
    option_a_results = []
    for task in ["multiclass", "ret_binary"]:
        col = f"attn_{task}"
        if col not in shapes.columns:
            continue
        sub = shapes.dropna(subset=[col])
        for patient, grp in sub.groupby("patient"):
            if len(grp) < 20:
                continue
            hi_thresh = grp[col].quantile(HIGH_ATTN_PERCENTILE / 100)
            lo_thresh = grp[col].quantile(LOW_ATTN_PERCENTILE / 100)
            hi = grp[grp[col] >= hi_thresh]
            lo = grp[grp[col] <= lo_thresh]
            if len(hi) < 10 or len(lo) < 10:
                continue
            for d in descriptors:
                hi_vals = hi[d].dropna()
                lo_vals = lo[d].dropna()
                if len(hi_vals) < 5 or len(lo_vals) < 5:
                    continue
                u, p = mannwhitneyu(hi_vals, lo_vals, alternative="two-sided")
                option_a_results.append({
                    "task": task, "patient": patient, "descriptor": d,
                    "hi_median": hi_vals.median(), "lo_median": lo_vals.median(),
                    "p_raw": p,
                })

    if option_a_results:
        a_df = pd.DataFrame(option_a_results)
        from statsmodels.stats.multitest import multipletests
        rej, p_adj, _, _ = multipletests(a_df["p_raw"].values, alpha=0.05, method="fdr_bh")
        a_df["p_bh"] = p_adj
        a_df["significant"] = rej
        for task in a_df["task"].unique():
            print(f"\n--- {task} ---")
            t_df = a_df[a_df["task"] == task]
            for patient in t_df["patient"].unique():
                p_df = t_df[t_df["patient"] == patient]
                print(f"  {patient}:")
                for _, row in p_df.iterrows():
                    flag = " *" if row["significant"] else ""
                    print(f"    {row['descriptor']:12s} hi_median={row['hi_median']:.3f}  "
                          f"lo_median={row['lo_median']:.3f}  "
                          f"p_raw={row['p_raw']:.3g}  p_BH={row['p_bh']:.3g}{flag}")
        print(f"\n  Total tests: {len(a_df)}. Significant after BH correction "
              f"(FDR 5%): {a_df['significant'].sum()}")
        a_df.to_csv(os.path.join(output_dir, "option_a_results.csv"), index=False)
    else:
        print("  No results (insufficient data per slide).")

    # ---- Option B: by genotype ----
    print("\n" + "=" * 70)
    print("OPTION B: shape descriptors by genotype")
    print("=" * 70)
    geno_map = {"BRAF_V600E": "BRAF", "RAS": "RAS", "Other": "Other"}
    shapes["genotype"] = shapes["mutation_label"].map(geno_map).fillna("Other")
    shapes.loc[shapes["RET"] == 1, "genotype"] = "RET"

    option_b_results = []
    for d in descriptors:
        groups, names = [], []
        for g in ["BRAF", "RAS", "RET", "Other"]:
            vals = shapes.loc[shapes["genotype"] == g, d].dropna()
            if len(vals) >= 10:
                groups.append(vals)
                names.append(g)
        if len(groups) < 2:
            continue
        h, p = kruskal(*groups)
        option_b_results.append({"descriptor": d, "H": h, "p_raw": p,
                                  "groups": {nm: (g.median(), g.mean(), len(g))
                                             for nm, g in zip(names, groups)}})

    if option_b_results:
        from statsmodels.stats.multitest import multipletests
        p_raws = [r["p_raw"] for r in option_b_results]
        rej, p_adj, _, _ = multipletests(p_raws, alpha=0.05, method="fdr_bh")
        for r, pa, sig in zip(option_b_results, p_adj, rej):
            print(f"\n  {r['descriptor']}:")
            for nm, (med, mean, n) in r["groups"].items():
                print(f"    {nm:6s} n={n:5d}  median={med:.3f}  mean={mean:.3f}")
            flag = " *" if sig else ""
            print(f"    Kruskal-Wallis: H={r['H']:.3f}, p_raw={r['p_raw']:.3g}, "
                  f"p_BH={pa:.3g}{flag}")
        n_sig = sum(rej)
        print(f"\n  Significant after BH correction (FDR 5%): {n_sig} / {len(option_b_results)}")
    else:
        print("  No results (insufficient data per group).")

    shapes.to_csv(os.path.join(output_dir, "all_epithelial_shapes.csv"), index=False)
    print(f"\nSaved full dataset: {os.path.join(output_dir, 'all_epithelial_shapes.csv')}")
    print("\nREMINDER: 7-slide case study, not a cohort-level claim (same scope as Hypothesis 2).")


# ==========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["probe", "time-one", "extract", "analyze",
                                         "analyze-aggregated"], required=True)
    ap.add_argument("--slide-list", choices=list(SLIDE_LISTS.keys()), default="h2",
                     help="'h2' = original 7 slides (residual-outlier selected). "
                          "'genotype' = new 30 slides (genotype-stratified, Design A). "
                          "These are SEPARATE samples - do not pool results across them.")
    ap.add_argument("--contour-field", default="contours",
                     help="Field name in HoVer-Net output holding per-nucleus contours "
                          "(confirmed via --stage probe: 'contours')")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    slide_list_path = SLIDE_LISTS[args.slide_list]
    if not os.path.exists(slide_list_path):
        sys.exit(f"Slide list not found: {slide_list_path}\n"
                  f"For --slide-list genotype, run genotype_stratified_slide_selection.py first.")

    # Keep the two slide sets' aggregate outputs separate, per-slide shape
    # files are safely shared/reused across both (keyed by patient ID only,
    # see get_tile_positions() and the resumability check in run_extraction).
    if args.slide_list == "genotype":
        OUTPUT_DIR_RUN = OUTPUT_DIR.rstrip("/") + "_genotype/"
    else:
        OUTPUT_DIR_RUN = OUTPUT_DIR
    os.makedirs(OUTPUT_DIR_RUN, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR_RUN, "per_slide"), exist_ok=True)

    print(f"Using slide list: {args.slide_list} ({slide_list_path})")
    print(f"Aggregate outputs will be written to: {OUTPUT_DIR_RUN}")
    print(f"Per-slide shape files are shared at: {os.path.join(OUTPUT_DIR, 'per_slide')} "
          f"(reused across both slide lists, keyed by patient ID)\n")

    if args.stage == "probe":
        probe(slide_list_path)
    elif args.stage == "time-one":
        time_one(args.contour_field, slide_list_path)
    elif args.stage == "extract":
        run_extraction(args.contour_field, slide_list_path, limit=args.limit)
    elif args.stage == "analyze":
        run_analysis(slide_list_path, OUTPUT_DIR_RUN)
    elif args.stage == "analyze-aggregated":
        run_aggregated_analysis(slide_list_path, OUTPUT_DIR_RUN)
