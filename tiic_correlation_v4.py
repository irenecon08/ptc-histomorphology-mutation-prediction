import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
import openslide
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata, mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
import json
import random
import time

from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

# ==========================================================================
# CONFIG
# ==========================================================================
EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
TILES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/tiles_v2"
SLIDES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/slides"
SPLITS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"
RESULTS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2"
OUTPUT_DIR = "/cs/student/project_msc/2025/aibh/iconstan/tiic_analysis_v4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM, ATTN_DIM, FC_DIM, DROPOUT = 1536, 512, 256, 0.0
N_TILES_PER_SLIDE = 250
HOVERNET_SUBCROP_SIZE = 256
SUBCROPS_PER_TILE = 2
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

TYPE_NAMES = {0: "Background/Other", 1: "Epithelial", 2: "Lymphocyte", 3: "Macrophage", 4: "Neutrophil"}
IMMUNE_TYPE_IDS = {2, 3, 4}
EPITHELIAL_TYPE_ID = 1


# ==========================================================================
# ABMIL model
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
    if task == "multiclass":
        num_classes = 3
        model_path = os.path.join(RESULTS_DIR, "best_multiclass_v2.pt")
    else:
        num_classes = 2
        model_path = os.path.join(RESULTS_DIR, "best_ret_binary_v2.pt")
    model = ABMIL(EMBED_DIM, ATTN_DIM, FC_DIM, num_classes, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


# ==========================================================================
# Helper functions
# ==========================================================================
def get_embedding_file(patient):
    for f in os.listdir(EMBEDDINGS_DIR):
        if f.startswith(patient) and f.endswith(".h5"):
            return os.path.join(EMBEDDINGS_DIR, f)
    return None


def get_slide_file(patient):
    for root, dirs, files in os.walk(SLIDES_DIR):
        for f in files:
            if f.startswith(patient) and f.endswith(".svs"):
                return os.path.join(root, f)
    return None


def get_tile_info(patient):
    for f in os.listdir(TILES_DIR):
        if f.startswith(patient) and f.endswith(".h5"):
            with h5py.File(os.path.join(TILES_DIR, f), "r") as h5:
                coords = h5["coords"][:]
                level = int(h5.attrs["level"])
                tile_size = int(h5.attrs["tile_size"])
                level_downsample = float(h5.attrs["level_downsample"])
            return coords, level, tile_size, level_downsample
    return None, None, None, None


def get_tile_attention_scores(model, patient):
    emb_file = get_embedding_file(patient)
    if emb_file is None:
        return None, None
    with h5py.File(emb_file, "r") as f:
        features = torch.tensor(f["features"][:], dtype=torch.float32).to(DEVICE)
        emb_coords = f["coords"][:]
    with torch.no_grad():
        _, A = model(features)
    attn_scores = A.squeeze().cpu().numpy()
    return attn_scores, emb_coords


def compute_nuclei_counts_for_tile(slide, x, y, level_downsample, orig_tile_size, segmentor):
    """
    Extract a 2x2 grid of HoVer-Net-resolution sub-crops covering the FULL physical
    area of the original tile, run HoVer-Net, and return RAW counts (not just density)
    so downstream analysis can compute density, partial correlations, etc.
    """
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


def partial_spearman(x, y, z):
    """
    Partial Spearman correlation between x and y, controlling for z.
    Implemented via rank-transform + OLS residuals (standard approach).
    """
    x_r = rankdata(x)
    y_r = rankdata(y)
    z_r = rankdata(z)

    # Regress out z from x and y (simple linear regression on ranks)
    def residualize(target, control):
        control_with_const = np.column_stack([np.ones(len(control)), control])
        coefs, _, _, _ = np.linalg.lstsq(control_with_const, target, rcond=None)
        predicted = control_with_const @ coefs
        return target - predicted

    x_resid = residualize(x_r, z_r)
    y_resid = residualize(y_r, z_r)

    r, p = spearmanr(x_resid, y_resid)
    return r, p


def analyse_slide(patient, model, task, segmentor, true_class, mutation_label_str):
    print(f"\n  Analysing {patient} (true class: {mutation_label_str})...", flush=True)
    coords, level, tile_size, level_downsample = get_tile_info(patient)
    slide_path = get_slide_file(patient)
    if coords is None or slide_path is None:
        print(f"    Skipping: missing data")
        return None, None

    attn_scores, emb_coords = get_tile_attention_scores(model, patient)
    if attn_scores is None:
        print(f"    Skipping: no embeddings found")
        return None, None

    coord_to_attn = {(int(x), int(y)): float(a) for (x, y), a in zip(emb_coords, attn_scores)}

    all_coords = [(int(x), int(y)) for x, y in coords]
    if len(all_coords) > N_TILES_PER_SLIDE:
        sampled_coords = random.sample(all_coords, N_TILES_PER_SLIDE)
    else:
        sampled_coords = all_coords

    slide = openslide.OpenSlide(slide_path)

    per_tile_records = []
    start_time = time.time()

    for idx, (x, y) in enumerate(sampled_coords):
        attn = coord_to_attn.get((x, y))
        if attn is None:
            continue

        counts = compute_nuclei_counts_for_tile(slide, x, y, level_downsample, tile_size, segmentor)
        if counts is None:
            continue

        total = counts["total_nuclei"]
        immune_density = counts["immune_nuclei"] / total
        epithelial_density = counts["epithelial_nuclei"] / total

        per_tile_records.append({
            "patient": patient, "true_class": mutation_label_str, "x": x, "y": y,
            "attn": attn, "total_nuclei": total,
            "immune_nuclei": counts["immune_nuclei"], "epithelial_nuclei": counts["epithelial_nuclei"],
            "immune_density": immune_density, "epithelial_density": epithelial_density,
        })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"    Processed {idx + 1}/{len(sampled_coords)} tiles "
                  f"({len(per_tile_records)} valid) | elapsed={elapsed/60:.1f}min", flush=True)

    slide.close()

    if len(per_tile_records) < 10:
        print(f"    Skipping: too few valid tiles ({len(per_tile_records)})")
        return None, None

    tile_df = pd.DataFrame(per_tile_records)

    attn_arr = tile_df["attn"].values
    immune_arr = tile_df["immune_density"].values
    epithelial_arr = tile_df["epithelial_density"].values
    total_arr = tile_df["total_nuclei"].values

    # Standard (unadjusted) correlation, same as v3
    corr, pval = spearmanr(attn_arr, immune_arr)

    # NEW: correlation between attention and RAW immune count (not density)
    corr_count, pval_count = spearmanr(attn_arr, tile_df["immune_nuclei"].values)

    # NEW: correlation between attention and epithelial density
    corr_epi, pval_epi = spearmanr(attn_arr, epithelial_arr)

    # NEW: partial correlation - attention vs immune density, controlling for epithelial density
    corr_partial, pval_partial = partial_spearman(attn_arr, immune_arr, epithelial_arr)

    print(f"    n={len(tile_df)} | "
          f"r(attn,immune_density)={corr:.3f} (p={pval:.4f}) | "
          f"r(attn,immune_COUNT)={corr_count:.3f} (p={pval_count:.4f}) | "
          f"r(attn,epithelial_density)={corr_epi:.3f} (p={pval_epi:.4f}) | "
          f"PARTIAL r(attn,immune|epithelial)={corr_partial:.3f} (p={pval_partial:.4f})")

    summary = {
        "patient": patient, "true_class": mutation_label_str, "n_tiles": len(tile_df),
        "spearman_r_immune_density": float(corr), "spearman_p_immune_density": float(pval),
        "spearman_r_immune_count": float(corr_count), "spearman_p_immune_count": float(pval_count),
        "spearman_r_epithelial_density": float(corr_epi), "spearman_p_epithelial_density": float(pval_epi),
        "partial_r_immune_given_epithelial": float(corr_partial), "partial_p_immune_given_epithelial": float(pval_partial),
        "mean_attn": float(attn_arr.mean()), "mean_immune_density": float(immune_arr.mean()),
        "mean_epithelial_density": float(epithelial_arr.mean()), "mean_total_nuclei": float(total_arr.mean()),
    }

    return summary, tile_df


def run_tiic_analysis(task, example_patients_with_labels, segmentor):
    print(f"\n{'='*60}\nTIIC correlation analysis v4 (with partial correlation): {task}\n{'='*60}")
    model = load_abmil_model(task)

    summaries = []
    all_tile_dfs = []
    total_patients = len(example_patients_with_labels)
    overall_start = time.time()

    for i, (patient, true_class, label_str) in enumerate(example_patients_with_labels):
        print(f"\n[Slide {i+1}/{total_patients}] elapsed so far: {(time.time()-overall_start)/60:.1f} min")
        summary, tile_df = analyse_slide(patient, model, task, segmentor, true_class, label_str)
        if summary is not None:
            summaries.append(summary)
            all_tile_dfs.append(tile_df)

    if not summaries:
        print("No valid results for this task.")
        return None

    summary_df = pd.DataFrame(summaries)
    reject, pvals_corrected, _, _ = multipletests(
        summary_df["spearman_p_immune_density"].values, alpha=0.05, method="fdr_bh"
    )
    summary_df["spearman_p_bh_corrected"] = pvals_corrected
    summary_df["significant_after_correction"] = reject
    summary_df.to_csv(os.path.join(OUTPUT_DIR, f"tiic_summary_{task}.csv"), index=False)

    # Save ALL per-tile raw data (pooled across slides) for pooled analysis
    pooled_tiles_df = pd.concat(all_tile_dfs, ignore_index=True)
    pooled_tiles_df.to_csv(os.path.join(OUTPUT_DIR, f"tiic_per_tile_{task}.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"=== SUMMARY ({task}) ===")
    print(f"Slides analysed: {len(summary_df)}")
    print(f"Mean r (attn vs immune density): {summary_df['spearman_r_immune_density'].mean():.3f}")
    print(f"Mean r (attn vs immune COUNT):   {summary_df['spearman_r_immune_count'].mean():.3f}")
    print(f"Mean r (attn vs epithelial density): {summary_df['spearman_r_epithelial_density'].mean():.3f}")
    print(f"Mean PARTIAL r (attn vs immune | epithelial): {summary_df['partial_r_immune_given_epithelial'].mean():.3f}")
    print(f"Slides where partial correlation still significant (p<0.05): "
          f"{(summary_df['partial_p_immune_given_epithelial'] < 0.05).sum()} / {len(summary_df)}")

    # === POOLED analysis across all tiles from all slides (much higher power) ===
    print(f"\n--- POOLED analysis across all {len(pooled_tiles_df)} tiles from {len(summary_df)} slides ---")
    pooled_r, pooled_p = spearmanr(pooled_tiles_df["attn"], pooled_tiles_df["immune_density"])
    pooled_r_count, pooled_p_count = spearmanr(pooled_tiles_df["attn"], pooled_tiles_df["immune_nuclei"])
    pooled_r_epi, pooled_p_epi = spearmanr(pooled_tiles_df["attn"], pooled_tiles_df["epithelial_density"])
    pooled_partial_r, pooled_partial_p = partial_spearman(
        pooled_tiles_df["attn"].values, pooled_tiles_df["immune_density"].values, pooled_tiles_df["epithelial_density"].values
    )
    print(f"Pooled r(attn, immune_density) = {pooled_r:.3f} (p={pooled_p:.2e})")
    print(f"Pooled r(attn, immune_COUNT)   = {pooled_r_count:.3f} (p={pooled_p_count:.2e})")
    print(f"Pooled r(attn, epithelial_density) = {pooled_r_epi:.3f} (p={pooled_p_epi:.2e})")
    print(f"Pooled PARTIAL r(attn, immune | epithelial) = {pooled_partial_r:.3f} (p={pooled_partial_p:.2e})")

    pooled_results = {
        "task": task, "n_tiles_pooled": len(pooled_tiles_df), "n_slides": len(summary_df),
        "pooled_r_immune_density": float(pooled_r), "pooled_p_immune_density": float(pooled_p),
        "pooled_r_immune_count": float(pooled_r_count), "pooled_p_immune_count": float(pooled_p_count),
        "pooled_r_epithelial_density": float(pooled_r_epi), "pooled_p_epithelial_density": float(pooled_p_epi),
        "pooled_partial_r": float(pooled_partial_r), "pooled_partial_p": float(pooled_partial_p),
    }
    with open(os.path.join(OUTPUT_DIR, f"pooled_results_{task}.json"), "w") as f:
        json.dump(pooled_results, f, indent=2)

    # Plot: partial vs unadjusted correlation per slide
    fig, ax = plt.subplots(figsize=(7, 5))
    x_pos = np.arange(len(summary_df))
    width = 0.35
    ax.bar(x_pos - width/2, summary_df["spearman_r_immune_density"], width, label="Unadjusted r")
    ax.bar(x_pos + width/2, summary_df["partial_r_immune_given_epithelial"], width, label="Partial r (controlling for epithelial density)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary_df["patient"], rotation=90, fontsize=7)
    ax.set_ylabel("Spearman r")
    ax.set_title(f"Unadjusted vs Partial Correlation per Slide ({task})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"partial_vs_unadjusted_{task}.png"), dpi=150)
    plt.close()

    print(f"\nSaved results and plots to {OUTPUT_DIR}")
    return summary_df, pooled_tiles_df


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Loading HoVer-Net (MoNuSAC)...")
    segmentor = MultiTaskSegmentor(
        model="hovernet_fast-monusac", batch_size=4, num_workers=1,
        device="cuda" if DEVICE == "cuda" else "cpu", verbose=False,
    )
    print("  Loaded.")

    splits_df = pd.read_csv(SPLITS_PATH)
    test_df = splits_df[splits_df["split"] == "test"]

    multiclass_examples = []
    for label in ["BRAF_V600E", "RAS", "Other"]:
        candidates = test_df[test_df["mutation_label"] == label]["patient"].tolist()
        for p in candidates[:3]:
            label_map = {"BRAF_V600E": 0, "RAS": 1, "Other": 2}
            multiclass_examples.append((p, label_map[label], label))

    ret_examples = []
    for ret_status, n in [(0, 3), (1, 5)]:
        candidates = test_df[test_df["RET"] == ret_status]["patient"].tolist()
        label_str = "RET_positive" if ret_status == 1 else "RET_negative"
        for p in candidates[:n]:
            ret_examples.append((p, ret_status, label_str))

    run_tiic_analysis("multiclass", multiclass_examples, segmentor)
    run_tiic_analysis("ret_binary", ret_examples, segmentor)

    print("\nAll TIIC v4 analyses (with partial correlation) complete!")
