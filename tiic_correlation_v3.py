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
from scipy.stats import spearmanr, mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
import json
import random

from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

# ==========================================================================
# CONFIG
# ==========================================================================
EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
TILES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/tiles_v2"
SLIDES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/slides"
SPLITS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"
RESULTS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2"
OUTPUT_DIR = "/cs/student/project_msc/2025/aibh/iconstan/tiic_analysis_v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM, ATTN_DIM, FC_DIM, DROPOUT = 1536, 512, 256, 0.0
N_TILES_PER_SLIDE = 250          # original (attention-scored) tiles sampled per slide
HOVERNET_SUBCROP_SIZE = 256      # size of each HoVer-Net input crop (matches model's expected patch_input_shape)
SUBCROPS_PER_TILE = 2            # 2x2 grid -> 4 sub-crops, covering the full physical area of one original tile
THUMBNAIL_MAX_DIM = 1024
ALPHA = 0.45
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# MoNuSAC nucleus type scheme
TYPE_NAMES = {0: "Background/Other", 1: "Epithelial", 2: "Lymphocyte", 3: "Macrophage", 4: "Neutrophil"}
IMMUNE_TYPE_IDS = {2, 3, 4}  # Lymphocyte, Macrophage, Neutrophil = immune cells


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


def compute_immune_density_for_tile(slide, x, y, level_downsample, orig_tile_size, segmentor):
    """
    Extract a 2x2 grid of HoVer-Net-resolution sub-crops covering the FULL physical
    area of the original (attention-scored) tile, run HoVer-Net on each, and combine
    nucleus counts to get one immune density value representing the whole tile area.
    """
    # Physical size of the original tile in level-0 pixels
    orig_tile_size_l0 = orig_tile_size * level_downsample  # e.g. 256 * 4.0 = 1024

    # Top-left of the original tile in level-0 coordinates
    x0_l0 = x * level_downsample
    y0_l0 = y * level_downsample

    # Step size for the 2x2 sub-crop grid (each sub-crop covers half the tile's width/height)
    step = orig_tile_size_l0 / SUBCROPS_PER_TILE  # e.g. 1024 / 2 = 512

    total_nuclei = 0
    immune_nuclei = 0
    sub_crops = []

    for i in range(SUBCROPS_PER_TILE):
        for j in range(SUBCROPS_PER_TILE):
            crop_x = int(round(x0_l0 + i * step))
            crop_y = int(round(y0_l0 + j * step))
            sub_crops.append((crop_x, crop_y))

    # Extract all sub-crop images
    sub_images = []
    for (cx, cy) in sub_crops:
        img = slide.read_region((cx, cy), 0, (HOVERNET_SUBCROP_SIZE, HOVERNET_SUBCROP_SIZE)).convert("RGB")
        sub_images.append(np.array(img))
    sub_images = np.array(sub_images)

    # Run HoVer-Net on all 4 sub-crops in one batch call
    try:
        output = segmentor.run(images=sub_images, patch_mode=True, save_dir=None, output_type="dict")
    except Exception:
        return None  # signal failure

    for i in range(len(sub_images)):
        types_i = output["type"][i]
        n = len(types_i) if hasattr(types_i, "__len__") else 0
        total_nuclei += n
        if n > 0:
            immune_nuclei += sum(1 for t in types_i if t in IMMUNE_TYPE_IDS)

    if total_nuclei == 0:
        return None  # no nuclei detected in any sub-crop; exclude this tile from analysis

    return immune_nuclei / total_nuclei


def analyse_slide(patient, model, task, segmentor, true_class, mutation_label_str):
    print(f"\n  Analysing {patient} (true class: {mutation_label_str})...", flush=True)
    coords, level, tile_size, level_downsample = get_tile_info(patient)
    slide_path = get_slide_file(patient)
    if coords is None or slide_path is None:
        print(f"    Skipping: missing data")
        return None

    attn_scores, emb_coords = get_tile_attention_scores(model, patient)
    if attn_scores is None:
        print(f"    Skipping: no embeddings found")
        return None

    coord_to_attn = {(int(x), int(y)): float(a) for (x, y), a in zip(emb_coords, attn_scores)}

    all_coords = [(int(x), int(y)) for x, y in coords]
    if len(all_coords) > N_TILES_PER_SLIDE:
        sampled_coords = random.sample(all_coords, N_TILES_PER_SLIDE)
    else:
        sampled_coords = all_coords

    slide = openslide.OpenSlide(slide_path)

    tile_attn_list = []
    tile_immune_density_list = []
    tile_xy_list = []

    for idx, (x, y) in enumerate(sampled_coords):
        attn = coord_to_attn.get((x, y))
        if attn is None:
            continue

        immune_density = compute_immune_density_for_tile(
            slide, x, y, level_downsample, tile_size, segmentor
        )
        if immune_density is None:
            continue

        tile_attn_list.append(attn)
        tile_immune_density_list.append(immune_density)
        tile_xy_list.append((x, y))

        if (idx + 1) % 50 == 0:
            print(f"    Processed {idx + 1}/{len(sampled_coords)} tiles "
                  f"({len(tile_attn_list)} with detected nuclei so far)...", flush=True)

    slide.close()

    if len(tile_attn_list) < 10:
        print(f"    Skipping: too few valid tiles with detected nuclei ({len(tile_attn_list)})")
        return None

    tile_attn_arr = np.array(tile_attn_list)
    tile_immune_arr = np.array(tile_immune_density_list)

    corr, pval = spearmanr(tile_attn_arr, tile_immune_arr)
    mean_immune_density = float(tile_immune_arr.mean())
    print(f"    Valid tiles: {len(tile_attn_list)} | Spearman r={corr:.3f} (p={pval:.4f}) | "
          f"Mean immune density={mean_immune_density:.3f}")

    # Per-slide scatter plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(tile_attn_arr, tile_immune_arr, alpha=0.5, s=20)
    ax.set_xlabel("Tile attention score")
    ax.set_ylabel("Tile immune cell density")
    ax.set_title(f"{patient} ({mutation_label_str})\nSpearman r={corr:.3f} (p={pval:.4f}), n={len(tile_attn_list)} tiles")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_{task}_{patient}.png"), dpi=120)
    plt.close()

    # Immune density heatmap paired with attention heatmap
    generate_paired_heatmap(patient, slide_path, tile_xy_list, tile_attn_list,
                             tile_immune_density_list, level_downsample, tile_size, task)

    return {
        "patient": patient,
        "true_class": mutation_label_str,
        "n_tiles": len(tile_attn_list),
        "spearman_r": float(corr),
        "spearman_p": float(pval),
        "mean_attn": float(tile_attn_arr.mean()),
        "mean_immune_density": mean_immune_density,
    }


def generate_paired_heatmap(patient, slide_path, tile_xy_list, attn_list, immune_list,
                             level_downsample, tile_size, task):
    """Generate a 3-panel figure: original thumbnail, attention heatmap, immune density heatmap."""
    slide = openslide.OpenSlide(slide_path)
    level0_w, level0_h = slide.dimensions
    scale = THUMBNAIL_MAX_DIM / max(level0_w, level0_h)
    target_w, target_h = int(level0_w * scale), int(level0_h * scale)
    thumbnail = slide.get_thumbnail((target_w, target_h)).convert("RGB")
    thumb_arr = np.array(thumbnail)
    thumb_h, thumb_w = thumb_arr.shape[0], thumb_arr.shape[1]
    scale_x = thumb_w / level0_w
    scale_y = thumb_h / level0_h
    slide.close()

    def build_map(values):
        heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float32)
        count_map = np.zeros((thumb_h, thumb_w), dtype=np.float32)
        for (x, y), val in zip(tile_xy_list, values):
            x0 = x * level_downsample
            y0 = y * level_downsample
            tile_size_l0 = tile_size * level_downsample
            tx0 = int(round(x0 * scale_x))
            ty0 = int(round(y0 * scale_y))
            tx1 = int(round((x0 + tile_size_l0) * scale_x))
            ty1 = int(round((y0 + tile_size_l0) * scale_y))
            tx1, ty1 = min(tx1, thumb_w), min(ty1, thumb_h)
            if tx1 <= tx0 or ty1 <= ty0:
                continue
            heatmap[ty0:ty1, tx0:tx1] += val
            count_map[ty0:ty1, tx0:tx1] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(count_map > 0, heatmap / np.maximum(count_map, 1), np.nan)
        return avg, count_map > 0

    from scipy.stats import rankdata
    attn_pct = rankdata(attn_list, method="average") / len(attn_list)
    immune_pct = rankdata(immune_list, method="average") / len(immune_list)

    attn_map, attn_mask = build_map(attn_pct)
    immune_map, immune_mask = build_map(immune_pct)

    cmap = plt.get_cmap("jet")

    def overlay(base, val_map, mask):
        normed = np.nan_to_num(val_map, nan=0.0)
        colored = (cmap(normed)[:, :, :3] * 255).astype(np.uint8)
        result = base.copy()
        result[mask] = (0.55 * base[mask] + 0.45 * colored[mask]).astype(np.uint8)
        return result

    attn_overlay = overlay(thumb_arr, attn_map, attn_mask)
    immune_overlay = overlay(thumb_arr, immune_map, immune_mask)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(thumb_arr)
    axes[0].set_title(f"{patient}\nOriginal", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(attn_overlay)
    axes[1].set_title("ABMIL Attention", fontsize=10)
    axes[1].axis("off")
    axes[2].imshow(immune_overlay)
    axes[2].set_title("Immune Cell Density\n(Lymphocyte+Macrophage+Neutrophil)", fontsize=10)
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"paired_heatmap_{task}_{patient}.png"), dpi=130, bbox_inches="tight")
    plt.close()


def run_tiic_analysis(task, example_patients_with_labels, segmentor):
    print(f"\n{'='*60}\nTIIC correlation analysis: {task}\n{'='*60}")
    model = load_abmil_model(task)

    results = []
    for patient, true_class, label_str in example_patients_with_labels:
        res = analyse_slide(patient, model, task, segmentor, true_class, label_str)
        if res is not None:
            results.append(res)

    if not results:
        print("No valid results for this task.")
        return None

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, f"tiic_correlation_{task}.csv"), index=False)

    # === Multiple testing correction (Benjamini-Hochberg) ===
    reject, pvals_corrected, _, _ = multipletests(results_df["spearman_p"].values, alpha=0.05, method="fdr_bh")
    results_df["spearman_p_bh_corrected"] = pvals_corrected
    results_df["significant_after_correction"] = reject
    results_df.to_csv(os.path.join(OUTPUT_DIR, f"tiic_correlation_{task}.csv"), index=False)

    print(f"\n=== SUMMARY ({task}) ===")
    print(f"Slides analysed: {len(results_df)}")
    print(f"Mean Spearman r: {results_df['spearman_r'].mean():.3f}")
    print(f"Median Spearman r: {results_df['spearman_r'].median():.3f}")
    print(f"Significant BEFORE correction (p<0.05): {(results_df['spearman_p'] < 0.05).sum()} / {len(results_df)}")
    print(f"Significant AFTER Benjamini-Hochberg correction: {results_df['significant_after_correction'].sum()} / {len(results_df)}")

    # Boxplot of correlation coefficients
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(results_df["spearman_r"], vert=True)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Spearman r (attention vs immune density)")
    ax.set_title(f"TIIC-Attention Correlation Distribution\n({task}, n={len(results_df)} slides)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"correlation_summary_{task}.png"), dpi=150)
    plt.close()

    # === Slide-level aggregate immune density comparison across classes ===
    fig, ax = plt.subplots(figsize=(7, 5))
    classes = results_df["true_class"].unique()
    data_by_class = [results_df[results_df["true_class"] == c]["mean_immune_density"].values for c in classes]
    ax.boxplot(data_by_class, tick_labels=classes)
    ax.set_ylabel("Mean immune density per slide")
    ax.set_title(f"Slide-level Immune Density by Class ({task})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"immune_density_by_class_{task}.png"), dpi=150)
    plt.close()

    # Statistical test across classes
    if len(classes) == 2:
        stat, p = mannwhitneyu(*data_by_class)
        print(f"Mann-Whitney U test (immune density between {classes[0]} vs {classes[1]}): p={p:.4f}")
    elif len(classes) > 2:
        stat, p = kruskal(*data_by_class)
        print(f"Kruskal-Wallis test (immune density across {list(classes)}): p={p:.4f}")

    print(f"Saved results and plots to {OUTPUT_DIR}")
    return results_df


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

    results_multiclass = run_tiic_analysis("multiclass", multiclass_examples, segmentor)
    results_ret = run_tiic_analysis("ret_binary", ret_examples, segmentor)

    print("\nAll TIIC correlation analyses complete!")
