"""Attention heatmaps for correctly and incorrectly classified test slides (dissertation §3.11).

For each task and class, two correctly and one incorrectly classified test slides are
chosen, and the ABMIL attention of every tile is drawn over the slide thumbnail
(percentile-rank normalised, jet colour map, alpha 0.45). RET predictions use the
validation-selected threshold.

Usage:
    python analyses/interpretability/generate_heatmaps.py
"""

import os

import h5py
import matplotlib
import numpy as np
import openslide
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from ptc_mil.analysis import load_context

ctx = load_context()
DEVICE = ctx.device
OUTPUT_DIR = str(ctx.output_dir("heatmaps"))
SPLITS_PATH = ctx.paths.splits_csv

THUMBNAIL_MAX_DIM = 1024
ALPHA = 0.45
RET_OPTIMAL_THRESHOLD = ctx.config.thresholds.abmil  # validation-selected RET threshold (0.13)


def get_embedding_file(patient):
    path = ctx.embedding_file(patient)
    return str(path) if path is not None else None


def get_slide_file(patient):
    path = ctx.slide_file(patient)
    return str(path) if path is not None else None


def get_tile_info(patient):
    path = ctx.tile_file(patient)
    if path is None:
        return None, None, None, None
    with h5py.File(path, "r") as h5:
        coords = h5["coords"][:]
        level = int(h5.attrs["level"])
        tile_size = int(h5.attrs["tile_size"])
        level_downsample = float(h5.attrs["level_downsample"])
    return coords, level, tile_size, level_downsample


def load_model(task):
    model, _ = ctx.load_abmil(task)
    return model


def select_examples(splits_df, task, n_correct=2, n_incorrect=1):
    test_df = splits_df[splits_df["split"] == "test"].copy()

    if task == "multiclass":
        label_map = {"BRAF_V600E": 0, "RAS": 1, "Other": 2}
        num_classes = 3
        test_df = test_df[test_df["mutation_label"].isin(label_map)]
        test_df["label"] = test_df["mutation_label"].map(label_map)
        class_names = ["BRAF_V600E", "RAS", "Other"]
    else:
        num_classes = 2
        test_df["label"] = test_df["RET"]
        class_names = ["RET_negative", "RET_positive"]

    model = load_model(task)

    selected = []
    for cls_idx, cls_name in enumerate(class_names):
        candidates = test_df[test_df["label"] == cls_idx]["patient"].tolist()
        correct_found, incorrect_found = 0, 0
        for patient in candidates:
            emb_file = get_embedding_file(patient)
            if emb_file is None:
                continue
            with h5py.File(emb_file, "r") as f:
                features = torch.tensor(f["features"][:], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                logits, A = model(features)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            if task == "ret_binary":
                pred = int(probs[1] >= RET_OPTIMAL_THRESHOLD)
            else:
                pred = int(np.argmax(probs))

            is_correct = pred == cls_idx
            if is_correct and correct_found < n_correct:
                selected.append(
                    {
                        "patient": patient,
                        "true_class": cls_name,
                        "true_idx": cls_idx,
                        "status": "correct",
                    }
                )
                correct_found += 1
            elif (not is_correct) and incorrect_found < n_incorrect:
                pred_class_name = class_names[pred]
                selected.append(
                    {
                        "patient": patient,
                        "true_class": cls_name,
                        "true_idx": cls_idx,
                        "status": f"misclassified_as_{pred_class_name}",
                    }
                )
                incorrect_found += 1
            if correct_found >= n_correct and incorrect_found >= n_incorrect:
                break
        print(
            f"  {task} / {cls_name}: {correct_found} correct, {incorrect_found} misclassified examples found"
        )

    return selected, model, num_classes, class_names


def generate_heatmap(patient, model, task, true_class, status):
    emb_file = get_embedding_file(patient)
    coords, level, tile_size, level_downsample = get_tile_info(patient)
    slide_path = get_slide_file(patient)

    if emb_file is None or coords is None or slide_path is None:
        print(f"  Skipping {patient}: missing data")
        return

    with h5py.File(emb_file, "r") as f:
        features = torch.tensor(f["features"][:], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits, A = model(features)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    if task == "ret_binary":
        pred_idx = int(probs[1] >= RET_OPTIMAL_THRESHOLD)
    else:
        pred_idx = int(np.argmax(probs))

    attn_scores = A.squeeze().cpu().numpy()
    percentile_scores = rankdata(attn_scores, method="average") / len(attn_scores)

    slide = openslide.OpenSlide(slide_path)
    level0_w, level0_h = slide.dimensions
    scale = THUMBNAIL_MAX_DIM / max(level0_w, level0_h)
    target_w, target_h = int(level0_w * scale), int(level0_h * scale)
    thumbnail = slide.get_thumbnail((target_w, target_h)).convert("RGB")
    thumb_arr = np.array(thumbnail)
    thumb_h, thumb_w = thumb_arr.shape[0], thumb_arr.shape[1]
    scale_x = thumb_w / level0_w
    scale_y = thumb_h / level0_h

    heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    count_map = np.zeros((thumb_h, thumb_w), dtype=np.float32)

    for (x, y), score in zip(coords, percentile_scores):
        x0 = x * level_downsample
        y0 = y * level_downsample
        tile_size_l0 = tile_size * level_downsample

        tx0 = int(round(x0 * scale_x))
        ty0 = int(round(y0 * scale_y))
        tx1 = int(round((x0 + tile_size_l0) * scale_x))
        ty1 = int(round((y0 + tile_size_l0) * scale_y))

        tx1 = min(tx1, thumb_w)
        ty1 = min(ty1, thumb_h)
        if tx1 <= tx0 or ty1 <= ty0:
            continue

        heatmap[ty0:ty1, tx0:tx1] += score
        count_map[ty0:ty1, tx0:tx1] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        heatmap_avg = np.where(count_map > 0, heatmap / np.maximum(count_map, 1), np.nan)

    cmap = plt.get_cmap("jet")
    normed = np.nan_to_num(heatmap_avg, nan=0.0)
    colored = (cmap(normed)[:, :, :3] * 255).astype(np.uint8)

    mask = count_map > 0
    overlay = thumb_arr.copy()
    overlay[mask] = ((1 - ALPHA) * thumb_arr[mask] + ALPHA * colored[mask]).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(thumb_arr)
    axes[0].set_title(f"{patient}\nTrue: {true_class} ({status})", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    pred_class_str = f"Pred idx: {pred_idx}, Probs: {np.round(probs, 3)}"
    axes[1].set_title(f"Attention Heatmap (final model)\n{pred_class_str}", fontsize=9)
    axes[1].axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Attention percentile", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{task}_{true_class}_{status}_{patient}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    slide.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    splits_df = pd.read_csv(SPLITS_PATH)

    print("=== Selecting examples: Aim a) Multiclass ===")
    examples_a, model_a, num_classes_a, class_names_a = select_examples(
        splits_df, "multiclass", n_correct=2, n_incorrect=1
    )
    print("\n=== Generating heatmaps: Aim a) Multiclass ===")
    for ex in examples_a:
        generate_heatmap(ex["patient"], model_a, "multiclass", ex["true_class"], ex["status"])

    print("\n=== Selecting examples: Aim b) RET binary (validation-selected threshold) ===")
    examples_b, model_b, num_classes_b, class_names_b = select_examples(
        splits_df, "ret_binary", n_correct=3, n_incorrect=2
    )
    print("\n=== Generating heatmaps: Aim b) RET binary ===")
    for ex in examples_b:
        generate_heatmap(ex["patient"], model_b, "ret_binary", ex["true_class"], ex["status"])

    print(f"\nAll heatmaps saved to {OUTPUT_DIR}")
