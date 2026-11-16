"""Sanity checks for the TIIC analysis on one test slide (run before the full analysis).

1. Tile coordinates and embedding coordinates match exactly, and attention sums to 1.
2. High-attention tiles do not simply contain fewer nuclei than low-attention tiles
   (which would make a negative attention-immune correlation a low-count artefact).
3. Example high- and low-attention tiles are saved for visual inspection.

Uses the multiclass model. Nuclei are counted on a 2x2 grid of 256x256 level-0 crops
per tile, as in the original check (the final TIIC analysis uses a full 4x4 grid).

Usage:
    python analyses/diagnostics/tiic_sanity_check.py
"""

import os
import random

import h5py
import matplotlib
import numpy as np
import openslide
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

from ptc_mil.analysis import load_context

ctx = load_context()
OUTPUT_DIR = str(ctx.output_dir("tiic_sanity_check"))
DEVICE = ctx.device
TEST_PATIENT = "TCGA-EL-A3T8"

random.seed(42)

model, _ = ctx.load_abmil("multiclass")

# Load tiles and embeddings
tile_file = ctx.tile_file(TEST_PATIENT)
with h5py.File(tile_file, "r") as h5:
    tile_coords = h5["coords"][:]
    level = int(h5.attrs["level"])
    tile_size = int(h5.attrs["tile_size"])
    level_downsample = float(h5.attrs["level_downsample"])

emb_file = ctx.embedding_file(TEST_PATIENT)
with h5py.File(emb_file, "r") as f:
    features = torch.tensor(f["features"][:], dtype=torch.float32).to(DEVICE)
    emb_coords = f["coords"][:]

print("\n=== CHECK 1: Coordinate matching integrity ===")
print(f"Tile coords count (tiles): {len(tile_coords)}")
print(f"Embedding coords count (embeddings): {len(emb_coords)}")
tile_set = set((int(x), int(y)) for x, y in tile_coords)
emb_set = set((int(x), int(y)) for x, y in emb_coords)
print(f"Coords in tiles but not in embeddings: {len(tile_set - emb_set)}")
print(f"Coords in embeddings but not in tiles: {len(emb_set - tile_set)}")
print(f"Coords matching exactly: {len(tile_set & emb_set)} / {len(tile_set)}")

with torch.no_grad():
    _, A = model(features)
attn_scores = A.squeeze().cpu().numpy()
print(f"\nAttention scores sum to: {attn_scores.sum():.6f} (should be ~1.0, softmax property)")
print(
    f"Attention scores count: {len(attn_scores)} (should match embeddings count: {len(emb_coords)})"
)

coord_to_attn = {(int(x), int(y)): float(a) for (x, y), a in zip(emb_coords, attn_scores)}

print("\n=== CHECK 2: Total nuclei count vs attention score (confound check) ===")
print("Sampling 40 tiles across the attention range to check for a low-n artifact...")

slide = openslide.OpenSlide(str(ctx.slide_file(TEST_PATIENT)))

segmentor = MultiTaskSegmentor(
    model="hovernet_fast-monusac",
    batch_size=4,
    num_workers=1,
    device="cuda" if DEVICE.type == "cuda" else "cpu",
    verbose=False,
)

# Sample tiles spanning the attention range: top 20 highest-attention + 20 random
sorted_by_attn = sorted(coord_to_attn.items(), key=lambda kv: kv[1], reverse=True)
high_attn_tiles = sorted_by_attn[:20]
low_attn_tiles = random.sample(sorted_by_attn[500:], 20)  # avoid the very top, sample broadly

check_tiles = high_attn_tiles + low_attn_tiles
results = []

IMMUNE_TYPE_IDS = {2, 3, 4}
saved_examples = {"high": [], "low": []}

for (x, y), attn in check_tiles:
    orig_tile_size_l0 = tile_size * level_downsample
    x0_l0 = x * level_downsample
    y0_l0 = y * level_downsample
    step = orig_tile_size_l0 / 2
    sub_crops = [
        (int(round(x0_l0 + i * step)), int(round(y0_l0 + j * step)))
        for i in range(2)
        for j in range(2)
    ]
    sub_images = np.array(
        [
            np.array(slide.read_region((cx, cy), 0, (256, 256)).convert("RGB"))
            for (cx, cy) in sub_crops
        ]
    )
    try:
        output = segmentor.run(
            images=sub_images, patch_mode=True, save_dir=None, output_type="dict"
        )
    except Exception:
        continue

    total_nuclei, immune_nuclei = 0, 0
    for i in range(len(sub_images)):
        types_i = output["type"][i]
        n = len(types_i) if hasattr(types_i, "__len__") else 0
        total_nuclei += n
        if n > 0:
            immune_nuclei += sum(1 for t in types_i if t in IMMUNE_TYPE_IDS)

    category = "high" if (x, y) in dict(high_attn_tiles) else "low"
    results.append(
        {
            "attn": attn,
            "total_nuclei": total_nuclei,
            "immune_nuclei": immune_nuclei,
            "category": category,
        }
    )

    # Save one example image from each category for visual inspection
    if len(saved_examples[category]) < 3 and total_nuclei > 0:
        saved_examples[category].append((x, y, sub_images[0], attn, total_nuclei, immune_nuclei))

slide.close()

high_totals = [r["total_nuclei"] for r in results if r["category"] == "high"]
low_totals = [r["total_nuclei"] for r in results if r["category"] == "low"]
print(
    f"\nHigh-attention tiles (top 20): mean total nuclei = {np.mean(high_totals):.1f} (range {min(high_totals)}-{max(high_totals)})"
)
print(
    f"Low/random-attention tiles (20): mean total nuclei = {np.mean(low_totals):.1f} (range {min(low_totals)}-{max(low_totals)})"
)
print("(If high-attention tiles have SIMILAR or MORE total nuclei than low-attention tiles,")
print(" the negative correlation is NOT explained by a low-sample-size artifact.)")

print("\n=== CHECK 3: Saving example tile images for visual inspection ===")
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for row, category in enumerate(["high", "low"]):
    for col, (x, y, img, attn, total, immune) in enumerate(saved_examples[category][:3]):
        axes[row, col].imshow(img)
        axes[row, col].set_title(
            f"{category}-attn: {attn:.4f}\ntotal={total}, immune={immune}", fontsize=9
        )
        axes[row, col].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sanity_check_example_tiles.png"), dpi=120)
print("Saved example tile comparison image")

print("\n=== SANITY CHECK COMPLETE ===")
