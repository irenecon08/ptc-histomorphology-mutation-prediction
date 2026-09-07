import os
import numpy as np
import h5py
import openslide
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
TILES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/tiles_v2"
SLIDES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/slides"
RESULTS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2"
OUTPUT_DIR = "/cs/student/project_msc/2025/aibh/iconstan/tiic_test_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_PATIENT = "TCGA-EL-A3T8"

random.seed(42)


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


print("Loading ABMIL...")
model = ABMIL(1536, 512, 256, 3, 0.0).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_multiclass_v2.pt"), map_location=DEVICE))
model.eval()

# Load tiles and embeddings
tile_file = [os.path.join(TILES_DIR, f) for f in os.listdir(TILES_DIR) if f.startswith(TEST_PATIENT)][0]
with h5py.File(tile_file, "r") as h5:
    tile_coords = h5["coords"][:]
    level = int(h5.attrs["level"])
    tile_size = int(h5.attrs["tile_size"])
    level_downsample = float(h5.attrs["level_downsample"])

emb_file = [os.path.join(EMBEDDINGS_DIR, f) for f in os.listdir(EMBEDDINGS_DIR) if f.startswith(TEST_PATIENT)][0]
with h5py.File(emb_file, "r") as f:
    features = torch.tensor(f["features"][:], dtype=torch.float32).to(DEVICE)
    emb_coords = f["coords"][:]

print(f"\n=== CHECK 1: Coordinate matching integrity ===")
print(f"Tile coords count (tiles_v2): {len(tile_coords)}")
print(f"Embedding coords count (embeddings_v2): {len(emb_coords)}")
tile_set = set((int(x), int(y)) for x, y in tile_coords)
emb_set = set((int(x), int(y)) for x, y in emb_coords)
print(f"Coords in tiles but not in embeddings: {len(tile_set - emb_set)}")
print(f"Coords in embeddings but not in tiles: {len(emb_set - tile_set)}")
print(f"Coords matching exactly: {len(tile_set & emb_set)} / {len(tile_set)}")

with torch.no_grad():
    _, A = model(features)
attn_scores = A.squeeze().cpu().numpy()
print(f"\nAttention scores sum to: {attn_scores.sum():.6f} (should be ~1.0, softmax property)")
print(f"Attention scores count: {len(attn_scores)} (should match embeddings count: {len(emb_coords)})")

coord_to_attn = {(int(x), int(y)): float(a) for (x, y), a in zip(emb_coords, attn_scores)}

print(f"\n=== CHECK 2: Total nuclei count vs attention score (confound check) ===")
print("Sampling 40 tiles across the attention range to check for a low-n artifact...")

slide_path = [os.path.join(root, f) for root, dirs, files in os.walk(SLIDES_DIR)
              for f in files if f.startswith(TEST_PATIENT) and f.endswith(".svs")][0]
slide = openslide.OpenSlide(slide_path)

segmentor = MultiTaskSegmentor(
    model="hovernet_fast-monusac", batch_size=4, num_workers=1,
    device="cuda" if DEVICE == "cuda" else "cpu", verbose=False,
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
    sub_crops = [(int(round(x0_l0 + i * step)), int(round(y0_l0 + j * step))) for i in range(2) for j in range(2)]
    sub_images = np.array([
        np.array(slide.read_region((cx, cy), 0, (256, 256)).convert("RGB")) for (cx, cy) in sub_crops
    ])
    try:
        output = segmentor.run(images=sub_images, patch_mode=True, save_dir=None, output_type="dict")
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
    results.append({"attn": attn, "total_nuclei": total_nuclei, "immune_nuclei": immune_nuclei, "category": category})

    # Save one example image from each category for visual inspection
    if len(saved_examples[category]) < 3 and total_nuclei > 0:
        saved_examples[category].append((x, y, sub_images[0], attn, total_nuclei, immune_nuclei))

slide.close()

high_totals = [r["total_nuclei"] for r in results if r["category"] == "high"]
low_totals = [r["total_nuclei"] for r in results if r["category"] == "low"]
print(f"\nHigh-attention tiles (top 20): mean total nuclei = {np.mean(high_totals):.1f} (range {min(high_totals)}-{max(high_totals)})")
print(f"Low/random-attention tiles (20): mean total nuclei = {np.mean(low_totals):.1f} (range {min(low_totals)}-{max(low_totals)})")
print("(If high-attention tiles have SIMILAR or MORE total nuclei than low-attention tiles,")
print(" the negative correlation is NOT explained by a low-sample-size artifact.)")

print(f"\n=== CHECK 3: Saving example tile images for visual inspection ===")
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for row, category in enumerate(["high", "low"]):
    for col, (x, y, img, attn, total, immune) in enumerate(saved_examples[category][:3]):
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{category}-attn: {attn:.4f}\ntotal={total}, immune={immune}", fontsize=9)
        axes[row, col].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sanity_check_example_tiles.png"), dpi=120)
print(f"Saved example tile comparison image")

print("\n=== SANITY CHECK COMPLETE ===")
