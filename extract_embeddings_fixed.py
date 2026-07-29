import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
import openslide
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

# === PATHS ===
TILES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/tiles_v2"          # corrected tiles
SLIDES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/slides"
EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"  # NEW folder
HF_TOKEN_PATH = "/cs/student/project_msc/2025/aibh/iconstan/hf_token.txt"

BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

with open(HF_TOKEN_PATH) as f:
    token = f.read().strip()
login(token=token)

timm_kwargs = {
    'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
    'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667 * 2,
    'num_classes': 0, 'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU,
    'reg_tokens': 8, 'dynamic_img_size': True
}

print("Loading UNI2-h model...")
model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
model = model.to(DEVICE)
model.eval()
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
print("Model loaded.\n")

slide_map = {}
for root, dirs, files in os.walk(SLIDES_DIR):
    for f in files:
        if f.endswith(".svs"):
            slide_map[f.replace(".svs", "")] = os.path.join(root, f)

h5_files = [f for f in os.listdir(TILES_DIR) if f.endswith(".h5")]
print(f"Extracting embeddings for {len(h5_files)} slides (corrected coordinates)...\n")

for h5_file in tqdm(h5_files):
    slide_name = h5_file.replace(".h5", "")
    output_path = os.path.join(EMBEDDINGS_DIR, f"{slide_name}.h5")

    if os.path.exists(output_path):
        continue
    if slide_name not in slide_map:
        print(f"  WARNING: No SVS found for {slide_name}")
        continue

    try:
        slide = openslide.OpenSlide(slide_map[slide_name])

        with h5py.File(os.path.join(TILES_DIR, h5_file), "r") as f:
            coords = f["coords"][:]
            level = int(f.attrs["level"])
            tile_size = int(f.attrs["tile_size"])
            # *** THE FIX: read the ACTUAL downsample stored during re-tiling ***
            level_downsample = float(f.attrs["level_downsample"])

        if len(coords) == 0:
            print(f"  WARNING: {slide_name} has 0 tiles, skipping")
            continue

        all_embeddings = []
        batch_tiles = []
        batch_coords = []

        for i, (x, y) in enumerate(coords):
            loc_x = int(round(x * level_downsample))
            loc_y = int(round(y * level_downsample))
            tile = slide.read_region((loc_x, loc_y), level, (tile_size, tile_size)).convert("RGB")

            batch_tiles.append(transform(tile))
            batch_coords.append([x, y])

            if len(batch_tiles) == BATCH_SIZE or i == len(coords) - 1:
                batch_tensor = torch.stack(batch_tiles).to(DEVICE)
                with torch.inference_mode():
                    emb = model(batch_tensor).cpu().numpy()
                all_embeddings.append(emb)
                batch_tiles = []

        all_embeddings = np.concatenate(all_embeddings, axis=0)

        with h5py.File(output_path, "w") as f:
            f.create_dataset("features", data=all_embeddings)
            f.create_dataset("coords", data=np.array(batch_coords))
            f.attrs["slide_name"] = slide_name
            f.attrs["n_tiles"] = len(coords)
            f.attrs["embed_dim"] = 1536
            f.attrs["level_downsample"] = level_downsample

        slide.close()

    except Exception as e:
        print(f"  ERROR on {slide_name}: {e}")

print("\nEmbedding extraction complete!")
