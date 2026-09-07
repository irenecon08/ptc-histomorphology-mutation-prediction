"""
extract_attention_crops.py

Extracts native-resolution (level 0, 40x) image crops from the highest- and
lowest-attention tiles of selected slides, for use as figure insets alongside
the whole-slide attention heatmaps in Section 5.4.

Reuses the coordinate/read_region logic from tiic_correlation_v6.py, which is
already validated: tile coordinates in the .h5 files are at the tiling level,
and level-0 position is (x * level_downsample, y * level_downsample), with the
tile covering (tile_size * level_downsample) pixels at level 0.

No HoVer-Net, no GPU. The only model work is one ABMIL forward pass per slide
over stored embeddings, which takes seconds on CPU.

SPATIAL SPREAD
--------------
Naive top-k selection tends to return k adjacent tiles from one small hotspot,
which misrepresents the attended region. This script instead walks down the
attention ranking and accepts a tile only if it is at least MIN_SEPARATION
tiles away (Chebyshev distance, in tile units) from every already-accepted
tile. Set --min-separation 0 to disable and get plain top-k.

OUTPUT
------
    attention_crops/{task}/{patient}/
        high_01_attn0.00412_x12345_y6789.png     individual crops, rank-ordered
        high_02_...
        low_01_...                                (only with --include-low)
        montage_high.png                          1 x N strip
        montage_high_labelled.png                 same, with rank/attn labels
        crop_manifest.csv                         provenance for every crop

USAGE
-----
    python extract_attention_crops.py --probe

    # the three Option B cases
    python extract_attention_crops.py --task multiclass \
        --patients TCGA-EL-A3ZR TCGA-EL-A3CX
    python extract_attention_crops.py --task ret_binary \
        --patients TCGA-EL-A3TB

    # architecture-level context instead of cytology
    python extract_attention_crops.py --task multiclass \
        --patients TCGA-EL-A3ZR --level 1 --n-crops 4

    # include low-attention crops for contrast panels
    python extract_attention_crops.py --task ret_binary \
        --patients TCGA-EL-A3TB --include-low

Run in venv310. Needs torch, h5py, openslide, numpy, pandas, PIL. CPU is fine.
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
import openslide
from PIL import Image, ImageDraw

# ==========================================================================
# CONFIG  (paths identical to tiic_correlation_v6.py)
# ==========================================================================
EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
TILES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/tiles_v2"
SLIDES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/slides"
RESULTS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2"
OUTPUT_DIR = "/cs/student/project_msc/2025/aibh/iconstan/attention_crops"

DEVICE = "cpu"          # ABMIL over stored embeddings; no GPU needed
EMBED_DIM = 1536

# Same authoritative architecture/model mapping as v6
TASK_ARCH = {
    "multiclass": {"attn_dim": 512, "fc_dim": 512,
                   "model_file": "best_multiclass_v2_retuned.pt"},
    "ret_binary": {"attn_dim": 512, "fc_dim": 256,
                   "model_file": "best_ret_binary_v2_retuned.pt"},
}

DEFAULT_N_CROPS = 5
DEFAULT_MIN_SEPARATION = 3   # in tile units
MONTAGE_PAD = 8
MONTAGE_BG = (255, 255, 255)


# ==========================================================================
# ABMIL  (identical to tiic_correlation_v6.py / train_abmil_v2_retrained.py)
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
    def __init__(self, embed_dim=1536, attn_dim=512, fc_dim=256,
                 num_classes=3, dropout=0.0):
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
    model.load_state_dict(state)     # strict: any mismatch must raise
    model.eval()
    return model, path


# ==========================================================================
# Data access  (identical helpers to v6)
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
                tile_size = int(h5.attrs["tile_size"])
                level_downsample = float(h5.attrs["level_downsample"])
            return coords, tile_size, level_downsample
    return None, None, None


def get_tile_attention_scores(model, patient):
    emb_file = get_embedding_file(patient)
    if emb_file is None:
        return None, None, None
    with h5py.File(emb_file, "r") as f:
        features = torch.tensor(f["features"][:], dtype=torch.float32).to(DEVICE)
        emb_coords = f["coords"][:]
    with torch.no_grad():
        logits, A = model(features)
    attn_scores = A.squeeze().cpu().numpy()
    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    return attn_scores, emb_coords, probs


# ==========================================================================
# Tile selection with spatial spread
# ==========================================================================
def select_spread_tiles(coords, attn, n_crops, min_separation, tile_size, descending=True):
    """
    Walk the attention ranking and accept a tile only if it is at least
    min_separation tiles from every already-accepted tile (Chebyshev distance
    in tile units). Falls back to plain rank order if fewer than n_crops tiles
    satisfy the constraint.
    """
    order = np.argsort(attn)
    if descending:
        order = order[::-1]

    if min_separation <= 0:
        return list(order[:n_crops])

    step = float(tile_size)     # coords are in tiling-level pixels
    accepted = []
    for idx in order:
        x, y = float(coords[idx][0]), float(coords[idx][1])
        ok = True
        for aidx in accepted:
            ax, ay = float(coords[aidx][0]), float(coords[aidx][1])
            cheb = max(abs(x - ax), abs(y - ay)) / step
            if cheb < min_separation:
                ok = False
                break
        if ok:
            accepted.append(idx)
        if len(accepted) == n_crops:
            break

    if len(accepted) < n_crops:
        print(f"      only {len(accepted)}/{n_crops} tiles met the "
              f"separation constraint; filling from rank order")
        for idx in order:
            if idx not in accepted:
                accepted.append(idx)
            if len(accepted) == n_crops:
                break
    return accepted


# ==========================================================================
# Crop extraction
# ==========================================================================
def extract_crop(slide, x, y, tile_size, level_downsample, level, out_size,
                  via_level0=True):
    """
    A tile at (x, y) in tiling-level pixels covers, at level 0:
        origin (x * level_downsample, y * level_downsample)
        extent (tile_size * level_downsample) pixels square

    read_region always takes level-0 coordinates for its location argument,
    but returns `size` pixels at the requested `level`. To capture the tile's
    full physical footprint at `level`, the size requested must be the level-0
    extent divided by that level's own downsample.

    via_level0 (default True): on some machines (observed on knuckles,
    CentOS 7), openslide's decoder for non-baseline pyramid levels of these
    SVS files raises SIGILL - an illegal-instruction crash, likely a codec
    code path compiled for an instruction set this CPU lacks. Level 0
    (baseline) decoding is unaffected and works fine on the same machine.
    As a workaround, when via_level0 is True and level > 0, this function
    reads the full-resolution region at level 0 and downsamples it with PIL
    (LANCZOS) to the target level's pixel size instead of asking openslide
    to decode that pyramid level directly. Since an SVS's lower-resolution
    levels are themselves downsamples of level 0 (with mild recompression),
    this gives visually equivalent output for figure purposes, though it is
    not byte-identical to the slide's own pre-computed level-1 tiles. This
    substitution is noted in the crop manifest via the `decoded_via` column.
    """
    x0 = int(round(x * level_downsample))
    y0 = int(round(y * level_downsample))
    extent_l0 = int(round(tile_size * level_downsample))

    ds_at_level = slide.level_downsamples[level]
    size_at_level = max(1, int(round(extent_l0 / ds_at_level)))

    if level > 0 and via_level0:
        img = slide.read_region((x0, y0), 0, (extent_l0, extent_l0)).convert("RGB")
        img = img.resize((size_at_level, size_at_level), Image.LANCZOS)
        decoded_via = "level0_downsampled"
    else:
        img = slide.read_region((x0, y0), level, (size_at_level, size_at_level)).convert("RGB")
        decoded_via = f"native_level{level}"

    if out_size and img.size != (out_size, out_size):
        img = img.resize((out_size, out_size), Image.LANCZOS)
    return img, x0, y0, extent_l0, decoded_via


def make_montage(images, labels=None, pad=MONTAGE_PAD):
    if not images:
        return None
    w, h = images[0].size
    label_h = 22 if labels else 0
    total_w = len(images) * w + (len(images) + 1) * pad
    total_h = h + 2 * pad + label_h
    canvas = Image.new("RGB", (total_w, total_h), MONTAGE_BG)
    for i, im in enumerate(images):
        canvas.paste(im, (pad + i * (w + pad), pad))
    if labels:
        draw = ImageDraw.Draw(canvas)
        for i, lab in enumerate(labels):
            draw.text((pad + i * (w + pad) + 2, pad + h + 4), lab, fill=(0, 0, 0))
    return canvas


# ==========================================================================
# Per-slide driver
# ==========================================================================
def process_slide(patient, task, model, args):
    print(f"\n  {patient}")

    coords, tile_size, level_downsample = get_tile_info(patient)
    if coords is None:
        print("    SKIP: no tile .h5 found")
        return None
    slide_path = get_slide_file(patient)
    if slide_path is None:
        print("    SKIP: no .svs found")
        return None

    attn, emb_coords, probs = get_tile_attention_scores(model, patient)
    if attn is None:
        print("    SKIP: no embeddings found")
        return None

    # Attention is indexed by the embedding file's coord order; use that
    # directly rather than the tile file's, so index alignment is guaranteed.
    coords_use = emb_coords
    if len(attn) != len(coords_use):
        print(f"    SKIP: attention/coord length mismatch "
              f"({len(attn)} vs {len(coords_use)})")
        return None

    print(f"    {len(attn)} tiles, tile_size={tile_size}, "
          f"level_downsample={level_downsample}")
    print(f"    attention range {attn.min():.3e} to {attn.max():.3e}, "
          f"sum {attn.sum():.4f}")
    if probs is not None and probs.ndim == 1:
        print(f"    predicted probabilities: "
              f"{np.array2string(probs, precision=3)}")

    slide = openslide.OpenSlide(slide_path)
    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X, "unknown")
    ds_at_level = slide.level_downsamples[args.level]
    extent_l0 = int(round(tile_size * level_downsample))
    try:
        um = float(mpp_x) * extent_l0
        physical = f"{um:.0f} um square"
    except (TypeError, ValueError):
        physical = "unknown physical size"
    print(f"    level {args.level} (downsample {ds_at_level:.1f}), "
          f"each crop covers {physical}")

    out_dir = os.path.join(OUTPUT_DIR, task, patient)
    os.makedirs(out_dir, exist_ok=True)

    manifest_rows = []
    groups = [("high", True)]
    if args.include_low:
        groups.append(("low", False))

    for group, descending in groups:
        picked = select_spread_tiles(coords_use, attn, args.n_crops,
                                     args.min_separation, tile_size,
                                     descending=descending)
        images, labels = [], []
        for rank, idx in enumerate(picked, start=1):
            x, y = int(coords_use[idx][0]), int(coords_use[idx][1])
            a = float(attn[idx])
            img, x0, y0, ext, decoded_via = extract_crop(
                slide, x, y, tile_size, level_downsample, args.level,
                args.out_size, via_level0=not args.native_level_read)
            fname = f"{group}_{rank:02d}_attn{a:.5f}_x{x}_y{y}.png"
            img.save(os.path.join(out_dir, fname))
            images.append(img)
            labels.append(f"{rank}  a={a:.4f}")
            manifest_rows.append({
                "patient": patient, "task": task, "group": group, "rank": rank,
                "attention": a, "attention_percentile":
                    float((attn < a).mean() * 100),
                "tile_x": x, "tile_y": y, "level0_x": x0, "level0_y": y0,
                "level0_extent_px": ext, "read_level": args.level,
                "decoded_via": decoded_via,
                "level_downsample_tiling": level_downsample,
                "slide_mpp_x": mpp_x, "filename": fname,
            })
            print(f"      {group} {rank}: attn={a:.5f} "
                  f"(pct {(attn < a).mean()*100:.1f}) at level0 ({x0}, {y0})")

        m = make_montage(images)
        if m:
            m.save(os.path.join(out_dir, f"montage_{group}.png"))
        ml = make_montage(images, labels)
        if ml:
            ml.save(os.path.join(out_dir, f"montage_{group}_labelled.png"))

    slide.close()

    mpath = os.path.join(out_dir, "crop_manifest.csv")
    with open(mpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"    wrote {len(manifest_rows)} crops + montages to {out_dir}")
    return manifest_rows


# ==========================================================================
# Probe
# ==========================================================================
def probe(args):
    print("=" * 66)
    print("PROBE")
    print("=" * 66)
    print(f"device: {DEVICE}")
    for task, arch in TASK_ARCH.items():
        path = os.path.join(RESULTS_DIR, arch["model_file"])
        print(f"\n{task}: {arch['model_file']}  exists={os.path.exists(path)}")
        if not os.path.exists(path):
            continue
        try:
            model, _ = load_abmil_model(task)
            dummy = torch.randn(20, EMBED_DIM).to(DEVICE)
            with torch.no_grad():
                logits, A = model(dummy)
            print(f"  loaded and forward pass OK; attention sums to "
                  f"{A.sum().item():.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")

    for p in (args.patients or []):
        print(f"\n{p}")
        c, ts, ld = get_tile_info(p)
        print(f"  tiles:      {'ok, %d tiles' % len(c) if c is not None else 'MISSING'}")
        if c is not None:
            print(f"              tile_size={ts}, level_downsample={ld}")
        print(f"  embeddings: {get_embedding_file(p) or 'MISSING'}")
        sp = get_slide_file(p)
        print(f"  slide:      {sp or 'MISSING'}")
        if sp:
            s = openslide.OpenSlide(sp)
            print(f"              levels={s.level_count}, "
                  f"downsamples={[round(d,2) for d in s.level_downsamples]}")
            print(f"              mpp_x="
                  f"{s.properties.get(openslide.PROPERTY_NAME_MPP_X,'unknown')}")
            s.close()


# ==========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="check models, tiles, embeddings and slides, then exit")
    ap.add_argument("--task", choices=["multiclass", "ret_binary"],
                    default="multiclass")
    ap.add_argument("--patients", nargs="+", required=False,
                    help="patient IDs, e.g. TCGA-EL-A3ZR TCGA-EL-A3CX")
    ap.add_argument("--n-crops", type=int, default=DEFAULT_N_CROPS)
    ap.add_argument("--min-separation", type=int, default=DEFAULT_MIN_SEPARATION,
                    help="minimum spacing between crops in tile units; 0 = plain top-k")
    ap.add_argument("--level", type=int, default=0,
                    help="0 = native 40x (cytology), 1 = ~10x (architecture)")
    ap.add_argument("--out-size", type=int, default=None,
                    help="resize each crop to this square size; default keeps native")
    ap.add_argument("--include-low", action="store_true",
                    help="also extract lowest-attention crops for contrast")
    ap.add_argument("--native-level-read", action="store_true",
                    help="force openslide to decode the requested pyramid level "
                         "directly instead of downsampling from level 0; use "
                         "only on a machine where this doesn't crash (SIGILL "
                         "was observed on knuckles for level>0 reads)")
    args = ap.parse_args()

    if args.probe:
        probe(args)
        sys.exit(0)

    if not args.patients:
        sys.exit("--patients is required (or use --probe)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, model_path = load_abmil_model(args.task)
    print(f"Loaded {args.task} model: {model_path}")
    print(f"  attn_dim={TASK_ARCH[args.task]['attn_dim']}, "
          f"fc_dim={TASK_ARCH[args.task]['fc_dim']}")
    print(f"  n_crops={args.n_crops}, min_separation={args.min_separation}, "
          f"level={args.level}")

    all_rows = []
    for patient in args.patients:
        rows = process_slide(patient, args.task, model, args)
        if rows:
            all_rows.extend(rows)

    if all_rows:
        combined = os.path.join(OUTPUT_DIR, f"crop_manifest_{args.task}.csv")
        with open(combined, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCombined manifest: {combined}")
    print("\nDone.")
