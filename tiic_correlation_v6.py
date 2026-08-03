"""
tiic_correlation_v6.py

The corrected TIIC attention-immune correlation analysis. Merges two
independent fixes that were made in two separate places and never combined
until now:

  1. CORRECT MODELS (fixed in a separate conversation, uploaded here as
     tiic_correlation_v5.py): loads best_multiclass_v2_retuned.pt
     (attn=512, fc=512) and best_ret_binary_v2_retuned.pt (attn=512, fc=256)
     - the actual final, leakage-free-tuned models - rather than the
     superseded best_{task}_v2.pt files with the old architecture/weights.

  2. FULL SUB-CROP COVERAGE (fixed in this conversation): nucleus detection
     uses a 4x4 grid of 256x256 crops (16 crops/tile, stride 256px) giving
     genuine 100% coverage of each tile's 1024x1024 level-0 footprint, not
     the 2x2 / 25%-coverage design shown to systematically bias the partial
     correlation toward more negative values (see subcrop_validation_*.csv
     and v4_vs_v5_*.csv in tiic_analysis_v4/ and tiic_analysis_v5/).

Neither fix alone is correct: the uploaded v5.py has (1) but not (2); this
thread's v5.py had (2) but not (1) (it inherited attention from v4, which
loads the wrong models). tiic_analysis_v5/ on disk was produced with correct
coverage but WRONG models and must not be used for final reporting.

Also corrects slide selection: uses the FULL 80-slide held-out test set for
each task, not the 17-slide curated subset (3/class multiclass, 3+5 RET) used
by both prior v5 scripts.

OUTPUT
------
Written to tiic_analysis_v6/ (v4 and v5 outputs are untouched).

    per_slide/{task}/{patient}.csv   per-slide tile-level results (resume unit)
    tiic_summary_{task}.csv          per-slide correlations + BH correction
    tiic_per_tile_{task}.csv         pooled tile-level data
    pooled_results_{task}.json       pooled statistics
    v5_vs_v6_{task}.csv              paired comparison vs the wrong-model v5
                                      run, isolating the effect of the model
                                      fix alone (both used full coverage)

USAGE
-----
    python tiic_correlation_v6.py --probe                     # sanity check only
    python tiic_correlation_v6.py --task ret_binary --limit 3 # smoke test
    python tiic_correlation_v6.py --task ret_binary           # full run
    python tiic_correlation_v6.py --task multiclass
    python tiic_correlation_v6.py --task ret_binary --aggregate-only

Fully resumable: slides with an existing per-slide CSV are skipped. Expect
roughly 160s/slide (16 HoVer-Net crops/tile), ~3.5h per task for the
remaining slides. Run inside screen.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
import openslide
from scipy.stats import spearmanr, rankdata
from statsmodels.stats.multitest import multipletests

from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

# ==========================================================================
# CONFIG
# ==========================================================================
EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
TILES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/tiles_v2"
SLIDES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/slides"
SPLITS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"
RESULTS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2"
OUTPUT_DIR = "/cs/student/project_msc/2025/aibh/iconstan/tiic_analysis_v6/"
V5_DIR = "/cs/student/project_msc/2025/aibh/iconstan/tiic_analysis_v5/"  # wrong-model, correct-coverage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM = 1536
N_TILES_PER_SLIDE = 250
HOVERNET_SUBCROP_SIZE = 256
SUBCROPS_PER_TILE = 4          # CORRECTED: 4x4 grid = 16 crops = full coverage
RANDOM_SEED = 42

# CORRECTED: authoritative per-task architecture and model files
# (per results_v2_model_manifest.json)
TASK_ARCH = {
    "multiclass": {"attn_dim": 512, "fc_dim": 512, "model_file": "best_multiclass_v2_retuned.pt"},
    "ret_binary": {"attn_dim": 512, "fc_dim": 256, "model_file": "best_ret_binary_v2_retuned.pt"},
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "per_slide"), exist_ok=True)
np.random.seed(RANDOM_SEED)

TYPE_NAMES = {0: "Background/Other", 1: "Epithelial", 2: "Lymphocyte", 3: "Macrophage", 4: "Neutrophil"}
IMMUNE_TYPE_IDS = {2, 3, 4}
EPITHELIAL_TYPE_ID = 1
ALPHA = 0.05


# ==========================================================================
# ABMIL model (matches train_abmil_v2_retrained.py exactly)
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
    model.load_state_dict(state)   # deliberately NOT strict=False: any mismatch must raise
    model.eval()
    return model, path


# ==========================================================================
# Data access
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
        return None, None
    with h5py.File(emb_file, "r") as f:
        features = torch.tensor(f["features"][:], dtype=torch.float32).to(DEVICE)
        emb_coords = f["coords"][:]
    with torch.no_grad():
        _, A = model(features)
    attn_scores = A.squeeze().cpu().numpy()
    return attn_scores, emb_coords


def compute_nuclei_counts_for_tile(slide, x, y, level_downsample, orig_tile_size, segmentor):
    """4x4 grid, stride = orig_tile_size_l0 / 4 = full, non-overlapping coverage."""
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
    x_r, y_r, z_r = rankdata(x), rankdata(y), rankdata(z)

    def residualize(target, control):
        control_with_const = np.column_stack([np.ones(len(control)), control])
        coefs, _, _, _ = np.linalg.lstsq(control_with_const, target, rcond=None)
        return target - control_with_const @ coefs

    x_resid = residualize(x_r, z_r)
    y_resid = residualize(y_r, z_r)
    r, p = spearmanr(x_resid, y_resid)
    return r, p


def per_slide_path(task, patient):
    d = os.path.join(OUTPUT_DIR, "per_slide", task)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{patient}.csv")


# ==========================================================================
# Per-slide analysis
# ==========================================================================
def analyse_slide(patient, model, true_class, mutation_label_str, segmentor):
    coords, tile_size, level_downsample = get_tile_info(patient)
    slide_path = get_slide_file(patient)
    if coords is None or slide_path is None:
        print(f"    skipping {patient}: missing tile info or slide file")
        return None

    attn_scores, emb_coords = get_tile_attention_scores(model, patient)
    if attn_scores is None:
        print(f"    skipping {patient}: no embeddings found")
        return None

    coord_to_attn = {(int(x), int(y)): float(a) for (x, y), a in zip(emb_coords, attn_scores)}
    all_coords = [(int(x), int(y)) for x, y in coords]
    if len(all_coords) > N_TILES_PER_SLIDE:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(all_coords), N_TILES_PER_SLIDE, replace=False)
        sampled_coords = [all_coords[i] for i in idx]
    else:
        sampled_coords = all_coords

    slide = openslide.OpenSlide(slide_path)
    rows = []
    t0 = time.time()
    n = len(sampled_coords)

    for i, (x, y) in enumerate(sampled_coords):
        attn = coord_to_attn.get((x, y))
        if attn is None:
            continue
        counts = compute_nuclei_counts_for_tile(slide, x, y, level_downsample, tile_size, segmentor)
        if counts is None:
            continue
        total = counts["total_nuclei"]
        if total <= 0:
            continue
        rows.append({
            "patient": patient, "true_class": mutation_label_str, "x": x, "y": y,
            "attn": attn, "total_nuclei": total,
            "immune_nuclei": counts["immune_nuclei"], "epithelial_nuclei": counts["epithelial_nuclei"],
            "immune_density": counts["immune_nuclei"] / total,
            "epithelial_density": counts["epithelial_nuclei"] / total,
        })
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"      {i+1}/{n} tiles ({el:.0f}s, {el/(i+1)*n:.0f}s projected)", flush=True)

    slide.close()
    if len(rows) < 10:
        print(f"    skipping {patient}: only {len(rows)} valid tiles")
        return None
    return pd.DataFrame(rows)


def summarise_slide(tile_df):
    a = tile_df["attn"].values
    imm_d = tile_df["immune_density"].values
    imm_n = tile_df["immune_nuclei"].values
    epi_d = tile_df["epithelial_density"].values

    r_id, p_id = spearmanr(a, imm_d)
    r_ic, p_ic = spearmanr(a, imm_n)
    r_ed, p_ed = spearmanr(a, epi_d)
    pr, pp = partial_spearman(a, imm_d, epi_d)

    return {
        "patient": tile_df["patient"].iloc[0],
        "true_class": tile_df["true_class"].iloc[0],
        "n_tiles": len(tile_df),
        "spearman_r_immune_density": r_id, "spearman_p_immune_density": p_id,
        "spearman_r_immune_count": r_ic, "spearman_p_immune_count": p_ic,
        "spearman_r_epithelial_density": r_ed, "spearman_p_epithelial_density": p_ed,
        "partial_r_immune_given_epithelial": pr, "partial_p_immune_given_epithelial": pp,
    }


# ==========================================================================
# Main run
# ==========================================================================
def get_full_test_slides(task, splits_df):
    """FULL 80-slide test set, not the 17-slide curated subset used by both prior v5 scripts."""
    test_df = splits_df[splits_df["split"] == "test"]
    label_map = {"BRAF_V600E": 0, "RAS": 1, "Other": 2}
    examples = []
    if task == "multiclass":
        for _, row in test_df.iterrows():
            lbl = row["mutation_label"]
            if lbl in label_map:
                examples.append((row["patient"], label_map[lbl], lbl))
    else:
        for _, row in test_df.iterrows():
            ret = row["RET"]
            label_str = "RET_positive" if ret == 1 else "RET_negative"
            examples.append((row["patient"], int(ret), label_str))
    return examples


def run_task(task, segmentor, limit=None):
    model, model_path = load_abmil_model(task)
    print(f"  Loaded {task} model from {model_path}")
    print(f"  Architecture: attn_dim={TASK_ARCH[task]['attn_dim']}, fc_dim={TASK_ARCH[task]['fc_dim']}")

    splits_df = pd.read_csv(SPLITS_PATH)
    examples = get_full_test_slides(task, splits_df)
    if limit:
        examples = examples[:limit]

    done = [p for p, _, _ in examples if os.path.exists(per_slide_path(task, p))]
    todo = [(p, c, l) for p, c, l in examples if p not in done]

    print(f"\n{'='*66}\nv6 CORRECTED RUN: {task}\n{'='*66}")
    print(f"  full test-set pool: {len(examples)}   done: {len(done)}   todo: {len(todo)}")

    if not todo:
        print("  Nothing to run.")
        return

    overall = time.time()
    for i, (patient, true_class, label_str) in enumerate(todo):
        el = (time.time() - overall) / 60
        eta = (el / max(i, 1)) * (len(todo) - i) if i else 0
        print(f"\n[{i+1}/{len(todo)}] {patient} ({label_str})  elapsed {el:.1f} min"
              + (f", ETA {eta:.0f} min" if i else ""))
        t0 = time.time()
        tile_df = analyse_slide(patient, model, true_class, label_str, segmentor)
        if tile_df is None:
            continue
        tile_df.to_csv(per_slide_path(task, patient), index=False)
        print(f"    done in {time.time()-t0:.0f}s -> {len(tile_df)} tiles saved")

    print(f"\n  Run complete in {(time.time()-overall)/60:.1f} min")


# ==========================================================================
# Aggregation and comparison
# ==========================================================================
def aggregate(task):
    d = os.path.join(OUTPUT_DIR, "per_slide", task)
    files = sorted(f for f in os.listdir(d) if f.endswith(".csv")) if os.path.isdir(d) else []
    if not files:
        sys.exit(f"No per-slide results at {d}")

    tile_dfs, summaries = [], []
    for f in files:
        td = pd.read_csv(os.path.join(d, f))
        if len(td) < 10:
            continue
        tile_dfs.append(td)
        summaries.append(summarise_slide(td))

    summary_df = pd.DataFrame(summaries)
    pooled = pd.concat(tile_dfs, ignore_index=True)

    rej_raw, p_raw_adj, _, _ = multipletests(
        summary_df["spearman_p_immune_density"].values, alpha=ALPHA, method="fdr_bh")
    summary_df["spearman_p_bh_corrected"] = p_raw_adj
    summary_df["significant_after_correction"] = rej_raw

    rej_par, p_par_adj, _, _ = multipletests(
        summary_df["partial_p_immune_given_epithelial"].values, alpha=ALPHA, method="fdr_bh")
    summary_df["partial_p_bh_corrected"] = p_par_adj
    summary_df["partial_significant_after_correction"] = rej_par

    summary_df.to_csv(os.path.join(OUTPUT_DIR, f"tiic_summary_{task}.csv"), index=False)
    pooled.to_csv(os.path.join(OUTPUT_DIR, f"tiic_per_tile_{task}.csv"), index=False)

    pr_id, pp_id = spearmanr(pooled["attn"], pooled["immune_density"])
    pr_ic, pp_ic = spearmanr(pooled["attn"], pooled["immune_nuclei"])
    pr_ed, pp_ed = spearmanr(pooled["attn"], pooled["epithelial_density"])
    pr_pa, pp_pa = partial_spearman(pooled["attn"].values, pooled["immune_density"].values,
                                     pooled["epithelial_density"].values)

    pooled_results = {
        "task": task, "coverage": "full (4x4 subcrops, 100%)",
        "model_file": TASK_ARCH[task]["model_file"],
        "n_tiles_pooled": int(len(pooled)), "n_slides": int(len(summary_df)),
        "pooled_r_immune_density": float(pr_id), "pooled_p_immune_density": float(pp_id),
        "pooled_r_immune_count": float(pr_ic), "pooled_p_immune_count": float(pp_ic),
        "pooled_r_epithelial_density": float(pr_ed), "pooled_p_epithelial_density": float(pp_ed),
        "pooled_partial_r": float(pr_pa), "pooled_partial_p": float(pp_pa),
    }
    with open(os.path.join(OUTPUT_DIR, f"pooled_results_{task}.json"), "w") as f:
        json.dump(pooled_results, f, indent=2)

    print(f"\n{'='*66}\nv6 AGGREGATE - {task}   ({len(summary_df)} slides, {len(pooled)} tiles)")
    print(f"  model: {TASK_ARCH[task]['model_file']}")
    print(f"{'='*66}")
    print(f"  raw r (attn~immune)     mean {summary_df['spearman_r_immune_density'].mean():+.3f}"
          f"  median {summary_df['spearman_r_immune_density'].median():+.3f}")
    print(f"  raw r (attn~epithelial) mean {summary_df['spearman_r_epithelial_density'].mean():+.3f}")
    print(f"  partial r               mean {summary_df['partial_r_immune_given_epithelial'].mean():+.3f}"
          f"  median {summary_df['partial_r_immune_given_epithelial'].median():+.3f}")
    print(f"  raw significant (BH)     {int(rej_raw.sum())} / {len(summary_df)}")
    print(f"  partial significant (BH) {int(rej_par.sum())} / {len(summary_df)}")
    sig = summary_df[rej_par]
    if len(sig):
        neg = int((sig['partial_r_immune_given_epithelial'] < 0).sum())
        print(f"    of those: {neg} negative, {len(sig)-neg} positive")
    print(f"\n  POOLED  r(attn,immune)={pr_id:+.3f}  r(attn,epi)={pr_ed:+.3f}  "
          f"partial={pr_pa:+.3f} (p={pp_pa:.3g})")
    return summary_df


def compare_v5_v6(task):
    """
    Isolates the effect of the MODEL fix alone: v5 (tiic_analysis_v5/) used
    correct 4x4 coverage but the WRONG models; v6 uses correct coverage AND
    correct models. Both used full coverage, so any difference here is due
    to the model fix specifically, not a coverage confound.
    """
    v5p = os.path.join(V5_DIR, f"tiic_summary_{task}.csv")
    v6p = os.path.join(OUTPUT_DIR, f"tiic_summary_{task}.csv")
    if not (os.path.exists(v5p) and os.path.exists(v6p)):
        print("  (skipping v5/v6 comparison - one summary missing)")
        return

    cols = ["patient", "spearman_r_immune_density", "spearman_r_epithelial_density",
            "partial_r_immune_given_epithelial"]
    v5 = pd.read_csv(v5p)[cols]
    v6 = pd.read_csv(v6p)[cols]
    m = v5.merge(v6, on="patient", suffixes=("_v5_wrongmodel", "_v6_correct"))
    if not len(m):
        return

    for base in ["spearman_r_immune_density", "spearman_r_epithelial_density",
                 "partial_r_immune_given_epithelial"]:
        m[f"d_{base}"] = m[f"{base}_v6_correct"] - m[f"{base}_v5_wrongmodel"]
    m.to_csv(os.path.join(OUTPUT_DIR, f"v5_vs_v6_{task}.csv"), index=False)

    print(f"\n{'='*66}\nv5 (correct coverage, WRONG model) vs v6 (correct coverage, "
          f"correct model) - {task}, n={len(m)}\n{'='*66}")
    for lab, base in [("attn~immune", "spearman_r_immune_density"),
                      ("attn~epithelial", "spearman_r_epithelial_density"),
                      ("partial", "partial_r_immune_given_epithelial")]:
        rho, _ = spearmanr(m[f"{base}_v5_wrongmodel"], m[f"{base}_v6_correct"])
        diff = m[f"d_{base}"]
        print(f"  {lab:16s} v5(wrong) {m[f'{base}_v5_wrongmodel'].mean():+.3f} -> "
              f"v6(correct) {m[f'{base}_v6_correct'].mean():+.3f}   "
              f"mean diff {diff.mean():+.3f}   rho={rho:+.3f}")
    print(f"\n  This isolates the MODEL fix (coverage was already correct in both). "
          f"Saved: {os.path.join(OUTPUT_DIR, f'v5_vs_v6_{task}.csv')}")


# ==========================================================================
# Probe
# ==========================================================================
def probe():
    print("=" * 66)
    print("PROBE")
    print("=" * 66)
    for task, arch in TASK_ARCH.items():
        path = os.path.join(RESULTS_DIR, arch["model_file"])
        exists = os.path.exists(path)
        print(f"\n{task}: {arch['model_file']}  exists={exists}")
        if not exists:
            continue
        try:
            model, _ = load_abmil_model(task)
            print(f"  loaded OK. attn_dim={arch['attn_dim']} fc_dim={arch['fc_dim']}")
            dummy = torch.randn(20, EMBED_DIM).to(DEVICE)
            with torch.no_grad():
                logits, A = model(dummy)
            print(f"  forward pass OK. logits shape={tuple(logits.shape)} "
                  f"(expect num_classes={3 if task=='multiclass' else 2})")
            print(f"  attention sums to {A.sum().item():.4f} (expect ~1.0)")
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nOld v5 dir (wrong model, correct coverage) present: {os.path.isdir(V5_DIR)}")
    for task in TASK_ARCH:
        p = os.path.join(V5_DIR, f"tiic_summary_{task}.csv")
        print(f"  {task}: {p}  exists={os.path.exists(p)}")


# ==========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--task", choices=["multiclass", "ret_binary", "both"], default="ret_binary")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.probe:
        probe()
        sys.exit(0)

    tasks = ["ret_binary", "multiclass"] if args.task == "both" else [args.task]

    if not args.aggregate_only:
        print(f"Device: {DEVICE}")
        print("Loading HoVer-Net (MoNuSAC)...")
        segmentor = MultiTaskSegmentor(
            model="hovernet_fast-monusac", batch_size=4, num_workers=1,
            device="cuda" if DEVICE == "cuda" else "cpu", verbose=False)
        print("  Loaded.")
        for task in tasks:
            run_task(task, segmentor, limit=args.limit)

    for task in tasks:
        aggregate(task)
        compare_v5_v6(task)

    print("\nv6 complete. v4 and v5 outputs are unchanged.")
