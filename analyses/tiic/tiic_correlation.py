"""Correlation between ABMIL attention and tumour-infiltrating immune cells (TIIC).

For every held-out test slide (dissertation §3.12.1):

* up to 250 tiles are sampled (seed 42);
* HoVer-Net (MoNuSAC weights) classifies the nuclei in each tile, read as a 4x4 grid
  of 256x256 level-0 crops covering the tile's whole 1024x1024 level-0 footprint;
* immune density = (lymphocytes + macrophages + neutrophils) / all nuclei, and
  epithelial density likewise;
* per slide, Spearman correlations between attention and immune density, immune
  count and epithelial density, plus the partial correlation of attention with immune
  density controlling for epithelial density (rank-residualised); p-values are
  Benjamini-Hochberg corrected across slides;
* the same statistics on all tiles pooled.

Outputs in ``results/analyses/tiic/``:

    per_slide/{task}/{patient}.csv   tile-level results per slide (the resume unit)
    tiic_summary_{task}.csv          per-slide correlations with BH correction
    tiic_per_tile_{task}.csv         pooled tile-level data
    pooled_results_{task}.json       pooled statistics

Usage:

    python analyses/tiic/tiic_correlation.py --probe
    python analyses/tiic/tiic_correlation.py --task ret_binary --limit 3   # smoke test
    python analyses/tiic/tiic_correlation.py --task both
    python analyses/tiic/tiic_correlation.py --task both --aggregate-only

Slides with an existing per-slide CSV are skipped, so runs can be resumed. Expect
about 160 s per slide on a GPU.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import openslide
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from statsmodels.stats.multitest import multipletests

from ptc_mil.analysis import load_context

ctx = load_context()
OUTPUT_DIR = str(ctx.output_dir("tiic"))
DEVICE = ctx.device

N_TILES_PER_SLIDE = 250
HOVERNET_SUBCROP_SIZE = 256
SUBCROPS_PER_TILE = 4  # 4x4 grid = 16 crops = full coverage of the tile
RANDOM_SEED = 42

TYPE_NAMES = {
    0: "Background/Other",
    1: "Epithelial",
    2: "Lymphocyte",
    3: "Macrophage",
    4: "Neutrophil",
}
IMMUNE_TYPE_IDS = {2, 3, 4}
EPITHELIAL_TYPE_ID = 1
ALPHA = 0.05
TASKS = ("multiclass", "ret_binary")


def load_segmentor():
    """HoVer-Net trained on MoNuSAC, via TIAToolbox."""
    from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

    return MultiTaskSegmentor(
        model="hovernet_fast-monusac",
        batch_size=4,
        num_workers=1,
        device="cuda" if DEVICE.type == "cuda" else "cpu",
        verbose=False,
    )


def load_abmil_model(task):
    return ctx.load_abmil(task)


# ==========================================================================
# Nucleus counting and statistics
# ==========================================================================
def compute_nuclei_counts_for_tile(slide, x, y, level_downsample, orig_tile_size, segmentor):
    """Count nuclei by type over a SUBCROPS_PER_TILE x SUBCROPS_PER_TILE grid of crops."""
    orig_tile_size_l0 = orig_tile_size * level_downsample
    x0_l0 = x * level_downsample
    y0_l0 = y * level_downsample
    step = orig_tile_size_l0 / SUBCROPS_PER_TILE

    sub_crops = [
        (int(round(x0_l0 + i * step)), int(round(y0_l0 + j * step)))
        for i in range(SUBCROPS_PER_TILE)
        for j in range(SUBCROPS_PER_TILE)
    ]
    sub_images = np.array(
        [
            np.array(
                slide.read_region(
                    (cx, cy), 0, (HOVERNET_SUBCROP_SIZE, HOVERNET_SUBCROP_SIZE)
                ).convert("RGB")
            )
            for (cx, cy) in sub_crops
        ]
    )

    try:
        output = segmentor.run(
            images=sub_images, patch_mode=True, save_dir=None, output_type="dict"
        )
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
    return {
        "total_nuclei": total_nuclei,
        "immune_nuclei": immune_nuclei,
        "epithelial_nuclei": epithelial_nuclei,
    }


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after regressing the ranks of z out of both."""
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
    coords, tile_size, level_downsample = ctx.tile_info(patient)
    slide_path = ctx.slide_file(patient)
    if coords is None or slide_path is None:
        print(f"    skipping {patient}: missing tile info or slide file")
        return None

    attn_scores, emb_coords = ctx.tile_attention(model, patient)
    if attn_scores is None:
        print(f"    skipping {patient}: no embeddings found")
        return None

    coord_to_attn = {
        (int(x), int(y)): float(a) for (x, y), a in zip(emb_coords, attn_scores, strict=True)
    }
    all_coords = [(int(x), int(y)) for x, y in coords]
    if len(all_coords) > N_TILES_PER_SLIDE:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(all_coords), N_TILES_PER_SLIDE, replace=False)
        sampled_coords = [all_coords[i] for i in idx]
    else:
        sampled_coords = all_coords

    slide = openslide.OpenSlide(str(slide_path))
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
        rows.append(
            {
                "patient": patient,
                "true_class": mutation_label_str,
                "x": x,
                "y": y,
                "attn": attn,
                "total_nuclei": total,
                "immune_nuclei": counts["immune_nuclei"],
                "epithelial_nuclei": counts["epithelial_nuclei"],
                "immune_density": counts["immune_nuclei"] / total,
                "epithelial_density": counts["epithelial_nuclei"] / total,
            }
        )
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(
                f"      {i + 1}/{n} tiles ({el:.0f}s, {el / (i + 1) * n:.0f}s projected)",
                flush=True,
            )

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
        "spearman_r_immune_density": r_id,
        "spearman_p_immune_density": p_id,
        "spearman_r_immune_count": r_ic,
        "spearman_p_immune_count": p_ic,
        "spearman_r_epithelial_density": r_ed,
        "spearman_p_epithelial_density": p_ed,
        "partial_r_immune_given_epithelial": pr,
        "partial_p_immune_given_epithelial": pp,
    }


# ==========================================================================
# Main run
# ==========================================================================
def get_full_test_slides(task, splits_df):
    """Every held-out test slide for the task, as (patient, class index, class name)."""
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
    settings = ctx.config.task(task)
    print(f"  Loaded {task} model from {model_path}")
    print(f"  Architecture: attn_dim={settings.attn_dim}, fc_dim={settings.fc_dim}")

    examples = get_full_test_slides(task, ctx.splits)
    if limit:
        examples = examples[:limit]

    done = [p for p, _, _ in examples if os.path.exists(per_slide_path(task, p))]
    todo = [(p, c, lbl) for p, c, lbl in examples if p not in done]

    print(f"\n{'=' * 66}\nTIIC run: {task}\n{'=' * 66}")
    print(f"  test slides: {len(examples)}   done: {len(done)}   todo: {len(todo)}")
    if not todo:
        print("  Nothing to run.")
        return

    overall = time.time()
    for i, (patient, true_class, label_str) in enumerate(todo):
        el = (time.time() - overall) / 60
        eta = (el / max(i, 1)) * (len(todo) - i) if i else 0
        print(
            f"\n[{i + 1}/{len(todo)}] {patient} ({label_str})  elapsed {el:.1f} min"
            + (f", ETA {eta:.0f} min" if i else "")
        )
        t0 = time.time()
        tile_df = analyse_slide(patient, model, true_class, label_str, segmentor)
        if tile_df is None:
            continue
        tile_df.to_csv(per_slide_path(task, patient), index=False)
        print(f"    done in {time.time() - t0:.0f}s -> {len(tile_df)} tiles saved")

    print(f"\n  Run complete in {(time.time() - overall) / 60:.1f} min")


# ==========================================================================
# Aggregation
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
        summary_df["spearman_p_immune_density"].values, alpha=ALPHA, method="fdr_bh"
    )
    summary_df["spearman_p_bh_corrected"] = p_raw_adj
    summary_df["significant_after_correction"] = rej_raw

    rej_par, p_par_adj, _, _ = multipletests(
        summary_df["partial_p_immune_given_epithelial"].values, alpha=ALPHA, method="fdr_bh"
    )
    summary_df["partial_p_bh_corrected"] = p_par_adj
    summary_df["partial_significant_after_correction"] = rej_par

    summary_df.to_csv(os.path.join(OUTPUT_DIR, f"tiic_summary_{task}.csv"), index=False)
    pooled.to_csv(os.path.join(OUTPUT_DIR, f"tiic_per_tile_{task}.csv"), index=False)

    pr_id, pp_id = spearmanr(pooled["attn"], pooled["immune_density"])
    pr_ic, pp_ic = spearmanr(pooled["attn"], pooled["immune_nuclei"])
    pr_ed, pp_ed = spearmanr(pooled["attn"], pooled["epithelial_density"])
    pr_pa, pp_pa = partial_spearman(
        pooled["attn"].values, pooled["immune_density"].values, pooled["epithelial_density"].values
    )

    pooled_results = {
        "task": task,
        "coverage": "full (4x4 subcrops, 100%)",
        "model_file": ctx.config.task(task).checkpoint,
        "n_tiles_pooled": int(len(pooled)),
        "n_slides": int(len(summary_df)),
        "pooled_r_immune_density": float(pr_id),
        "pooled_p_immune_density": float(pp_id),
        "pooled_r_immune_count": float(pr_ic),
        "pooled_p_immune_count": float(pp_ic),
        "pooled_r_epithelial_density": float(pr_ed),
        "pooled_p_epithelial_density": float(pp_ed),
        "pooled_partial_r": float(pr_pa),
        "pooled_partial_p": float(pp_pa),
    }
    with open(os.path.join(OUTPUT_DIR, f"pooled_results_{task}.json"), "w") as f:
        json.dump(pooled_results, f, indent=2)

    print(f"\n{'=' * 66}\nAGGREGATE - {task}   ({len(summary_df)} slides, {len(pooled)} tiles)")
    print(f"{'=' * 66}")
    print(
        f"  raw r (attn~immune)     mean {summary_df['spearman_r_immune_density'].mean():+.3f}"
        f"  median {summary_df['spearman_r_immune_density'].median():+.3f}"
    )
    print(
        f"  raw r (attn~epithelial) mean {summary_df['spearman_r_epithelial_density'].mean():+.3f}"
    )
    print(
        f"  partial r               mean {summary_df['partial_r_immune_given_epithelial'].mean():+.3f}"
        f"  median {summary_df['partial_r_immune_given_epithelial'].median():+.3f}"
    )
    print(f"  raw significant (BH)     {int(rej_raw.sum())} / {len(summary_df)}")
    print(f"  partial significant (BH) {int(rej_par.sum())} / {len(summary_df)}")
    sig = summary_df[rej_par]
    if len(sig):
        neg = int((sig["partial_r_immune_given_epithelial"] < 0).sum())
        print(f"    of those: {neg} negative, {len(sig) - neg} positive")
    print(
        f"\n  POOLED  r(attn,immune)={pr_id:+.3f}  r(attn,epi)={pr_ed:+.3f}  "
        f"partial={pr_pa:+.3f} (p={pp_pa:.3g})"
    )
    return summary_df


def probe():
    """Check that both models load and run before starting a long job."""
    for task in TASKS:
        path = ctx.checkpoint(task)
        print(f"\n{task}: {path}  exists={path.exists()}")
        if not path.exists():
            continue
        model, _ = load_abmil_model(task)
        dummy = torch.randn(20, ctx.files.embeddings.embed_dim).to(DEVICE)
        with torch.no_grad():
            logits, attention = model(dummy)
        print(
            f"  logits shape={tuple(logits.shape)}, attention sums to {attention.sum().item():.4f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--task", choices=[*TASKS, "both"], default="ret_binary")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.probe:
        probe()
        sys.exit(0)

    tasks = ["ret_binary", "multiclass"] if args.task == "both" else [args.task]
    if not args.aggregate_only:
        print(f"Device: {DEVICE}")
        segmentor = load_segmentor()
        for task in tasks:
            run_task(task, segmentor, limit=args.limit)
    for task in tasks:
        aggregate(task)
