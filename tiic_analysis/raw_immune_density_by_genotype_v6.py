"""
raw_immune_density_by_genotype.py

The matched, apples-to-apples counterpart to mutation_immune_comparison.py.

That script tested: do genotypes differ in MOLECULAR immune infiltration
(Thorsson Leukocyte Fraction), across 489 patients?

This script tests the SAME question using RAW per-slide immune density from
your own HoVer-Net output, across the 80 test slides — no attention, no
epithelial-density adjustment, no residual. Just mean immune density per
slide, grouped by genotype, same Kruskal-Wallis test.

This is the fair comparison: same construct (immune infiltration), same
statistical test, two measurement modalities. It is NOT a replacement for
either the molecular analysis or the earlier residual-by-genotype check —
both of those answer different, still-valid questions and should be kept.

Usage (CPU only, no GPU needed - can run on knuckles or a GPU machine):
    python raw_immune_density_by_genotype.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = "/cs/student/project_msc/2025/aibh/iconstan/"
TIIC_DIR = PROJ + "tiic_analysis_v6/"
LABELS_PATH = PROJ + "final_labels.csv"
OUTPUT_DIR = PROJ + "immune_molecular_analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA = 0.05


def epsilon_squared(H, n, k):
    if n - k <= 0:
        return float("nan")
    return (H - k + 1) / (n - k)


def rank_biserial(x, y):
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    U = mannwhitneyu(x, y, alternative="two-sided").statistic
    return 1 - (2 * U) / (nx * ny)


def build_genotype(df, mutation_col, ret_col):
    g = pd.Series(index=df.index, dtype=object)
    g[df[mutation_col] == "BRAF_V600E"] = "BRAF"
    g[df[mutation_col] == "RAS"] = "RAS"
    g[(df[mutation_col] == "Other") & (df[ret_col] == 1)] = "RET"
    g[(df[mutation_col] == "Other") & (df[ret_col] != 1)] = "DriverNeg"
    return g


def four_group_test(df, feature, label=""):
    groups, names = [], []
    for gname in ["BRAF", "RAS", "RET", "DriverNeg"]:
        v = df.loc[df["genotype"] == gname, feature].dropna().values
        if len(v) >= 2:
            groups.append(v)
            names.append(gname)
    if len(groups) < 2:
        print(f"  Not enough groups with data for {label or feature}.")
        return None

    H, p = kruskal(*groups)
    n = sum(len(g) for g in groups)
    eps2 = epsilon_squared(H, n, len(groups))

    print(f"\n  {label or feature}")
    for nm, g in zip(names, groups):
        print(f"    {nm:10s} n={len(g):3d}  median={np.median(g):.4f}  mean={g.mean():.4f}")
    print(f"    Kruskal-Wallis: H={H:.3f}, p={p:.4g}, epsilon^2={eps2:.4f}")

    pairs, praw, effects = [], [], []
    if p < 0.10:
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                _, pp = mannwhitneyu(groups[i], groups[j], alternative="two-sided")
                pairs.append(f"{names[i]} vs {names[j]}")
                praw.append(pp)
                effects.append(rank_biserial(groups[i], groups[j]))
        if praw:
            _, padj, _, _ = multipletests(praw, alpha=ALPHA, method="fdr_bh")
            print("    pairwise (BH-adjusted) — CAUTION: small n per group, exploratory only:")
            for pr, pa, ef in zip(pairs, padj, effects):
                flag = " *" if pa < ALPHA else ""
                print(f"      {pr:22s} p={pa:.4f}  rank-biserial={ef:+.3f}{flag}")
    return {"feature": feature, "H": H, "p": p, "epsilon2": eps2, "n": n,
            "group_sizes": {nm: len(g) for nm, g in zip(names, groups)}}


def main():
    lab = pd.read_csv(LABELS_PATH)

    print("=" * 70)
    print("RAW PER-SLIDE IMMUNE DENSITY BY GENOTYPE")
    print("(matched comparison to the molecular Leukocyte Fraction test)")
    print("=" * 70)

    all_results = {}
    for task in ["multiclass", "ret_binary"]:
        tile_path = os.path.join(TIIC_DIR, f"tiic_per_tile_{task}.csv")
        if not os.path.exists(tile_path):
            print(f"\n  {task}: no per-tile file found at {tile_path}, skipping.")
            continue

        tiles = pd.read_csv(tile_path)
        # Aggregate to one raw immune density value per slide (no attention weighting,
        # no epithelial adjustment - the direct analogue of Leukocyte Fraction).
        per_slide = (
            tiles.groupby("patient")["immune_density"]
            .mean()
            .reset_index()
            .rename(columns={"immune_density": "raw_immune_density"})
        )
        n_slides = len(per_slide)

        m = per_slide.merge(lab, left_on="patient", right_on="patient", how="left")
        m["genotype"] = build_genotype(m, "mutation_label", "RET")
        m = m[m["genotype"].notna()]
        dropped = per_slide[~per_slide["patient"].isin(m["patient"])]
        print("DROPPED PATIENTS:", dropped["patient"].tolist())
        print(f"\n--- {task}: {n_slides} slides aggregated, {len(m)} matched to a genotype ---")
        print(m["genotype"].value_counts().to_string())

        res = four_group_test(m, "raw_immune_density", label=f"{task}: raw immune density by genotype")
        all_results[task] = (m, res)

        # RET binary framing too, for direct comparability with the earlier RET+ vs RET- molecular test
        pos = m.loc[m["genotype"] == "RET", "raw_immune_density"].dropna().values
        neg = m.loc[m["genotype"] != "RET", "raw_immune_density"].dropna().values
        if len(pos) >= 2 and len(neg) >= 2:
            U, p = mannwhitneyu(pos, neg, alternative="two-sided")
            rb = rank_biserial(pos, neg)
            print(f"\n  RET+ (n={len(pos)}) vs others (n={len(neg)}) on raw immune density")
            print(f"    median RET+ = {np.median(pos):.4f}, others = {np.median(neg):.4f}")
            print(f"    Mann-Whitney U={U:.1f}, p={p:.4g}, rank-biserial={rb:+.3f}")
            if len(pos) < 10:
                print(f"    NOTE: n={len(pos)} - underpowered, treat as descriptive.")

    # Combined plot across both tasks' slide pools if both exist
    if all_results:
        fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5), squeeze=False)
        order = ["BRAF", "RAS", "RET", "DriverNeg"]
        for ax, (task, (m, res)) in zip(axes[0], all_results.items()):
            data = [m.loc[m["genotype"] == g, "raw_immune_density"].dropna().values for g in order]
            keep = [(d, g) for d, g in zip(data, order) if len(d) > 0]
            ax.boxplot([d for d, _ in keep], tick_labels=[g for _, g in keep], showfliers=False)
            for i, (d, _) in enumerate(keep):
                jitter = np.random.normal(i + 1, 0.05, len(d))
                ax.scatter(jitter, d, s=20, alpha=0.6)
            ax.set_title(f"{task}: raw immune density by genotype", fontsize=10)
            ax.set_ylabel("Mean immune density (per slide)")
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, "raw_immune_density_by_genotype.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"\nSaved plot: {out}")

    print("\n" + "=" * 70)
    print("HOW TO READ THIS AGAINST THE MOLECULAR RESULT (489 patients)")
    print("=" * 70)
    print("If this comes out significant with a similar direction (RAS lowest),")
    print("that is convergent evidence: your own images independently detect the")
    print("same genotype-immune pattern the molecular data found.")
    print("If it does not reach significance, note the much smaller n here (~80")
    print("slides, fewer still per genotype group, esp. RAS) before concluding")
    print("the effect is invisible in morphology - the two tests do not have")
    print("comparable power, only a comparable design.")


if __name__ == "__main__":
    main()
