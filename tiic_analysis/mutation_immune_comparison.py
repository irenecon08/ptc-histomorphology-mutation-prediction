"""
mutation_immune_comparison.py

Tests whether PTC driver genotypes (BRAF V600E / RAS / RET fusion / driver-negative)
differ in molecular immune infiltration, using the Thorsson et al. 2018 PanImmune
feature matrix. Entirely independent of the imaging pipeline - no tiles, no model,
no HoVer-Net, no GPU.

USAGE
-----
Step 1, inspect the Thorsson file and check merge coverage:
    python mutation_immune_comparison.py --inspect

    This prints the available columns, fuzzy-matches the ones we need, filters to
    THCA, and reports how many of your labelled patients are actually present.
    Edit the CONFIG block below to match the real column names, then:

Step 2, run the analysis:
    python mutation_immune_comparison.py

ENVIRONMENT
-----------
Needs pandas / scipy / statsmodels / matplotlib. No GPU required, but venv310 lives
on the project store and was built for Rocky 9, so run this on a GPU machine (any
one, idle is fine) rather than knuckles.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu, rankdata, chi2_contingency
from statsmodels.stats.multitest import multipletests

# ==========================================================================
# CONFIG  -- edit after running --inspect
# ==========================================================================
PROJ = "/cs/student/project_msc/2025/aibh/iconstan/"
THORSSON_PATH = PROJ + "thorsson_table_s1.csv"
LABELS_PATH = PROJ + "final_labels.csv"
OUTPUT_DIR = PROJ + "immune_molecular_analysis/"

# Column names in the Thorsson file. Run --inspect to confirm these.
COL_BARCODE = "TCGA Participant Barcode"
COL_STUDY = "TCGA Study"
STUDY_VALUE = "THCA"

# Primary outcome: methylation-derived, direct analogue of the HoVer-Net measure.
COL_PRIMARY = "Leukocyte Fraction"

# Pre-specified secondary outcomes (expression-derived signature scores).
COLS_SECONDARY = [
    "Lymphocyte Infiltration Signature Score",
    "Macrophage Regulation",
    "IFN-gamma Response",
    "TGF-beta Response",
    "Wound Healing",
]

# For the composition adjustment.
COL_STROMAL = "Stromal Fraction"
COL_PURITY = None          # set if a purity column exists, else leave None

# Categorical immune subtype (C1-C6).
COL_SUBTYPE = "Immune Subtype"

# Columns in your labels file.
LAB_PATIENT = "patient"
LAB_MUTATION = "mutation_label"
LAB_RET = "RET"

ALPHA = 0.05
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================================
# Helpers
# ==========================================================================
def read_any(path):
    """Read csv / tsv / xlsx without caring which."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: file not found: {path}")
    if path.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(path)
        except ImportError:
            sys.exit("ERROR: reading xlsx needs openpyxl. Convert to CSV on your "
                     "laptop and re-transfer instead.")
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep, comment="#", low_memory=False)


def fuzzy_find(cols, *keywords):
    """Return columns whose name contains all keywords (case-insensitive)."""
    out = []
    for c in cols:
        lc = str(c).lower()
        if all(k.lower() in lc for k in keywords):
            out.append(c)
    return out


def epsilon_squared(H, n, k):
    """Effect size for Kruskal-Wallis. Roughly: 0.01 small, 0.08 medium, 0.26 large."""
    if n - k <= 0:
        return float("nan")
    return (H - k + 1) / (n - k)


def rank_biserial(x, y):
    """Effect size for Mann-Whitney. -1 to +1; sign gives direction."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    U = mannwhitneyu(x, y, alternative="two-sided").statistic
    return 1 - (2 * U) / (nx * ny)


def residualise(target, control):
    """
    Rank-based residualisation, mirroring the partial_spearman approach used in
    the TIIC analysis: removes the linear-in-ranks contribution of `control`
    from `target`. Used here to adjust immune measures for stromal/purity
    composition, which is the group-level analogue of adjusting tile-level
    immune density for epithelial density.
    """
    t = rankdata(target)
    c = rankdata(control)
    c_centred = c - c.mean()
    denom = (c_centred ** 2).sum()
    if denom == 0:
        return t - t.mean()
    beta = (c_centred * (t - t.mean())).sum() / denom
    return (t - t.mean()) - beta * c_centred


def build_genotype(df):
    """
    Four mutually exclusive groups. Necessary because RET+ patients currently sit
    inside 'Other' (mutation_label == 'Other' AND RET == 1), so the raw
    mutation_label groups overlap with the RET grouping.
    """
    g = pd.Series(index=df.index, dtype=object)
    g[df[LAB_MUTATION] == "BRAF_V600E"] = "BRAF"
    g[df[LAB_MUTATION] == "RAS"] = "RAS"
    g[(df[LAB_MUTATION] == "Other") & (df[LAB_RET] == 1)] = "RET"
    g[(df[LAB_MUTATION] == "Other") & (df[LAB_RET] != 1)] = "DriverNeg"
    return g


# ==========================================================================
# Inspection mode
# ==========================================================================
def inspect():
    print("=" * 70)
    print("INSPECTING THORSSON FILE")
    print("=" * 70)
    t = read_any(THORSSON_PATH)
    print(f"\nShape: {t.shape[0]} rows x {t.shape[1]} columns\n")

    print("--- all columns ---")
    for c in t.columns:
        print(f"  {c}")

    print("\n--- fuzzy matches for what we need ---")
    checks = {
        "barcode":    fuzzy_find(t.columns, "barcode") or fuzzy_find(t.columns, "participant"),
        "study":      fuzzy_find(t.columns, "study") or fuzzy_find(t.columns, "cancer", "type"),
        "leukocyte":  fuzzy_find(t.columns, "leukocyte"),
        "lymphocyte": fuzzy_find(t.columns, "lymphocyte"),
        "macrophage": fuzzy_find(t.columns, "macrophage"),
        "ifn":        fuzzy_find(t.columns, "ifn"),
        "tgf":        fuzzy_find(t.columns, "tgf"),
        "wound":      fuzzy_find(t.columns, "wound"),
        "stromal":    fuzzy_find(t.columns, "stromal"),
        "purity":     fuzzy_find(t.columns, "purity"),
        "subtype":    fuzzy_find(t.columns, "immune", "subtype"),
    }
    for k, v in checks.items():
        print(f"  {k:12s}: {v if v else 'NOT FOUND'}")

    # Filter to THCA if we can work out how
    study_col = COL_STUDY if COL_STUDY in t.columns else (checks["study"][0] if checks["study"] else None)
    if study_col:
        vals = t[study_col].astype(str).unique()
        print(f"\n  '{study_col}' has {len(vals)} distinct values; "
              f"THCA present: {'THCA' in vals}")
        thca = t[t[study_col].astype(str).str.upper() == STUDY_VALUE]
        print(f"  THCA rows: {len(thca)}")
    else:
        print("\n  Could not identify a study column - check the list above.")
        thca = t

    # Merge coverage
    bc_col = COL_BARCODE if COL_BARCODE in t.columns else (checks["barcode"][0] if checks["barcode"] else None)
    if bc_col and os.path.exists(LABELS_PATH):
        lab = read_any(LABELS_PATH)
        print(f"\n--- merge coverage ---")
        print(f"  labels file: {len(lab)} patients")
        thca = thca.copy()
        thca["_bc"] = thca[bc_col].astype(str).str.strip().str[:12]
        matched = lab[LAB_PATIENT].astype(str).str[:12].isin(set(thca["_bc"]))
        print(f"  matched: {matched.sum()} / {len(lab)}")
        if matched.sum() < len(lab):
            missing = lab[~matched]
            print(f"  unmatched by genotype:")
            print(missing.groupby(LAB_MUTATION).size().to_string())
            print("  (check whether missingness skews by genotype)")

    print("\nEdit the CONFIG block to match the real names, then run without --inspect.")


# ==========================================================================
# Analysis
# ==========================================================================
def load_merged():
    t = read_any(THORSSON_PATH)
    lab = read_any(LABELS_PATH)

    for c in (COL_BARCODE, COL_STUDY):
        if c not in t.columns:
            sys.exit(f"ERROR: column '{c}' not in Thorsson file. Run --inspect.")

    t = t[t[COL_STUDY].astype(str).str.upper() == STUDY_VALUE].copy()
    t["_bc"] = t[COL_BARCODE].astype(str).str.strip().str[:12]
    lab = lab.copy()
    lab["_bc"] = lab[LAB_PATIENT].astype(str).str.strip().str[:12]

    m = lab.merge(t, on="_bc", how="inner", suffixes=("", "_thor"))
    m["genotype"] = build_genotype(m)
    m = m[m["genotype"].notna()]

    print(f"Labelled patients: {len(lab)}")
    print(f"Matched to Thorsson THCA: {len(m)}")
    print("\nGroup sizes:")
    print(m["genotype"].value_counts().to_string())
    return m


def describe_features(m, features):
    """Check distributions before interpreting anything - signature scores may be
    pan-cancer standardised, which compresses within-THCA variance."""
    print("\n" + "=" * 70)
    print("FEATURE DISTRIBUTIONS WITHIN THCA")
    print("=" * 70)
    present = [f for f in features if f in m.columns]
    if present:
        print(m[present].describe().T.to_string())
    missing = [f for f in features if f not in m.columns]
    if missing:
        print(f"\nNOT PRESENT (skipped): {missing}")
    return present


def four_group_test(m, feature, label=""):
    groups, names = [], []
    for g in ["BRAF", "RAS", "RET", "DriverNeg"]:
        v = m.loc[m["genotype"] == g, feature].dropna().values
        if len(v) >= 3:
            groups.append(v)
            names.append(g)
    if len(groups) < 2:
        return None

    H, p = kruskal(*groups)
    n = sum(len(g) for g in groups)
    eps2 = epsilon_squared(H, n, len(groups))

    print(f"\n  {label or feature}")
    for nm, g in zip(names, groups):
        print(f"    {nm:10s} n={len(g):3d}  median={np.median(g):.4f}  mean={g.mean():.4f}")
    print(f"    Kruskal-Wallis: H={H:.3f}, p={p:.4g}, epsilon^2={eps2:.4f}")

    # Pairwise post-hoc (Mann-Whitney + BH), only if the omnibus is suggestive
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
            print("    pairwise (BH-adjusted):")
            for pr, pa, ef in zip(pairs, padj, effects):
                flag = " *" if pa < ALPHA else ""
                print(f"      {pr:22s} p={pa:.4f}  rank-biserial={ef:+.3f}{flag}")

    return {"feature": feature, "H": H, "p": p, "epsilon2": eps2, "n": n}


def ret_binary_test(m, feature):
    pos = m.loc[m["genotype"] == "RET", feature].dropna().values
    neg = m.loc[m["genotype"] != "RET", feature].dropna().values
    if len(pos) < 3 or len(neg) < 3:
        return None
    U, p = mannwhitneyu(pos, neg, alternative="two-sided")
    rb = rank_biserial(pos, neg)
    print(f"\n  RET+ (n={len(pos)}) vs all others (n={len(neg)}) on {feature}")
    print(f"    median RET+ = {np.median(pos):.4f}, others = {np.median(neg):.4f}")
    print(f"    Mann-Whitney U={U:.1f}, p={p:.4g}, rank-biserial={rb:+.3f}")
    if p > ALPHA:
        print(f"    NOTE: n={len(pos)} detects only moderate-to-large effects. "
              f"A null here means 'no large difference', not 'no difference'.")
    return {"feature": feature, "p": p, "rank_biserial": rb}


def subtype_test(m):
    if COL_SUBTYPE not in m.columns:
        print("\n  (immune subtype column not present, skipped)")
        return
    print("\n" + "=" * 70)
    print("IMMUNE SUBTYPE (C1-C6) BY GENOTYPE")
    print("=" * 70)
    ct = pd.crosstab(m["genotype"], m[COL_SUBTYPE])
    print(ct.to_string())
    conc = ct.sum(axis=0) / ct.values.sum()
    top = conc.max()
    print(f"\n  Most common subtype accounts for {100*top:.1f}% of THCA cases.")
    if top > 0.80:
        print("  WARNING: subtypes are heavily concentrated - this comparison has "
              "little discriminating power in THCA. Report descriptively only.")
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, p, dof, exp = chi2_contingency(ct)
        small = (exp < 5).sum()
        print(f"  Chi-square: chi2={chi2:.2f}, dof={dof}, p={p:.4g}")
        if small:
            print(f"  WARNING: {small} cells have expected count < 5 - chi-square "
                  f"unreliable, treat as descriptive.")


def adjusted_analysis(m, feature):
    """Repeat the group comparison after removing composition effects."""
    ctrl = COL_PURITY if (COL_PURITY and COL_PURITY in m.columns) else COL_STROMAL
    if ctrl not in m.columns:
        print(f"\n  (no '{COL_STROMAL}' column, composition adjustment skipped)")
        return
    sub = m[[feature, ctrl, "genotype"]].dropna()
    if len(sub) < 20:
        return
    sub = sub.copy()
    sub[feature + "_adj"] = residualise(sub[feature].values, sub[ctrl].values)
    print(f"\n  --- adjusted for {ctrl} ---")
    four_group_test(sub, feature + "_adj", label=f"{feature} | {ctrl}")


def make_plots(m, features):
    order = ["BRAF", "RAS", "RET", "DriverNeg"]
    present = [f for f in features if f in m.columns]
    if not present:
        return
    ncol = min(3, len(present))
    nrow = int(np.ceil(len(present) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    for ax, feat in zip(axes.ravel(), present):
        data = [m.loc[m["genotype"] == g, feat].dropna().values for g in order]
        keep = [(d, g) for d, g in zip(data, order) if len(d) > 0]
        ax.boxplot([d for d, _ in keep], labels=[g for _, g in keep], showfliers=False)
        for i, (d, _) in enumerate(keep):
            jitter = np.random.normal(i + 1, 0.05, len(d))
            ax.scatter(jitter, d, s=6, alpha=0.35)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    for ax in axes.ravel()[len(present):]:
        ax.axis("off")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "immune_by_genotype.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved plot: {out}")


def main():
    m = load_merged()

    all_feats = [COL_PRIMARY] + COLS_SECONDARY
    present = describe_features(m, all_feats)
    if COL_PRIMARY not in present:
        sys.exit(f"ERROR: primary outcome '{COL_PRIMARY}' not found. Run --inspect.")

    print("\n" + "=" * 70)
    print(f"PRIMARY OUTCOME: {COL_PRIMARY}")
    print("=" * 70)
    primary = four_group_test(m, COL_PRIMARY)
    ret_binary_test(m, COL_PRIMARY)
    adjusted_analysis(m, COL_PRIMARY)

    secondary = [f for f in COLS_SECONDARY if f in present]
    if secondary:
        print("\n" + "=" * 70)
        print("SECONDARY OUTCOMES")
        print("=" * 70)
        results = [r for r in (four_group_test(m, f) for f in secondary) if r]
        if results:
            praw = [r["p"] for r in results]
            _, padj, _, _ = multipletests(praw, alpha=ALPHA, method="fdr_bh")
            print("\n  BH-adjusted across secondary features:")
            for r, pa in zip(results, padj):
                flag = " *" if pa < ALPHA else ""
                print(f"    {r['feature']:45s} p={r['p']:.4g} -> p_adj={pa:.4f}"
                      f"  eps2={r['epsilon2']:.4f}{flag}")

    subtype_test(m)
    make_plots(m, all_feats)

    m.to_csv(os.path.join(OUTPUT_DIR, "merged_immune_genotype.csv"), index=False)
    print(f"\nSaved merged data: {OUTPUT_DIR}merged_immune_genotype.csv")
    print("\nReminder: this is observational. Association, not causation. "
          "Check thyroiditis and histological variant as confounders before "
          "interpreting any positive finding.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true",
                    help="Print columns, fuzzy-match, and check merge coverage. Run this first.")
    args = ap.parse_args()
    inspect() if args.inspect else main()
