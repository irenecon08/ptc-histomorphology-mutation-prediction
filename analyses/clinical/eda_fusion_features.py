"""
EDA for the fusion clinical feature set, BEFORE any imputation/encoding.

Checks the 9 confirmed variables (Diagnosis Age, Sex, Race Category,
Ethnicity Category, Overall Disease Stage, T Stage, N Stage, M Stage,
Histological Subtype) on the RAW merged data, to surface what
the fusion feature preparation (ptc_mil.data.clinical) needs to handle:

  1. Missingness -- overall and split-by-split (is missingness concentrated
     in one split, e.g. the held-out EL site? that would bias imputation
     if train-only statistics don't represent val/test patterns).
  2. Rare categories in nominal variables -- one-hot columns with very few
     positive cases contribute little signal and inflate dimensionality;
     flags any category below a configurable threshold as a candidate for
     collapsing into "Other".
  3. Continuous variable (Age) distribution and outliers.
  4. Cross-split distributional consistency -- since the split is
     institutional (site-based) rather than random, clinical variable
     distributions could differ across train/val/test beyond what a random
     split would produce. This checks for that directly rather than
     assuming it away.
  5. Multicollinearity among the four staging variables (Overall/T/N/M),
     which are clinically related by construction (AJCC overall stage is
     partly derived from T/N/M) and may be redundant as separate features.

Outputs:
  - eda_summary.csv: one row per variable with missingness/cardinality stats
  - eda_report.txt: printed findings + preprocessing recommendations
  - age_distribution.png, stage_correlation.png: plots

All written to results/analyses/clinical_eda/.

Usage:
    python analyses/clinical/eda_fusion_features.py
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ptc_mil.analysis import load_context
from ptc_mil.data.clinical import (
    load_and_merge,
    map_m_stage,
    map_n_stage,
    map_overall_stage,
    map_t_stage,
)

RARE_CATEGORY_THRESHOLD = 10  # flag categories with fewer than N patients

# The same ordinal stage maps as the fusion feature preparation, used here only for
# the multicollinearity check (raw string values are used everywhere else).
_map_overall_stage = map_overall_stage
_map_t_stage = map_t_stage
_map_n_stage = map_n_stage
_map_m_stage = map_m_stage


VARIABLES = [
    ("Diagnosis Age", "Diagnosis Age", "continuous", None),
    ("Sex", "Sex", "nominal", None),
    ("Race Category", "Race Category", "nominal", None),
    ("Ethnicity Category", "Ethnicity Category", "nominal", None),
    (
        "Overall Disease Stage",
        "Neoplasm Disease Stage American Joint Committee on Cancer Code",
        "ordinal",
        _map_overall_stage,
    ),
    (
        "Tumor (T) Stage",
        "American Joint Committee on Cancer Tumor Stage Code",
        "ordinal",
        _map_t_stage,
    ),
    (
        "Lymph Node (N) Stage",
        "Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code",
        "ordinal",
        _map_n_stage,
    ),
    (
        "Metastasis (M) Stage",
        "American Joint Committee on Cancer Metastasis Stage Code",
        "ordinal",
        _map_m_stage,
    ),
    ("Histological Subtype", "Tumor Type", "nominal", None),
]


def missingness_by_split(df, col):
    rows = []
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        n = len(sub)
        n_missing = sub[col].isna().sum()
        rows.append((split, n, n_missing, 100 * n_missing / n if n else np.nan))
    return pd.DataFrame(rows, columns=["split", "n", "n_missing", "pct_missing"])


def rare_categories(df, col, threshold=RARE_CATEGORY_THRESHOLD):
    counts = df[col].value_counts(dropna=True)
    rare = counts[counts < threshold]
    return rare


def cross_split_consistency_continuous(df, col):
    """Kruskal-Wallis across splits -- flags if a continuous variable's
    distribution differs significantly by split (expected to some degree
    given the institutional split, but large differences are worth noting
    since train-only imputation/scaling statistics would then not represent
    val/test well)."""
    sub = df[[col, "split"]].dropna()
    groups = [sub.loc[sub["split"] == s, col].values for s in ["train", "val", "test"]]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return np.nan, np.nan
    stat, p = kruskal(*groups)
    return stat, p


def cross_split_consistency_categorical(df, col):
    """Chi-square across splits for a categorical variable's distribution."""
    sub = df[[col, "split"]].dropna()
    table = pd.crosstab(sub[col], sub["split"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan
    chi2, p, _, _ = chi2_contingency(table)
    return chi2, p


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    args = parser.parse_args()
    ctx = load_context()
    args.out_dir = str(ctx.output_dir("clinical_eda"))
    df = load_and_merge(
        ctx.paths.labels_csv, ctx.paths.require("clinical_tsv"), ctx.paths.splits_csv
    )
    print(f"Merged cohort: {len(df)} patients\n")

    report_lines = [f"EDA report -- fusion clinical features\nCohort: {len(df)} patients\n"]
    summary_rows = []

    for display_name, col, vtype, mapper in VARIABLES:
        report_lines.append(
            f"\n{'=' * 70}\n{display_name}  (column: {col}, type: {vtype})\n{'=' * 70}"
        )

        # --- Missingness overall + by split ---
        n_missing = df[col].isna().sum()
        pct_missing = 100 * n_missing / len(df)
        report_lines.append(f"Overall missingness: {n_missing}/{len(df)} ({pct_missing:.1f}%)")
        by_split = missingness_by_split(df, col)
        report_lines.append(by_split.to_string(index=False))

        row = {
            "variable": display_name,
            "type": vtype,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
        }

        if vtype == "continuous":
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            desc = vals.describe()
            report_lines.append(
                f"\nDistribution: mean={desc['mean']:.1f}, median={vals.median():.1f}, "
                f"std={desc['std']:.1f}, min={desc['min']:.0f}, max={desc['max']:.0f}"
            )
            z = (vals - vals.mean()) / vals.std()
            n_outliers = (z.abs() > 3).sum()
            report_lines.append(f"Outliers (|z|>3): {n_outliers} patients")
            row.update({"n_categories": None, "n_rare_categories": None, "n_outliers": n_outliers})

            stat, p = cross_split_consistency_continuous(df, col)
            report_lines.append(
                f"Cross-split consistency (Kruskal-Wallis): stat={stat:.3f}, p={p:.4f}"
                f"{'  <-- FLAG: distribution differs significantly by split' if p < 0.05 else ''}"
            )

            fig, ax = plt.subplots(figsize=(7, 4))
            for split, color in zip(["train", "val", "test"], ["#4C72B0", "#DD8452", "#55A868"]):
                sub = pd.to_numeric(df.loc[df["split"] == split, col], errors="coerce").dropna()
                ax.hist(sub, bins=20, alpha=0.5, label=split, color=color, density=True)
            ax.set_title(f"{display_name} distribution by split")
            ax.set_xlabel(display_name)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "age_distribution.png"), dpi=150)
            plt.close()

        else:  # nominal or ordinal -- treat as categorical for cardinality/rarity purposes
            counts = df[col].value_counts(dropna=True)
            report_lines.append(f"\nCategories ({len(counts)}):\n{counts.to_string()}")
            rare = rare_categories(df, col)
            row.update(
                {"n_categories": len(counts), "n_rare_categories": len(rare), "n_outliers": None}
            )
            if len(rare) > 0:
                report_lines.append(
                    f"\nRARE categories (n<{RARE_CATEGORY_THRESHOLD}): "
                    f"{dict(rare)}  <-- FLAG: candidates for collapsing into 'Other' before one-hot encoding"
                )

            stat, p = cross_split_consistency_categorical(df, col)
            if not np.isnan(p):
                report_lines.append(
                    f"Cross-split consistency (Chi-square): stat={stat:.3f}, p={p:.4f}"
                    f"{'  <-- FLAG: category distribution differs significantly by split' if p < 0.05 else ''}"
                )

        summary_rows.append(row)

    # --- Multicollinearity among the 4 staging variables ---
    report_lines.append(f"\n{'=' * 70}\nMulticollinearity check: staging variables\n{'=' * 70}")
    stage_cols = {}
    for display_name, col, vtype, mapper in VARIABLES:
        if mapper is not None:
            stage_cols[display_name] = df[col].map(mapper)
    stage_df = pd.DataFrame(stage_cols)
    corr = stage_df.corr(method="spearman")
    report_lines.append(corr.to_string())
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                high_corr_pairs.append((corr.columns[i], corr.columns[j], r))
    if high_corr_pairs:
        report_lines.append(
            "\nFLAG: highly correlated (|rho|>0.7) staging variable pairs -- "
            "consider whether all four add independent signal or if the model "
            "may end up relying on redundant inputs:"
        )
        for a, b, r in high_corr_pairs:
            report_lines.append(f"  {a} <-> {b}: rho={r:.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, label="Spearman rho")
    ax.set_title("Staging variable correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "stage_correlation.png"), dpi=150)
    plt.close()

    # --- Save outputs ---
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.out_dir, "eda_summary.csv"), index=False)

    report_text = "\n".join(report_lines)
    with open(os.path.join(args.out_dir, "eda_report.txt"), "w") as f:
        f.write(report_text)

    print(report_text)
    print(
        f"\n\nSaved: {args.out_dir}/eda_summary.csv, eda_report.txt, "
        f"age_distribution.png, stage_correlation.png"
    )


if __name__ == "__main__":
    main()
