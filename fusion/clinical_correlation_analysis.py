"""
Clinical variable <-> target correlation analysis.

Tests each clinical variable (demographics, staging, histology, molecular
burden, treatment, survival) for association with:
  - Aim (a): mutation_label (BRAF_V600E / RAS / Other) -- 3 groups
  - Aim (b): RET (0 / 1)                                -- 2 groups

Continuous / ordinal variables -> Kruskal-Wallis (Aim a) / Mann-Whitney U (Aim b)
Nominal categorical variables   -> Chi-square (Aim a and Aim b), with a
                                    Monte Carlo permutation p-value fallback
                                    when expected cell counts are too low
                                    for the chi-square approximation to be
                                    reliable (i.e. the "Fisher's exact"
                                    substitute for r x c tables, since scipy's
                                    fisher_exact only supports 2x2).

Benjamini-Hochberg correction is applied separately within each aim's list
of variables (mirroring Table 5.4.1 in the dissertation).

Usage:
    python clinical_correlation_analysis.py \
        --labels final_labels.csv \
        --clinical thca_tcga_pan_can_atlas_2018_clinical_data.tsv \
        --out correlation_results.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests

RNG = np.random.default_rng(0)
N_PERMUTATIONS = 10000

# ---------------------------------------------------------------------------
# Ordinal staging variables are stored as AJCC codes (e.g. "STAGE IVA",
# "T4A", "N1B", "MX"), not plain numbers -- pd.to_numeric() would silently
# turn every one of these into NaN. Each is mapped here to an ordinal rank
# by taking its major stage (sub-stage letters A/B/C collapsed into the
# parent stage, since sub-staging is not consistently ordered across all
# codes). Indeterminate codes (TX, NX, "STAGE X"-type) have no ordinal
# meaning and are mapped to NaN, dropping those patients from this
# variable's test only (not from the cohort as a whole).
# ---------------------------------------------------------------------------


def _map_overall_stage(v):
    if pd.isna(v):
        return np.nan
    v = str(v).upper()
    mapping = {"STAGE I": 1, "STAGE II": 2, "STAGE III": 3,
               "STAGE IVA": 4, "STAGE IVB": 4, "STAGE IVC": 4, "STAGE IV": 4}
    return mapping.get(v, np.nan)


def _map_t_stage(v):
    if pd.isna(v):
        return np.nan
    v = str(v).upper()
    if v.startswith("TX"):
        return np.nan
    if v.startswith("T1"):
        return 1
    if v.startswith("T2"):
        return 2
    if v.startswith("T3"):
        return 3
    if v.startswith("T4"):
        return 4
    return np.nan


def _map_n_stage(v):
    if pd.isna(v):
        return np.nan
    v = str(v).upper()
    if v.startswith("NX"):
        return np.nan
    if v.startswith("N0"):
        return 0
    if v.startswith("N1"):
        return 1
    return np.nan


def _map_m_stage(v):
    if pd.isna(v):
        return np.nan
    v = str(v).upper()
    if v.startswith("MX"):
        return np.nan
    if v.startswith("M0"):
        return 0
    if v.startswith("M1"):
        return 1
    return np.nan


ORDINAL_MAPS = {
    "Neoplasm Disease Stage American Joint Committee on Cancer Code": _map_overall_stage,
    "American Joint Committee on Cancer Tumor Stage Code": _map_t_stage,
    "Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code": _map_n_stage,
    "American Joint Committee on Cancer Metastasis Stage Code": _map_m_stage,
}

# ---------------------------------------------------------------------------
# Variable specification: (display name, actual column in clinical file, type)
# type is one of: "continuous", "ordinal", "nominal"
# ---------------------------------------------------------------------------
VARIABLES = [
    ("Diagnosis Age", "Diagnosis Age", "continuous"),
    ("Sex", "Sex", "nominal"),
    ("Race Category", "Race Category", "nominal"),
    ("Ethnicity Category", "Ethnicity Category", "nominal"),
    ("Overall Disease Stage", "Neoplasm Disease Stage American Joint Committee on Cancer Code", "ordinal"),
    ("Tumor (T) Stage", "American Joint Committee on Cancer Tumor Stage Code", "ordinal"),
    ("Lymph Node (N) Stage", "Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code", "ordinal"),
    ("Metastasis (M) Stage", "American Joint Committee on Cancer Metastasis Stage Code", "ordinal"),
    # NOTE: "Cancer Type Detailed" is constant ("Papillary Thyroid Cancer")
    # for all patients in this download and carries no information.
    # The actual PTC histological subtype (classical/follicular/tall cell)
    # lives in the "Tumor Type" column instead -- confirmed by direct
    # inspection of the file, see chat.
    ("Histological Subtype", "Tumor Type", "nominal"),
    ("Person Neoplasm Cancer Status", "Person Neoplasm Cancer Status", "nominal"),
    ("Radiation Therapy", "Radiation Therapy", "nominal"),
    ("Mutation Count", "Mutation Count", "continuous"),
    ("Fraction Genome Altered", "Fraction Genome Altered", "continuous"),
    ("MSI MANTIS Score", "MSI MANTIS Score", "continuous"),
    ("MSIsensor Score", "MSIsensor Score", "continuous"),
    ("TMB (nonsynonymous)", "TMB (nonsynonymous)", "continuous"),
    ("Aneuploidy Score", "Aneuploidy Score", "continuous"),
    ("Tumor Break Load", "Tumor Break Load", "continuous"),
    ("Buffa Hypoxia Score", "Buffa Hypoxia Score", "continuous"),
    ("Ragnum Hypoxia Score", "Ragnum Hypoxia Score", "continuous"),
    ("Winter Hypoxia Score", "Winter Hypoxia Score", "continuous"),
    ("Overall Survival (Months)", "Overall Survival (Months)", "continuous"),
    ("Disease Free (Months)", "Disease Free (Months)", "continuous"),
    ("Progress Free Survival (Months)", "Progress Free Survival (Months)", "continuous"),
    ("Disease-Specific Survival (Months)", "Months of disease-specific survival", "continuous"),
    # NOTE: "Neoplasm Histologic Grade" deliberately excluded -- 100% missing.
]


def load_data(labels_path, clinical_path):
    labels = pd.read_csv(labels_path)
    clinical = pd.read_csv(clinical_path, sep="\t")

    # Some rows use the literal string "NA" rather than a true NaN -- normalise.
    clinical = clinical.replace("NA", np.nan)

    # A small number of patients have more than one sample row (e.g. a
    # recurrence or additional sample). Keep only the primary tumor sample
    # per patient so the merge below is one-row-per-patient.
    if "Sample Type" in clinical.columns:
        clinical = clinical.sort_values("Sample Type", key=lambda s: s != "Primary")
    clinical = clinical.drop_duplicates(subset="Patient ID", keep="first")

    # Left join on the modelling cohort (final_labels.csv) so every patient
    # in the cohort is retained even if clinical data is missing for them --
    # those rows simply carry NaN across all clinical columns.
    merged = labels.merge(
        clinical, left_on="patient", right_on="Patient ID", how="left"
    )
    return merged


def to_numeric(series, col_name):
    """Convert a column to numeric. Staging columns need their AJCC codes
    mapped to ordinal ranks first (see ORDINAL_MAPS); everything else is
    assumed already numeric-like and goes through pd.to_numeric directly."""
    if col_name in ORDINAL_MAPS:
        return series.map(ORDINAL_MAPS[col_name])
    return pd.to_numeric(series, errors="coerce")


def test_continuous(df, col, group_col, groups):
    """Kruskal-Wallis (>2 groups) or Mann-Whitney U (2 groups) on a
    continuous/ordinal variable, dropping missing values pairwise."""
    sub = df[[col, group_col]].copy()
    sub[col] = to_numeric(sub[col], col)
    sub = sub.dropna(subset=[col, group_col])

    samples = [sub.loc[sub[group_col] == g, col].values for g in groups]
    ns = [len(s) for s in samples]
    if any(n < 2 for n in ns):
        return dict(test="insufficient_data", statistic=np.nan, p=np.nan, n_per_group=ns)

    if len(groups) == 2:
        stat, p = mannwhitneyu(samples[0], samples[1], alternative="two-sided")
        test_name = "Mann-Whitney U"
    else:
        stat, p = kruskal(*samples)
        test_name = "Kruskal-Wallis"

    return dict(test=test_name, statistic=stat, p=p, n_per_group=ns)


def test_categorical(df, col, group_col, groups):
    """Chi-square test of independence on a nominal categorical variable.
    Falls back to a Monte Carlo permutation p-value (a practical r x c
    substitute for Fisher's exact test) when >20% of expected cell counts
    are below 5, the standard threshold for the chi-square approximation."""
    sub = df[[col, group_col]].copy()
    sub = sub.dropna(subset=[col, group_col])
    sub = sub[sub[group_col].isin(groups)]

    table = pd.crosstab(sub[col], sub[group_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return dict(test="insufficient_categories", statistic=np.nan, p=np.nan,
                     n_per_group=[int((sub[group_col] == g).sum()) for g in groups])

    chi2, p, dof, expected = chi2_contingency(table)
    low_expected_frac = (expected < 5).mean()

    if low_expected_frac > 0.20:
        # Monte Carlo permutation test: shuffle group labels, recompute chi2,
        # estimate p as the fraction of permuted statistics >= observed.
        observed_stat = chi2
        perm_stats = np.empty(N_PERMUTATIONS)
        group_labels = np.array(sub[group_col].values, dtype=object)
        values = np.array(sub[col].values, dtype=object)
        for i in range(N_PERMUTATIONS):
            RNG.shuffle(group_labels)
            perm_table = pd.crosstab(values, group_labels)
            perm_chi2, _, _, _ = chi2_contingency(perm_table)
            perm_stats[i] = perm_chi2
        p = (np.sum(perm_stats >= observed_stat) + 1) / (N_PERMUTATIONS + 1)
        test_name = "Chi-square (Monte Carlo p, low expected counts)"
    else:
        test_name = "Chi-square"

    return dict(test=test_name, statistic=chi2, p=p,
                n_per_group=[int((sub[group_col] == g).sum()) for g in groups])


def run_analysis(df):
    results = []

    for aim, group_col, groups in [
        ("Aim a (3-class)", "mutation_label", ["BRAF_V600E", "RAS", "Other"]),
        ("Aim b (RET binary)", "RET", [0, 1]),
    ]:
        for display_name, col, vtype in VARIABLES:
            if col not in df.columns:
                results.append(dict(aim=aim, variable=display_name, column=col,
                                     type=vtype, test="column_not_found",
                                     statistic=np.nan, p=np.nan, n_per_group=None))
                continue

            if vtype in ("continuous", "ordinal"):
                r = test_continuous(df, col, group_col, groups)
            else:
                r = test_categorical(df, col, group_col, groups)

            results.append(dict(aim=aim, variable=display_name, column=col,
                                 type=vtype, **r))

    res_df = pd.DataFrame(results)

    # BH correction applied separately within each aim, over only the
    # variables that produced a valid p-value.
    res_df["p_bh"] = np.nan
    for aim in res_df["aim"].unique():
        mask = (res_df["aim"] == aim) & res_df["p"].notna()
        if mask.sum() == 0:
            continue
        _, qvals, _, _ = multipletests(res_df.loc[mask, "p"], method="fdr_bh")
        res_df.loc[mask, "p_bh"] = qvals

    res_df["significant_bh_0.05"] = res_df["p_bh"] < 0.05

    col_order = ["aim", "variable", "column", "type", "test",
                 "n_per_group", "statistic", "p", "p_bh", "significant_bh_0.05"]
    res_df = res_df[col_order]
    return res_df.sort_values(["aim", "p_bh"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--clinical", required=True)
    parser.add_argument("--out", default="correlation_results.csv")
    args = parser.parse_args()

    df = load_data(args.labels, args.clinical)
    print(f"Merged cohort: {len(df)} patients "
          f"({df['Patient ID'].notna().sum()} with clinical data)")

    results = run_analysis(df)
    results.to_csv(args.out, index=False)
    print(f"\nSaved results to {args.out}")
    print(f"\n{results[['aim','variable','type','test','p','p_bh','significant_bh_0.05']].to_string(index=False)}")


if __name__ == "__main__":
    main()
