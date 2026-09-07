"""
Fusion clinical feature prep (v2 -- full feature set).

Per Petru's guidance (see chat): use the full clinical variable set for BOTH
models (no significance pre-filtering -- "your models should be able to
pick up the ones that matter"), EXCLUDING:
  - molecular/genomic burden variables (Mutation Count, TMB, Aneuploidy
    Score, Fraction Genome Altered, Tumor Break Load) -- "might be
    confounding factors"
  - hypoxia scores (Buffa, Ragnum) -- molecular-test-derived
  - survival/outcome and treatment variables (survival months, Radiation
    Therapy) -- not available at the point we'd want to predict mutation
    status in practice
  - Person Neoplasm Cancer Status -- same temporal-availability issue as
    survival (records tumour status AT LAST FOLLOW-UP, i.e. post-treatment),
    excluded by extension of Petru's stated reasoning even though not
    explicitly named

Final feature set (9 variables, SAME set used for both Aim a and Aim b):
    Diagnosis Age, Sex, Race Category, Ethnicity Category,
    Overall Disease Stage, T Stage, N Stage, M Stage, Histological Subtype

All imputation statistics (train-set median/mode) and the Age mean/std used
for z-scoring are computed on the TRAIN split ONLY, then applied unchanged
to val/test -- same leakage-avoidance principle as v1 of this script and as
your train-only RET threshold calibration.

A missingness indicator column is added per variable (1 = value was
missing and got imputed, 0 = observed).

Usage:
    python fusion_clinical_prep.py \
        --labels final_labels.csv \
        --clinical thca_tcga_pan_can_atlas_2018_clinical_data.tsv \
        --splits splits.csv \
        --out clinical_features.csv
"""

import argparse
import numpy as np
import pandas as pd

# Same ordinal maps as clinical_correlation_analysis.py / fusion_clinical_prep.py v1.

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


def load_and_merge(labels_path, clinical_path, splits_path):
    labels = pd.read_csv(labels_path)
    clinical = pd.read_csv(clinical_path, sep="\t").replace("NA", np.nan)
    if "Sample Type" in clinical.columns:
        clinical = clinical.sort_values("Sample Type", key=lambda s: s != "Primary")
    clinical = clinical.drop_duplicates(subset="Patient ID", keep="first")
    splits = pd.read_csv(splits_path)

    df = labels.merge(clinical, left_on="patient", right_on="Patient ID", how="left")
    df = df.merge(splits[["patient", "split"]], on="patient", how="left")
    return df


def _onehot_with_missingness(df, col_actual, prefix, train_mask, out):
    """Shared helper: impute a nominal column with the TRAIN-set mode, add
    a missingness flag, then one-hot encode."""
    series = df[col_actual]
    train_mode = series[train_mask].mode().iloc[0]
    out[f"{prefix}_missing"] = series.isna().astype(int)
    imputed = series.fillna(train_mode)
    dummies = pd.get_dummies(imputed, prefix=prefix)
    return pd.concat([out, dummies], axis=1)


def build_features(df, train_split_label="train"):
    train_mask = df["split"] == train_split_label
    out = pd.DataFrame({"patient": df["patient"], "split": df["split"]})

    # --- Diagnosis Age: continuous, z-scored on TRAIN mean/std ---
    age = pd.to_numeric(df["Diagnosis Age"], errors="coerce")
    train_mean, train_std = age[train_mask].mean(), age[train_mask].std()
    out["age_missing"] = age.isna().astype(int)
    out["age_z"] = (age.fillna(train_mean) - train_mean) / train_std

    # --- Overall / T / N / M Stage: ordinal maps, TRAIN-mode imputation ---
    for label, col, mapper in [
        ("overall_stage", "Neoplasm Disease Stage American Joint Committee on Cancer Code", _map_overall_stage),
        ("t_stage", "American Joint Committee on Cancer Tumor Stage Code", _map_t_stage),
        ("n_stage", "Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code", _map_n_stage),
        ("m_stage", "American Joint Committee on Cancer Metastasis Stage Code", _map_m_stage),
    ]:
        mapped = df[col].map(mapper)
        train_mode = mapped[train_mask].mode().iloc[0]
        out[f"{label}_missing"] = mapped.isna().astype(int)
        out[label] = mapped.fillna(train_mode)

    # --- Sex, Race Category, Ethnicity Category, Histological Subtype: one-hot ---
    out = _onehot_with_missingness(df, "Sex", "sex", train_mask, out)
    out = _onehot_with_missingness(df, "Race Category", "race", train_mask, out)
    out = _onehot_with_missingness(df, "Ethnicity Category", "ethnicity", train_mask, out)
    out = _onehot_with_missingness(df, "Tumor Type", "subtype", train_mask, out)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--clinical", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out", default="clinical_features.csv")
    parser.add_argument("--train_split_label", default="train")
    args = parser.parse_args()

    df = load_and_merge(args.labels, args.clinical, args.splits)
    print(f"Merged: {len(df)} patients")
    print(f"Split counts:\n{df['split'].value_counts()}")

    features = build_features(df, train_split_label=args.train_split_label)
    features.to_csv(args.out, index=False)

    non_id_cols = [c for c in features.columns if c not in ("patient", "split")]
    print(f"\nSaved clinical feature matrix to {args.out}")
    print(f"Total feature columns: {len(non_id_cols)}")
    print(non_id_cols)

    print(f"\nMissingness rate per variable:")
    for col in [c for c in features.columns if c.endswith("_missing")]:
        print(f"  {col}: {features[col].sum()}/{len(features)} ({100*features[col].mean():.1f}%)")


if __name__ == "__main__":
    main()
