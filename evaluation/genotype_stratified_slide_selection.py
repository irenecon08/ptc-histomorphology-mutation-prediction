"""
genotype_stratified_slide_selection.py

Selects slides for a genotype-BALANCED extension of the shape-descriptor
and Hypothesis 2 (spatial exclusion) analyses, distinct from the original
7-slide Hypothesis 2 case study (which was selected by residual-correlation
strength, not genotype, and is unsuitable for a fair cross-genotype
comparison since its BRAF slides are systematically outlier-selected while
RAS/RET are not).

SELECTION RULE ("Design A"), fixed before any result on these slides exists:
  - RAS:        ALL slides in the test set (exhaustive, n=9)
  - RET:        ALL slides in the test set (exhaustive, n=6)
  - BRAF:       random sample, same size as RAS (n=9), fixed seed
  - DriverNeg:  random sample, same size as RET (n=6), fixed seed

This gives every genotype group a selection rule that is either "take
everyone" (rare groups) or "take a fair random subset" (common groups),
with no group selected by an analytical/outcome-related criterion. This is
a SEPARATE sample from the original Hypothesis 2 7-slide case study, not an
extension of it - the two should not be pooled or presented as one uniform
sample, since their selection logic differs.

Usage:
    python genotype_stratified_slide_selection.py
"""

import pandas as pd
import numpy as np

PROJ = "/cs/student/project_msc/2025/aibh/iconstan/"
SPLITS_PATH = PROJ + "splits.csv"
OUTPUT_PATH = PROJ + "genotype_stratified_slides.csv"

RANDOM_SEED = 42  # matches the seed convention used throughout this project


def main():
    splits = pd.read_csv(SPLITS_PATH)
    merged = splits[splits["split"] == "test"].copy()
    # splits.csv already carries mutation_label and RET directly, no merge
    # with final_labels.csv needed (and merging them would collide on these
    # column names, since both files carry them).

    def genotype(row):
        if row["RET"] == 1:
            return "RET"
        if row["mutation_label"] == "BRAF_V600E":
            return "BRAF"
        if row["mutation_label"] == "RAS":
            return "RAS"
        return "DriverNeg"

    merged["genotype"] = merged.apply(genotype, axis=1)

    print("=" * 70)
    print("TEST SET COMPOSITION BY GENOTYPE")
    print("=" * 70)
    counts = merged["genotype"].value_counts()
    print(counts.to_string())

    rng = np.random.RandomState(RANDOM_SEED)

    ras_all = merged[merged["genotype"] == "RAS"]["patient"].tolist()
    ret_all = merged[merged["genotype"] == "RET"]["patient"].tolist()

    n_ras = len(ras_all)
    n_ret = len(ret_all)

    braf_pool = merged[merged["genotype"] == "BRAF"]["patient"].tolist()
    driverneg_pool = merged[merged["genotype"] == "DriverNeg"]["patient"].tolist()

    braf_sample = list(rng.choice(braf_pool, size=min(n_ras, len(braf_pool)), replace=False))
    driverneg_sample = list(rng.choice(driverneg_pool, size=min(n_ret, len(driverneg_pool)), replace=False))

    selected = []
    for p in ras_all:
        selected.append({"patient": p, "genotype": "RAS", "selection_rule": "exhaustive"})
    for p in ret_all:
        selected.append({"patient": p, "genotype": "RET", "selection_rule": "exhaustive"})
    for p in braf_sample:
        selected.append({"patient": p, "genotype": "BRAF", "selection_rule": f"random (seed={RANDOM_SEED})"})
    for p in driverneg_sample:
        selected.append({"patient": p, "genotype": "DriverNeg", "selection_rule": f"random (seed={RANDOM_SEED})"})

    out_df = pd.DataFrame(selected)

    print(f"\n{'='*70}")
    print(f"SELECTED SET ({len(out_df)} slides)")
    print(f"{'='*70}")
    print(out_df.groupby(["genotype", "selection_rule"]).size().to_string())
    print(f"\nFull list:")
    print(out_df.sort_values(["genotype", "patient"]).to_string(index=False))

    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"\nThis is a SEPARATE sample from hypothesis2_selected_slides.csv (the")
    print(f"original 7-slide, residual-outlier-selected case study). Do not pool")
    print(f"the two or present them as one uniform sample.")


if __name__ == "__main__":
    main()
