"""
hypothesis2_select_slides.py

Selects slides for the Hypothesis 2 spatial exclusion check.

SELECTION CRITERION (fixed, documented, and applied BEFORE this spatial
question was ever examined): slides ranked by strongest (most negative)
partial correlation (attention vs immune density, controlling for
epithelial density) from the ALREADY-COMPLETED v6 residual analysis
(tiic_summary_{task}.csv). This criterion comes from a different analysis,
run for a different purpose, with no knowledge of spatial patterns. It is
therefore not selection bias with respect to the spatial question tested
here: these slides could not have been chosen because they "looked
spatially interesting", since nobody had examined their spatial patterns
when the selection was made.

Selects the union of the top-N most negative slides across BOTH tasks
(multiclass and ret_binary), so a slide flagged as an outlier in either
model qualifies.

No GPU required.

Usage:
    python hypothesis2_select_slides.py --n 5
"""

import argparse
import pandas as pd

TIIC_DIR = "/cs/student/project_msc/2025/aibh/iconstan/tiic_analysis_v6/"


def top_n_outliers(task, n):
    df = pd.read_csv(TIIC_DIR + f"tiic_summary_{task}.csv")
    df = df.sort_values("partial_r_immune_given_epithelial")  # most negative first
    top = df.head(n)[["patient", "partial_r_immune_given_epithelial",
                       "spearman_r_epithelial_density"]].copy()
    top["source_task"] = task
    return top


def main(n):
    print("=" * 70)
    print(f"HYPOTHESIS 2 SLIDE SELECTION")
    print(f"Criterion: top {n} most negative partial correlation, per task, from")
    print(f"the already-completed v6 residual analysis (tiic_summary_{{task}}.csv).")
    print(f"Union taken across both tasks.")
    print("=" * 70)

    mc = top_n_outliers("multiclass", n)
    rb = top_n_outliers("ret_binary", n)

    print(f"\nTop {n}, multiclass:")
    print(mc.to_string(index=False))
    print(f"\nTop {n}, ret_binary:")
    print(rb.to_string(index=False))

    union = pd.concat([mc, rb]).drop_duplicates(subset="patient")
    union = union.sort_values("partial_r_immune_given_epithelial")

    print(f"\n{'='*70}")
    print(f"SELECTED SET (union, {len(union)} unique slides):")
    print(f"{'='*70}")
    print(union.to_string(index=False))

    out_path = "/cs/student/project_msc/2025/aibh/iconstan/hypothesis2_selected_slides.csv"
    union.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"\nThis file is the documented, reproducible record of which slides")
    print(f"were selected and why, to be cited in the Methods write-up.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5,
                     help="Top-N most negative slides to take PER TASK before union.")
    args = ap.parse_args()
    main(args.n)
