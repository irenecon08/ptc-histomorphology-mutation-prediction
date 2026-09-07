"""
hypothesis1_compositional_checks.py

Tests Hypothesis 1: the negative attention-immune correlation is a mechanical
side-effect of immune and epithelial density being compositionally coupled
(both are proportions of the same tile, so more of one leaves less room for
the other), rather than evidence attention has any independent relationship
to immune content.

Uses only the existing tiic_per_tile_{task}.csv files. No GPU, no HoVer-Net,
no model loading required.

Four checks, per task:

  1a. Correlation between epithelial density and immune density directly
      (attention not involved). Tests how tight the mechanical trade-off
      actually is.

  1b. Average share of nuclei per tile classified as "other" (neither
      epithelial nor immune). Tests how much compositional slack exists.

  1c. Attention-immune correlation using immune COUNT instead of density.
      Count is not compositionally constrained the way density is. If the
      relationship survives with count, it's not purely a density artifact.

  1d. Attention-immune correlation computed WITHIN narrow epithelial-density
      bins, so epithelium is held roughly constant within each bin. If the
      relationship persists within bins, it can't be fully explained by
      epithelium differences between tiles.

Usage:
    python hypothesis1_compositional_checks.py
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

PROJ = "/cs/student/project_msc/2025/aibh/iconstan/"
TIIC_DIR = PROJ + "tiic_analysis_v6/"   # point at the corrected v6 data
N_BINS = 10


def load_tiles(task):
    path = os.path.join(TIIC_DIR, f"tiic_per_tile_{task}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def check_1a(df):
    r, p = spearmanr(df["epithelial_density"], df["immune_density"])
    print(f"  1a. epithelial density vs immune density (direct, no attention): "
          f"r={r:.3f}, p={p:.3g}")
    print(f"      -> close to -1 supports near-total mechanical exclusion.")
    print(f"      -> more moderate (e.g. -0.4 to -0.6) suggests real slack "
          f"exists.")
    return r


def check_1b(df):
    other_frac = 1 - (df["immune_nuclei"] + df["epithelial_nuclei"]) / df["total_nuclei"]
    mean_other = other_frac.mean()
    print(f"  1b. mean 'other' (unclassified) nuclei share per tile: "
          f"{mean_other:.1%}")
    print(f"      -> near 0% means epithelial+immune ~fully occupy each "
          f"tile (tighter compositional trap).")
    print(f"      -> a substantial share suggests more compositional slack.")
    return mean_other


def check_1c(df):
    r_density, p_density = spearmanr(df["attn"], df["immune_density"])
    r_count, p_count = spearmanr(df["attn"], df["immune_nuclei"])
    print(f"  1c. attention vs immune DENSITY: r={r_density:.3f} (p={p_density:.3g})")
    print(f"      attention vs immune COUNT:   r={r_count:.3f} (p={p_count:.3g})")
    ratio = abs(r_count) / abs(r_density) if r_density != 0 else float("nan")
    print(f"      count/density strength ratio: {ratio:.2f}")
    print(f"      -> ratio near 1.0: relationship survives using count, "
          f"weakens the pure-density-artifact explanation.")
    print(f"      -> ratio well below 1.0: relationship is much weaker with "
          f"count, supports the density-artifact explanation.")
    return r_density, r_count


def check_1d(df, n_bins=N_BINS):
    df = df.copy()
    df["epi_bin"] = pd.qcut(df["epithelial_density"], n_bins, duplicates="drop")

    print(f"  1d. attention vs immune density, WITHIN {n_bins} epithelial-density bins:")
    bin_results = []
    for b, grp in df.groupby("epi_bin", observed=True):
        if len(grp) < 20:
            continue
        r, p = spearmanr(grp["attn"], grp["immune_density"])
        bin_results.append({"bin": str(b), "n": len(grp), "r": r, "p": p})
        print(f"      epi_density in {b}: n={len(grp):5d}  r={r:+.3f}  p={p:.3g}")

    rdf = pd.DataFrame(bin_results)
    if len(rdf):
        print(f"\n      mean within-bin r: {rdf['r'].mean():+.3f}")
        print(f"      bins with r < 0:   {(rdf['r'] < 0).sum()} / {len(rdf)}")
        print(f"      -> if within-bin correlations remain consistently "
              f"negative (even if weaker than the overall -0.55ish), the "
              f"relationship is not fully explained by epithelium varying "
              f"between tiles.")
        print(f"      -> if within-bin correlations are near zero or "
              f"inconsistent in sign, epithelium alone plausibly explains "
              f"the overall pattern.")
    return rdf


def main():
    for task in ["multiclass", "ret_binary"]:
        print(f"\n{'='*70}")
        print(f"HYPOTHESIS 1 CHECKS — {task}")
        print(f"{'='*70}")
        df = load_tiles(task)
        print(f"  ({len(df)} tiles, {df['patient'].nunique()} slides)\n")

        check_1a(df)
        print()
        check_1b(df)
        print()
        check_1c(df)
        print()
        check_1d(df)


if __name__ == "__main__":
    main()
