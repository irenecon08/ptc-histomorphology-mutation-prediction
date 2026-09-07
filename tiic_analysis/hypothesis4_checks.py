"""
hypothesis4_checks.py

HYPOTHESIS 4: the negative attention-immune correlation is partly or wholly
a measurement artefact, if HoVer-Net's nucleus detection becomes less
reliable in epithelium-dense, densely packed tissue (e.g. due to nuclear
crowding or overlap), this could produce spuriously low nucleus counts in
exactly the tiles attention concentrates on, mimicking a real immune-avoidance
signal without one existing.

CHECK: correlation between total detected nuclei per tile and epithelial
density. If detection is degraded in dense tissue, total nuclei detected
should fall as epithelial density rises. If detection is not obviously
failing, total nuclei should remain roughly stable (or even rise, since
epithelial tissue is often more densely cellular than stroma).

This is a plausibility check, not a definitive test: a stable or rising
nuclei count does not prove HoVer-Net's PER-CLASS accuracy is unaffected
by domain shift (MoNuSAC does not include thyroid), only that it is not
failing to detect nuclei altogether in dense regions.

Uses only the existing tiic_per_tile_{task}.csv files (tiic_analysis_v6/).
No GPU, no HoVer-Net, no model loading required.

Usage:
    python hypothesis4_checks.py
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

PROJ = "/cs/student/project_msc/2025/aibh/iconstan/"
TIIC_DIR = PROJ + "tiic_analysis_v6/"
N_BINS = 10


def load_tiles(task):
    path = os.path.join(TIIC_DIR, f"tiic_per_tile_{task}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def check_4a(df):
    """Overall correlation: total nuclei vs epithelial density."""
    r, p = spearmanr(df["epithelial_density"], df["total_nuclei"])
    print(f"  4a. epithelial density vs total nuclei detected: r={r:+.3f}, p={p:.3g}")
    print(f"      mean total_nuclei overall: {df['total_nuclei'].mean():.1f}")
    print(f"      -> if r is strongly NEGATIVE, detection may be degraded in dense tissue "
          f"(measurement artefact concern).")
    print(f"      -> if r is near zero or POSITIVE, detection is not obviously failing "
          f"in epithelium-dense regions.")
    return r


def check_4b(df):
    """Compare total nuclei at the extremes of epithelial density."""
    low_epi = df[df["epithelial_density"] <= df["epithelial_density"].quantile(0.2)]
    high_epi = df[df["epithelial_density"] >= df["epithelial_density"].quantile(0.8)]
    print(f"\n  4b. total nuclei at the extremes of epithelial density:")
    print(f"      lowest-epithelium 20% of tiles:  mean={low_epi['total_nuclei'].mean():.1f}  "
          f"median={low_epi['total_nuclei'].median():.1f}  n={len(low_epi)}")
    print(f"      highest-epithelium 20% of tiles: mean={high_epi['total_nuclei'].mean():.1f}  "
          f"median={high_epi['total_nuclei'].median():.1f}  n={len(high_epi)}")
    pct_change = 100 * (high_epi["total_nuclei"].mean() - low_epi["total_nuclei"].mean()) / low_epi["total_nuclei"].mean()
    print(f"      change (high vs low epithelium): {pct_change:+.1f}%")
    return low_epi["total_nuclei"].mean(), high_epi["total_nuclei"].mean()


def check_4c(df, n_bins=N_BINS):
    """Total nuclei across epithelial-density deciles, to see the full trend
    (not just the two extremes), and check for a non-monotonic pattern that
    would suggest something more specific than a simple linear artefact."""
    df = df.copy()
    df["epi_bin"] = pd.qcut(df["epithelial_density"], n_bins, duplicates="drop")
    print(f"\n  4c. mean total nuclei across {n_bins} epithelial-density bins:")
    for b, grp in df.groupby("epi_bin", observed=True):
        print(f"      epi_density in {b}: n={len(grp):5d}  mean total_nuclei={grp['total_nuclei'].mean():6.1f}")


def check_4d(df):
    """Sanity cross-check: does total nuclei relate similarly to immune density
    (the other component), which should NOT show a detection-artefact pattern
    if the artefact is specific to epithelium-dense crowding."""
    r, p = spearmanr(df["immune_density"], df["total_nuclei"])
    print(f"\n  4d. immune density vs total nuclei detected (cross-check): r={r:+.3f}, p={p:.3g}")
    print(f"      -> for comparison against 4a. If total nuclei falls with epithelial density")
    print(f"      but not with immune density, this points more specifically at epithelium-")
    print(f"      dense crowding as the cause, rather than a general detection instability.")


def main():
    for task in ["multiclass", "ret_binary"]:
        print(f"\n{'='*70}")
        print(f"HYPOTHESIS 4 CHECKS — {task}")
        print(f"{'='*70}")
        df = load_tiles(task)
        print(f"  ({len(df)} tiles, {df['patient'].nunique()} slides)\n")

        check_4a(df)
        check_4b(df)
        check_4c(df)
        check_4d(df)


if __name__ == "__main__":
    main()
