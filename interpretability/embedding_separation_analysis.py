"""
embedding_separation_analysis.py

Quantifies whether the 2D PCA/UMAP projections of the slide-level ABMIL
embeddings show genuine genotype-related structure, using permutation-tested
k-nearest-neighbour (k-NN) label purity rather than visual inspection of the
scatter plots.

For each point, "purity" is the fraction of its k nearest neighbours (in the
2D projection) that share its own label. This is computed per point, then
averaged. Three layers of rigour are applied on top of the raw statistic:

  1. Permutation testing. The neighbour graph is fixed; labels are shuffled
     N_PERM times to build a null distribution, giving a p-value rather than
     a qualitative "looks separated" impression. This does not assume convex
     clusters (unlike e.g. a silhouette score) and correctly accounts for
     class imbalance, since the permutation null preserves class proportions.

  2. Benjamini-Hochberg correction. Multiple tests are run per task (pooled
     UMAP, pooled PCA, site, split, and per-class one-vs-rest for the
     multiclass task) — 8 tests across the two tasks in the primary battery.
     Raw p-values are corrected across this full family, consistent with the
     BH correction used elsewhere in this project (TIIC analysis).

  3. Bootstrap confidence intervals. Per-point purity values are resampled
     with replacement (2,000 iterations, matching the bootstrap convention
     used for the headline performance metrics in this project) to give a
     95% CI on the purity estimate itself. This resamples the *per-point
     purity values*, not the raw coordinates — resampling coordinates and
     rebuilding the neighbour graph would let a bootstrap-duplicated patient
     become its own nearest neighbour, trivially inflating purity. Resampling
     the already-computed per-point statistic avoids this and is the same
     principle as bootstrapping any other per-observation metric.

  4. k-sensitivity sweep. The headline tests (pooled purity by genotype/RET,
     and by site) are repeated across a range of k values, so the finding
     is not reported as an artefact of one specific, arbitrary k choice.
     k=15 (matching the n_neighbors used for UMAP itself) is the primary,
     pre-specified value; the sweep is a robustness check.

Usage
-----
    python embedding_separation_analysis.py

Expects embedding_coords_multiclass.csv and embedding_coords_ret_binary.csv
(the output of embedding_analysis_v3.py) in the same directory, or edit the
paths below.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# === PATHS — edit if running from a different location ===
MULTICLASS_CSV = "embedding_coords_multiclass.csv"
RET_BINARY_CSV = "embedding_coords_ret_binary.csv"

K_PRIMARY = 15                       # matches n_neighbors used for UMAP in embedding_analysis_v3.py
K_SWEEP = [5, 10, 15, 20, 30, 50]     # robustness check around K_PRIMARY
N_PERM = 20000                        # permutation iterations
N_BOOT = 2000                        # bootstrap iterations (matches thesis-wide convention)
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Core statistic
# ---------------------------------------------------------------------------

def _neighbor_index(coords, k):
    """k nearest-neighbour indices for each point, excluding the point itself."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    return idx[:, 1:]


def _per_point_purity(idx, labels):
    """Per-point fraction of k nearest neighbours sharing that point's label."""
    labels = np.asarray(labels)
    neighbor_labels = labels[idx]
    return (neighbor_labels == labels[:, None]).mean(axis=1)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(coords, labels, k, n_perm=N_PERM, rng=None):
    """
    Observed mean k-NN purity vs. a permutation null (labels shuffled
    n_perm times, neighbour graph fixed).

    Returns: observed, null_mean, null_sd, z, p (one-sided, purity > chance)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    idx = _neighbor_index(coords, k=k)
    labels = np.asarray(labels)

    observed = _per_point_purity(idx, labels).mean()
    perm_scores = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(labels)
        perm_scores[i] = _per_point_purity(idx, shuffled).mean()

    null_mean = perm_scores.mean()
    null_sd = perm_scores.std(ddof=1)
    z = (observed - null_mean) / null_sd if null_sd > 0 else np.nan
    p = (1 + np.sum(perm_scores >= observed)) / (n_perm + 1)
    return observed, null_mean, null_sd, z, p


# ---------------------------------------------------------------------------
# Bootstrap CI (on per-point purity values, not on the coordinates)
# ---------------------------------------------------------------------------

def bootstrap_ci(coords, labels, k, n_boot=N_BOOT, rng=None, alpha=0.05):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    idx = _neighbor_index(coords, k=k)
    per_point = _per_point_purity(idx, labels)
    n = len(per_point)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(per_point, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


# ---------------------------------------------------------------------------
# Multiple-comparisons correction
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvals):
    """Standard BH step-up procedure. Returns adjusted p-values, same order as input."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = ranked * m / (np.arange(m) + 1)
    # enforce monotonicity from the largest p-value downward
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(m)
    out[order] = adjusted
    return out


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_primary_battery(df, task_name, label_col, rng):
    """Runs the full pre-specified test battery at K_PRIMARY and returns a results table."""
    umap_coords = df[["umap_x", "umap_y"]].values
    pca_coords = df[["pca_x", "pca_y"]].values
    labels = df[label_col].values

    tests = [
        (f"{task_name}: UMAP by site", umap_coords, df["site"].values),
        (f"{task_name}: UMAP by split", umap_coords, df["split"].values),
    ]

    rows = []
    for name, coords, grp in tests:
        observed, null_mean, null_sd, z, p = permutation_test(coords, grp, k=K_PRIMARY, rng=rng)
        ci_lo, ci_hi = bootstrap_ci(coords, grp, k=K_PRIMARY, rng=rng)
        rows.append({
            "test": name, "observed": observed, "null_mean": null_mean,
            "z": z, "p_raw": p, "ci_lo": ci_lo, "ci_hi": ci_hi,
        })
    return pd.DataFrame(rows)


def run_k_sweep(df, task_name, label_col, rng):
    """Robustness check: repeat the two headline tests (label, site) across a range of k."""
    umap_coords = df[["umap_x", "umap_y"]].values
    labels = df[label_col].values
    site = df["site"].values

    rows = []
    for k in K_SWEEP:
        for name, grp in [(f"by {label_col}", labels), ("by site", site)]:
            observed, null_mean, null_sd, z, p = permutation_test(umap_coords, grp, k=k, rng=rng)
            rows.append({"task": task_name, "test": name, "k": k,
                          "observed": observed, "null_mean": null_mean, "p_raw": p})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df_mc = pd.read_csv(MULTICLASS_CSV)
    df_ret = pd.read_csv(RET_BINARY_CSV)

    print(f"Multiclass: n={len(df_mc)}, class counts: {df_mc['mutation_label'].value_counts().to_dict()}")
    print(f"RET binary: n={len(df_ret)}, class counts: {df_ret['RET'].value_counts().to_dict()}")

    # --- Primary battery, both tasks, corrected together as one family ---
    results_mc = run_primary_battery(df_mc, "Multiclass", "mutation_label", rng)
    results_ret = run_primary_battery(df_ret, "RET binary", "RET", rng)
    all_results = pd.concat([results_mc, results_ret], ignore_index=True)
    all_results["p_bh"] = benjamini_hochberg(all_results["p_raw"].values)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 40)
    print(f"\n{'=' * 100}")
    print(f"PRIMARY BATTERY (k={K_PRIMARY}), BH-corrected across all {len(all_results)} tests")
    print(f"{'=' * 100}")
    print(all_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    n_sig_raw = (all_results["p_raw"] < 0.05).sum()
    n_sig_bh = (all_results["p_bh"] < 0.05).sum()
    print(f"\nSignificant at alpha=0.05: {n_sig_raw}/{len(all_results)} raw, "
          f"{n_sig_bh}/{len(all_results)} after BH correction")

    # --- k-sensitivity sweep for the two headline tests per task ---
    sweep_mc = run_k_sweep(df_mc, "Multiclass", "mutation_label", rng)
    sweep_ret = run_k_sweep(df_ret, "RET binary", "RET", rng)
    sweep = pd.concat([sweep_mc, sweep_ret], ignore_index=True)

    print(f"\n{'=' * 100}")
    print(f"K-SENSITIVITY SWEEP (headline tests only: label, site)")
    print(f"{'=' * 100}")
    print(sweep.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    all_results.to_csv("embedding_separation_primary_results.csv", index=False)
    sweep.to_csv("embedding_separation_k_sweep.csv", index=False)
    print("\nSaved: embedding_separation_primary_results.csv, embedding_separation_k_sweep.csv")


if __name__ == "__main__":
    main()
