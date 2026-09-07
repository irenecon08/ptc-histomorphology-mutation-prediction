import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import kruskal, mannwhitneyu

base = "/cs/student/project_msc/2025/aibh/iconstan/tiic_analysis_v6/"
sp = pd.read_csv("/cs/student/project_msc/2025/aibh/iconstan/splits.csv")

for task in ["multiclass", "ret_binary"]:
    d = pd.read_csv(base + f"tiic_summary_{task}.csv")
    n = len(d)
    print(f"\n{'='*55}\n=== {task} ({n} slides) ===")
    print(f"raw r      : mean {d['spearman_r_immune_density'].mean():.3f}  median {d['spearman_r_immune_density'].median():.3f}")
    print(f"epithelial : mean {d['spearman_r_epithelial_density'].mean():.3f}")
    print(f"partial r  : mean {d['partial_r_immune_given_epithelial'].mean():.3f}  median {d['partial_r_immune_given_epithelial'].median():.3f}")
    print(f"raw significant after BH   : {d['significant_after_correction'].sum()} / {n}")

    rej, _, _, _ = multipletests(d['partial_p_immune_given_epithelial'].values,
                                 alpha=0.05, method='fdr_bh')
    d = d.assign(sig_partial=rej)
    print(f"partial significant after BH: {rej.sum()} / {n}")
    sig = d[d['sig_partial']]
    print(f"  of those, negative: {(sig['partial_r_immune_given_epithelial'] < 0).sum()}, "
          f"positive: {(sig['partial_r_immune_given_epithelial'] > 0).sum()}")
    print(f"partial r range: {d['partial_r_immune_given_epithelial'].min():.3f} to "
          f"{d['partial_r_immune_given_epithelial'].max():.3f}")

    m = d.merge(sp, on="patient", how="left")

    # --- residual by mutation class (valid for BOTH tasks: genotype is a slide property) ---
    print("\n  -- residual by mutation_label --")
    g = m.groupby("mutation_label").agg(
        n_slides=("patient", "count"),
        n_sig=("sig_partial", "sum"),
        mean_partial_r=("partial_r_immune_given_epithelial", "mean"),
        median_partial_r=("partial_r_immune_given_epithelial", "median"),
    )
    g["pct_sig"] = (100 * g["n_sig"] / g["n_slides"]).round(1)
    print(g)

    groups = [v["partial_r_immune_given_epithelial"].values
              for _, v in m.groupby("mutation_label") if len(v) >= 3]
    if len(groups) >= 2:
        H, p = kruskal(*groups)
        print(f"  Kruskal-Wallis on partial r across mutation classes: H={H:.2f}, p={p:.4f}")

    # --- residual by RET status ---
    print("\n  -- residual by RET status --")
    g2 = m.groupby("RET").agg(
        n_slides=("patient", "count"),
        n_sig=("sig_partial", "sum"),
        mean_partial_r=("partial_r_immune_given_epithelial", "mean"),
    )
    print(g2)
    pos = m[m["RET"] == 1]["partial_r_immune_given_epithelial"].values
    neg = m[m["RET"] == 0]["partial_r_immune_given_epithelial"].values
    if len(pos) >= 3 and len(neg) >= 3:
        U, p = mannwhitneyu(pos, neg)
        print(f"  Mann-Whitney RET+ vs RET- on partial r: U={U:.1f}, p={p:.4f}")

    # --- outlier slides ---
    print("\n  -- strongest residuals --")
    print(m.nsmallest(3, "partial_r_immune_given_epithelial")[
        ["patient", "mutation_label", "RET", "partial_r_immune_given_epithelial",
         "spearman_r_epithelial_density"]].to_string(index=False))
