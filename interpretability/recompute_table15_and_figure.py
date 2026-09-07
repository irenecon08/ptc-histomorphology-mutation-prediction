"""
Recompute Table 15 (paired Wilcoxon signed-rank test, per-slide medians,
n = 30 genotype-stratified slides) from the correct per-slide files, and
redraw the boxplot-with-paired-lines figure from the same data.

Source files (30-slide, genotype-stratified sample — NOT the 7-slide h2 sample):
    shape_descriptor_analysis_genotype/option_a_per_slide_multiclass.csv
    shape_descriptor_analysis_genotype/option_a_per_slide_ret_binary.csv

Each file has columns: <patient_index>, area_hi, circularity_hi, solidity_hi,
eccentricity_hi, area_lo, circularity_lo, solidity_lo, eccentricity_lo

Run with venv310 (needs pandas, scipy, matplotlib, numpy).

Usage:
    python3 recompute_table15_and_figure.py
Output:
    table15_recomputed.csv           (p-values + medians, for checking against draft)
    fig_shape_attention_corrected.png
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

GENO_DIR = "/cs/student/project_msc/2025/aibh/iconstan/shape_descriptor_analysis_genotype/"
IN_PATHS = {
    "multiclass": GENO_DIR + "option_a_per_slide_multiclass.csv",
    "ret_binary": GENO_DIR + "option_a_per_slide_ret_binary.csv",
}
TABLE_OUT = GENO_DIR + "table15_recomputed.csv"
FIG_OUT = GENO_DIR + "fig_shape_attention_corrected.png"

DESCRIPTORS = ["area", "circularity", "solidity", "eccentricity"]
TASK_LABELS = {"multiclass": "Multiclass", "ret_binary": "RET binary"}
TASK_COLORS = {"multiclass": "#1f77b4", "ret_binary": "#d62728"}

# ---- load and sanity check ----
data = {}
for task, path in IN_PATHS.items():
    df = pd.read_csv(path, index_col=0)
    print(f"{task}: {len(df)} slides, columns: {list(df.columns)}")
    assert len(df) == 30, f"expected 30 slides for {task}, got {len(df)}"
    data[task] = df

# ---- compute paired Wilcoxon per task, per descriptor, BH-correct across the 4 descriptors ----
results = []
for task, df in data.items():
    pvals = []
    rows = []
    for desc in DESCRIPTORS:
        hi = df[f"{desc}_hi"].values
        lo = df[f"{desc}_lo"].values
        stat, p_raw = wilcoxon(hi, lo)
        rows.append({
            "task": task,
            "descriptor": desc,
            "hi_median": np.median(hi),
            "lo_median": np.median(lo),
            "p_raw": p_raw,
        })
        pvals.append(p_raw)
    _, p_bh, _, _ = multipletests(pvals, method="fdr_bh")
    for row, p_corrected in zip(rows, p_bh):
        row["p_bh"] = p_corrected
        row["significant"] = p_corrected < 0.05
        direction = "higher in high-attention" if row["hi_median"] > row["lo_median"] else "lower in high-attention"
        row["direction"] = direction if row["significant"] else "no consistent difference"
        results.append(row)

table15 = pd.DataFrame(results)
table15.to_csv(TABLE_OUT, index=False)
print("\n=== Recomputed Table 15 ===")
print(table15[["task", "descriptor", "hi_median", "lo_median", "p_raw", "p_bh", "significant", "direction"]].to_string(index=False))
print(f"\nSaved: {TABLE_OUT}")

# ---- rebuild the figure from this same, correct data ----
fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))

for ax, desc in zip(axes, DESCRIPTORS):
    tasks = list(data.keys())
    n_tasks = len(tasks)
    task_offset = {t: (i - (n_tasks - 1) / 2) * 0.4 for i, t in enumerate(tasks)}

    for task in tasks:
        df = data[task]
        hi = df[f"{desc}_hi"].values
        lo = df[f"{desc}_lo"].values
        color = TASK_COLORS.get(task, "gray")
        offset = task_offset[task]
        x_lo, x_hi = 0 + offset, 1 + offset

        bp = ax.boxplot(
            [lo, hi],
            positions=[x_lo, x_hi],
            widths=0.32,
            patch_artist=True,
            showfliers=False,
            zorder=2,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color(color)

        for lo_v, hi_v in zip(lo, hi):
            ax.plot([x_lo, x_hi], [lo_v, hi_v], color=color, alpha=0.3, linewidth=0.8, zorder=1)

        ax.scatter([x_lo] * len(lo), lo, color=color, s=10, zorder=3, alpha=0.7)
        ax.scatter([x_hi] * len(hi), hi, color=color, s=10, zorder=3, alpha=0.7,
                   label=TASK_LABELS.get(task, task))

    tick_positions = [np.mean([0 + o for o in task_offset.values()]),
                       np.mean([1 + o for o in task_offset.values()])]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(["Low attention", "High attention"])
    ax.set_xlim(-0.8, 1.8)
    ax.set_title(desc.capitalize())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Per-slide median value")
handles, labels = axes[-1].get_legend_handles_labels()
seen = dict(zip(labels, handles))
axes[-1].legend(seen.values(), seen.keys(), loc="upper right", fontsize=8, frameon=False)

fig.suptitle(
    "Epithelial nuclear shape descriptors: high- vs low-attention tiles\n(boxplots with per-slide paired values, n = 30 genotype-stratified slides)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
print(f"Saved: {FIG_OUT}")
