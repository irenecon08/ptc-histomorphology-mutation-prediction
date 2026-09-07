"""
Grouped boxplot: epithelial nuclear shape descriptors by genotype.
Source: shape_descriptor_analysis_genotype/option_b_per_patient.csv
Run with venv310 (needs pandas, matplotlib).

Usage:
    python3 plot_shape_genotype.py
Output:
    fig_shape_genotype.png  (300 dpi, ready for the thesis)
"""

import pandas as pd
import matplotlib.pyplot as plt

IN_PATH = "/cs/student/project_msc/2025/aibh/iconstan/shape_descriptor_analysis_genotype/option_b_per_patient.csv"
OUT_PATH = "/cs/student/project_msc/2025/aibh/iconstan/shape_descriptor_analysis_genotype/fig_shape_genotype.png"

DESCRIPTORS = ["area", "circularity", "solidity", "eccentricity"]
GENOTYPE_ORDER = ["BRAF", "RAS", "RET", "Other"]  # adjust to match actual labels in the CSV
GENOTYPE_COLORS = {
    "BRAF": "#1f77b4",
    "RAS": "#2ca02c",
    "RET": "#d62728",
    "Other": "#7f7f7f",
}

df = pd.read_csv(IN_PATH)

# confirm genotype labels match GENOTYPE_ORDER before plotting
actual = sorted(df["genotype"].unique())
print("Genotype labels found in file:", actual)
order = [g for g in GENOTYPE_ORDER if g in actual] + [
    g for g in actual if g not in GENOTYPE_ORDER
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

for ax, desc in zip(axes, DESCRIPTORS):
    data = [df.loc[df["genotype"] == g, desc].dropna().values for g in order]
    n_per_group = [len(d) for d in data]

    bp = ax.boxplot(
        data,
        labels=[f"{g}\n(n={n})" for g, n in zip(order, n_per_group)],
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(GENOTYPE_COLORS.get(g, "lightgray"))
        patch.set_alpha(0.7)

    # overlay individual points with jitter for transparency about n
    for i, d in enumerate(data, start=1):
        jitter = (pd.Series(range(len(d))).sample(frac=1, random_state=0).values % 1) * 0  # placeholder no-op
        import numpy as np
        x = np.random.default_rng(0).normal(i, 0.05, size=len(d))
        ax.scatter(x, d, color="black", s=10, alpha=0.4, zorder=3)

    ax.set_title(desc.capitalize())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Per-patient value")

fig.suptitle(
    "Epithelial nuclear shape descriptors by driver genotype (BRAF n=9, RAS n=9, RET n=6, Other n=6)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
