"""
Boxplot (low vs high attention) with per-slide paired lines overlaid.
Source: shape_descriptor_analysis/option_a_results.csv
Run with venv310 (needs pandas, matplotlib).

Usage:
    python3 plot_shape_attention.py
Output:
    fig_shape_attention.png  (300 dpi, ready for the thesis)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_PATH = "/cs/student/project_msc/2025/aibh/iconstan/shape_descriptor_analysis/option_a_results.csv"
OUT_PATH = "/cs/student/project_msc/2025/aibh/iconstan/shape_descriptor_analysis/fig_shape_attention.png"

DESCRIPTORS = ["area", "circularity", "solidity", "eccentricity"]
TASK_LABELS = {"multiclass": "Multiclass", "ret_binary": "RET binary"}
TASK_COLORS = {"multiclass": "#1f77b4", "ret_binary": "#d62728"}
BOX_POSITIONS = {"lo": 0, "hi": 1}

df = pd.read_csv(IN_PATH)
df["descriptor"] = df["descriptor"].str.lower()

print("Tasks found in file:", sorted(df["task"].unique()))

fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))

for ax, desc in zip(axes, DESCRIPTORS):
    sub = df[df["descriptor"] == desc]
    if sub.empty:
        ax.set_title(f"{desc} (no data)")
        continue

    tasks = list(sub["task"].unique())
    n_tasks = len(tasks)
    # lay tasks side by side within each attention level: e.g. for 2 tasks,
    # low-attention boxes at x = -0.2, 0.2 and high-attention at x = 0.8, 1.2
    task_offset = {
        t: (i - (n_tasks - 1) / 2) * 0.4 for i, t in enumerate(tasks)
    }

    for task in tasks:
        t = sub[sub["task"] == task]
        color = TASK_COLORS.get(task, "gray")
        offset = task_offset[task]
        x_lo, x_hi = BOX_POSITIONS["lo"] + offset, BOX_POSITIONS["hi"] + offset

        # boxplots for this task at its own x position
        bp = ax.boxplot(
            [t["lo_median"].values, t["hi_median"].values],
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

        # paired lines per slide, on top of the boxes
        for _, row in t.iterrows():
            ax.plot(
                [x_lo, x_hi],
                [row["lo_median"], row["hi_median"]],
                color=color,
                alpha=0.3,
                linewidth=0.8,
                zorder=1,
            )
        ax.scatter(
            [x_lo] * len(t), t["lo_median"], color=color, s=10, zorder=3, alpha=0.7
        )
        ax.scatter(
            [x_hi] * len(t),
            t["hi_median"],
            color=color,
            s=10,
            zorder=3,
            alpha=0.7,
            label=TASK_LABELS.get(task, task),
        )

    # tick positions centred between the low/high groups
    tick_positions = [np.mean([BOX_POSITIONS["lo"] + o for o in task_offset.values()]),
                       np.mean([BOX_POSITIONS["hi"] + o for o in task_offset.values()])]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(["Low attention", "High attention"])
    ax.set_xlim(-0.8, 1.8)
    ax.set_title(desc.capitalize())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Per-slide median value")
handles, labels = axes[-1].get_legend_handles_labels()
# de-duplicate legend entries (each task appears twice: lo and hi scatter calls)
seen = dict(zip(labels, handles))
axes[-1].legend(seen.values(), seen.keys(), loc="upper right", fontsize=8, frameon=False)

fig.suptitle(
    "Epithelial nuclear shape descriptors: high- vs low-attention tiles\n(boxplots with per-slide paired values, n = 30 slides)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")

summary = (
    df.groupby(["task", "descriptor"])[["p_bh", "significant"]]
    .first()
    .reset_index()
)
print(summary.to_string(index=False))
