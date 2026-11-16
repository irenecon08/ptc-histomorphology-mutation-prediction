"""Forest plot of the paired bootstrap differences (fusion and baseline vs ABMIL).

Reads results/comparisons/*_vs_abmil_*.json written by `ptc-mil compare`.

Usage:
    python analyses/evaluation/plot_paired_forest.py
"""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ptc_mil.analysis import load_context

ctx = load_context()
COMPARISONS_DIR = ctx.paths.results_dir / "comparisons"
OUT_PATH = ctx.output_dir("figures") / "paired_bootstrap_forest.png"


def load(other, task):
    with open(COMPARISONS_DIR / f"{other}_vs_abmil_{task}.json") as f:
        return json.load(f)


METRICS = ["bal_acc", "auc", "auprc", "f1"]
METRIC_LABELS = {"bal_acc": "Balanced Acc.", "auc": "AUROC", "auprc": "AUPRC", "f1": "F1 (macro)"}

panels = [
    (
        "Aim (a) Multiclass",
        [
            ("ABMIL vs Fusion", load("fusion", "multiclass")),
            ("ABMIL vs Baseline", load("baseline", "multiclass")),
        ],
    ),
    (
        "Aim (b) RET binary",
        [
            ("ABMIL vs Fusion", load("fusion", "ret_binary")),
            ("ABMIL vs Baseline", load("baseline", "ret_binary")),
        ],
    ),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True, sharex=True)

for ax, (panel_title, comparisons) in zip(axes, panels):
    labels, points, los, his, sigs = [], [], [], [], []
    for comp_name, block in comparisons:
        for m in METRICS:
            labels.append(f"{comp_name}\n{METRIC_LABELS[m]}")
            points.append(block[m]["point_diff"])
            los.append(block[m]["95_ci"][0])
            his.append(block[m]["95_ci"][1])
            sigs.append(block[m]["significant"])

    y = np.arange(len(labels))[::-1]
    for yi, p, lo, hi, sig in zip(y, points, los, his, sigs):
        colour = "#c0392b" if sig else "#555555"
        ax.plot([lo, hi], [yi, yi], color=colour, lw=2, solid_capstyle="round")
        ax.plot(p, yi, "o", color=colour, ms=7, markeredgecolor="white", markeredgewidth=1)

    ax.axvline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("")
    ax.set_title(panel_title, fontsize=12)
    ax.grid(axis="x", alpha=0.25)

legend_elements = [
    Line2D(
        [0],
        [0],
        color="#c0392b",
        marker="o",
        lw=2,
        ms=7,
        markeredgecolor="white",
        markeredgewidth=1,
        label="Significant (95% CI excludes 0)",
    ),
    Line2D(
        [0],
        [0],
        color="#555555",
        marker="o",
        lw=2,
        ms=7,
        markeredgecolor="white",
        markeredgewidth=1,
        label="Not significant",
    ),
]
fig.legend(
    handles=legend_elements,
    loc="upper center",
    ncol=2,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, 1.02),
)

fig.supxlabel("Difference (second model − ABMIL)   —   negative favours ABMIL", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print("Saved:", OUT_PATH)
