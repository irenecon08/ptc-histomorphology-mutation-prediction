"""
Reproduces the ABMIL-vs-Fusion bootstrap distribution figure (box + jittered
dots per metric, one panel per aim), matching the style already used for
Baseline vs ABMIL.

Reads the raw per-iteration bootstrap arrays saved by bootstrap_ci.py
(ABMIL) and fusion_bootstrap_ci.py (Fusion) -- both scripts must be rerun
with the raw_bootstrap-saving edit before this will work; the original run
only saved percentiles, not the full arrays needed to draw a distribution.

Whiskers are set explicitly to the 2.5th/97.5th percentile of the bootstrap
distribution (i.e. exactly your reported 95% CI), NOT matplotlib's default
1.5x-IQR convention -- so the plot never visually disagrees with the CIs
quoted in your text.

Usage:
    python plot_abmil_vs_fusion.py \
        --abmil_json bootstrap_ci_results.json \
        --fusion_multiclass_json fusion_bootstrap_ci_multiclass.json \
        --fusion_ret_json fusion_bootstrap_ci_ret_binary.json \
        --out fusion_vs_abmil_bootstrap.png
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["bal_acc", "auc", "auprc", "f1"]
METRIC_LABELS = {"bal_acc": "Balanced Accuracy", "auc": "AUROC", "auprc": "AUPRC", "f1": "F1 (macro)"}

COLOR_ABMIL = "#4C72B0"
COLOR_FUSION = "#DD8452"


def percentile_whiskers(data):
    """2.5th/97.5th percentile -- matches the reported 95% CI exactly,
    NOT matplotlib's default 1.5x-IQR whisker convention."""
    return np.percentile(data, 2.5), np.percentile(data, 97.5)


def draw_panel(ax, abmil_data, fusion_data, title, max_dots=300):
    """abmil_data / fusion_data: dict of metric -> list of bootstrap values."""
    y_positions = []
    y_labels = []
    box_data = []
    colors = []

    for i, metric in enumerate(METRICS):
        # Fusion above ABMIL within each metric group, matching the
        # reference image's paired layout
        y_fusion = i * 3 + 2
        y_abmil = i * 3 + 1
        y_positions.extend([y_fusion, y_abmil])
        y_labels.append(METRIC_LABELS[metric])
        box_data.append((y_fusion, fusion_data[metric], COLOR_FUSION))
        box_data.append((y_abmil, abmil_data[metric], COLOR_ABMIL))

    rng = np.random.default_rng(0)
    for y, data, color in box_data:
        data = np.array(data)
        lo, hi = percentile_whiskers(data)
        median = np.median(data)
        q1, q3 = np.percentile(data, [25, 75])

        # Box (Q1-Q3) + whiskers explicitly set to the 95% CI bounds
        ax.plot([lo, hi], [y, y], color=color, lw=1.2, zorder=2)
        ax.add_patch(plt.Rectangle((q1, y - 0.32), q3 - q1, 0.64,
                                    facecolor=color, alpha=0.35, edgecolor=color, zorder=3))
        ax.plot([median, median], [y - 0.32, y + 0.32], color=color, lw=1.8, zorder=4)

        # Downsampled jittered dots -- avoid overplotting 2000 points
        n_show = min(max_dots, len(data))
        sample = rng.choice(data, size=n_show, replace=False)
        jitter = rng.uniform(-0.22, 0.22, size=n_show)
        ax.scatter(sample, y + jitter, s=4, color=color, alpha=0.25, zorder=1, linewidths=0)

    ax.set_yticks([i * 3 + 1.5 for i in range(len(METRICS))])
    ax.set_yticklabels(y_labels)
    ax.set_xlim(0.25, 1.02)
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.set_ylim(-0.5, len(METRICS) * 3 - 0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abmil_json", required=True,
                         help="bootstrap_ci_results.json from bootstrap_ci.py")
    parser.add_argument("--fusion_multiclass_json", required=True)
    parser.add_argument("--fusion_ret_json", required=True)
    parser.add_argument("--out", default="fusion_vs_abmil_bootstrap.png")
    args = parser.parse_args()

    with open(args.abmil_json) as f:
        abmil = json.load(f)
    with open(args.fusion_multiclass_json) as f:
        fusion_mc = json.load(f)
    with open(args.fusion_ret_json) as f:
        fusion_ret = json.load(f)

    for name, d in [("ABMIL multiclass", abmil.get("multiclass", {})),
                     ("Fusion multiclass", fusion_mc.get("multiclass", {}))]:
        if "raw_bootstrap" not in d:
            raise ValueError(f"{name} JSON has no 'raw_bootstrap' key -- rerun the "
                              f"bootstrap script with the raw-array-saving edit first.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    draw_panel(
        axes[0],
        abmil["multiclass"]["raw_bootstrap"],
        fusion_mc["multiclass"]["raw_bootstrap"],
        "Aim (a) Multiclass",
    )

    draw_panel(
        axes[1],
        abmil["ret_binary_optimal_threshold"]["raw_bootstrap"],
        fusion_ret["ret_binary_optimal_threshold"]["raw_bootstrap"],
        "Aim (b) RET binary (optimal threshold)",
    )

    handles = [plt.Line2D([0], [0], color=COLOR_ABMIL, lw=6, alpha=0.6, label="ABMIL (WSI-only)"),
               plt.Line2D([0], [0], color=COLOR_FUSION, lw=6, alpha=0.6, label="Fusion")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08), frameon=False)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
