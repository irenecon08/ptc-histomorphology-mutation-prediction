"""
Bootstrap 95% CIs for the fusion model's test-set performance.

Replicates bootstrap_ci.py's exact methodology (see that script) so fusion
results are directly comparable to the base ABMIL/baseline CIs already in
your Results tables:
  - Stratified resampling WITHIN each class separately, at that class's own
    original size, with replacement (NOT a single pooled stratified draw)
  - N_BOOTSTRAP=2000, RANDOM_SEED=42
  - Point estimate computed once on the real test set; each bootstrap
    iteration resamples from the already-computed prediction arrays (model
    inference happens once, not 2000 times)
  - If AUROC/AUPRC fail on a resample (e.g. only one class present), that
    ENTIRE iteration is dropped from all four metrics' CIs, not just the
    failing metric -- copied verbatim from bootstrap_ci.py's behavior

Unlike bootstrap_ci.py, this loads the FUSION model (frozen ABMIL z ++
clinical features -> FusionClassifier) rather than the base ABMIL model
directly, and for RET uses the fusion model's OWN validation-selected
threshold (read from fusion_results_ret_binary.json) rather than the base
model's 0.13.

Usage:
    python fusion_bootstrap_ci.py --task multiclass --fc_dim 512 --dropout 0.0
    python fusion_bootstrap_ci.py --task ret_binary --fc_dim 256 --dropout 0.0
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
)

from fusion_train import (
    ABMIL, FusionClassifier, TASK_ARCH, RESULTS_DIR, FUSION_RESULTS_DIR,
    SPLITS_PATH, EMBED_DIM, build_embedding_index, load_clinical_features,
)
import h5py

DEVICE = "cpu"  # matches bootstrap_ci.py -- inference-only, fast enough on CPU
N_BOOTSTRAP = 2000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# --- Copied verbatim from bootstrap_ci.py for exact methodological match ---

def stratified_bootstrap_indices(labels, n_classes):
    """Resample WITHIN each class separately, preserving class proportions in every bootstrap sample."""
    indices = []
    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        if len(class_idx) == 0:
            continue
        sampled = np.random.choice(class_idx, size=len(class_idx), replace=True)
        indices.extend(sampled)
    return np.array(indices)


def compute_metrics_multiclass(probs, labels, num_classes=3):
    preds = np.argmax(probs, axis=1)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        auprc = np.mean([
            average_precision_score((labels == c).astype(int), probs[:, c])
            for c in range(num_classes)
        ])
    except Exception:
        auc, auprc = np.nan, np.nan
    return bal_acc, auc, auprc, f1


def compute_metrics_binary(probs, labels, threshold):
    pos_probs = probs[:, 1]
    preds = (pos_probs >= threshold).astype(int)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    try:
        auc = roc_auc_score(labels, pos_probs)
        auprc = average_precision_score(labels, pos_probs)
    except Exception:
        auc, auprc = np.nan, np.nan
    return bal_acc, auc, auprc, f1


def run_bootstrap_binary(task, probs, labels, thresholds):
    """Fixed version for binary (RET) tasks: draws each bootstrap resample
    ONCE per iteration (not once per threshold), computes AUROC/AUPRC once
    per resample (threshold-independent, so identical across every
    threshold reported), and computes BalAcc/F1 separately for each
    threshold in `thresholds` from that SAME resample.

    This fixes a bug in the original two-call version: calling
    stratified_bootstrap_indices() a second time for a second threshold
    continues consuming from the same global RNG sequence rather than
    reusing the first call's draws, so the default- and optimal-threshold
    runs were silently using two different sets of 2,000 resamples. Since
    AUROC/AUPRC don't depend on threshold at all, this made their CIs
    differ slightly between the two runs by resampling noise alone, not
    for any real reason.

    thresholds: dict of {label: threshold_value}, e.g. {"default": 0.5, "optimal": 0.13}
    """
    print(f"\nRunning {N_BOOTSTRAP} stratified bootstrap iterations for {task} "
          f"(single resample per iteration, shared across thresholds {list(thresholds.keys())})...")

    point_auc, point_auprc = compute_metrics_binary(probs, labels, threshold=0.5)[1:3]
    point_per_threshold = {
        label: compute_metrics_binary(probs, labels, threshold=t) for label, t in thresholds.items()
    }

    boot_auc, boot_auprc = [], []
    boot_bal_acc = {label: [] for label in thresholds}
    boot_f1 = {label: [] for label in thresholds}

    for i in range(N_BOOTSTRAP):
        idx = stratified_bootstrap_indices(labels, 2)
        b_probs, b_labels = probs[idx], labels[idx]
        try:
            # AUROC/AUPRC computed ONCE per resample -- threshold-independent
            _, auc, auprc, _ = compute_metrics_binary(b_probs, b_labels, threshold=0.5)
            if np.isnan(auc) or np.isnan(auprc):
                continue
            per_thresh_this_iter = {}
            for label, t in thresholds.items():
                bal_acc, _, _, f1 = compute_metrics_binary(b_probs, b_labels, threshold=t)
                per_thresh_this_iter[label] = (bal_acc, f1)
            # only commit once we know every threshold succeeded for this resample
            boot_auc.append(auc)
            boot_auprc.append(auprc)
            for label, (bal_acc, f1) in per_thresh_this_iter.items():
                boot_bal_acc[label].append(bal_acc)
                boot_f1[label].append(f1)
        except Exception:
            continue

    def ci(arr):
        arr = np.array(arr)
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    results = {}
    for label in thresholds:
        point_bal_acc, _, _, point_f1 = point_per_threshold[label]
        results[label] = {
            "point_estimate": {
                "bal_acc": float(point_bal_acc), "auc": float(point_auc),
                "auprc": float(point_auprc), "f1": float(point_f1)
            },
            "95_ci": {
                "bal_acc": [float(x) for x in ci(boot_bal_acc[label])],
                "auc": [float(x) for x in ci(boot_auc)],
                "auprc": [float(x) for x in ci(boot_auprc)],
                "f1": [float(x) for x in ci(boot_f1[label])],
            },
            "raw_bootstrap": {
                "bal_acc": [float(x) for x in boot_bal_acc[label]],
                "auc": [float(x) for x in boot_auc],
                "auprc": [float(x) for x in boot_auprc],
                "f1": [float(x) for x in boot_f1[label]],
            },
            "n_bootstrap_valid": len(boot_auc),
            "n_bootstrap_requested": N_BOOTSTRAP,
        }
        print(f"  [{label}] BalAcc: {point_bal_acc:.4f} (95% CI: {results[label]['95_ci']['bal_acc'][0]:.4f}-{results[label]['95_ci']['bal_acc'][1]:.4f})"
              f"  F1: {point_f1:.4f} (95% CI: {results[label]['95_ci']['f1'][0]:.4f}-{results[label]['95_ci']['f1'][1]:.4f})")
    print(f"  [shared, threshold-independent] AUROC: {point_auc:.4f} (95% CI: {ci(boot_auc)[0]:.4f}-{ci(boot_auc)[1]:.4f})"
          f"  AUPRC: {point_auprc:.4f} (95% CI: {ci(boot_auprc)[0]:.4f}-{ci(boot_auprc)[1]:.4f})")

    return results


def run_bootstrap(task, probs, labels, num_classes, compute_fn, **kwargs):
    print(f"\nRunning {N_BOOTSTRAP} stratified bootstrap iterations for {task}...")
    boot_bal_acc, boot_auc, boot_auprc, boot_f1 = [], [], [], []

    point_bal_acc, point_auc, point_auprc, point_f1 = compute_fn(probs, labels, **kwargs)

    for i in range(N_BOOTSTRAP):
        idx = stratified_bootstrap_indices(labels, num_classes)
        b_probs, b_labels = probs[idx], labels[idx]
        try:
            bal_acc, auc, auprc, f1 = compute_fn(b_probs, b_labels, **kwargs)
            if not (np.isnan(auc) or np.isnan(auprc)):
                boot_bal_acc.append(bal_acc)
                boot_auc.append(auc)
                boot_auprc.append(auprc)
                boot_f1.append(f1)
        except Exception:
            continue

    def ci(arr):
        arr = np.array(arr)
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    results = {
        "point_estimate": {
            "bal_acc": float(point_bal_acc), "auc": float(point_auc),
            "auprc": float(point_auprc), "f1": float(point_f1)
        },
        "95_ci": {
            "bal_acc": [float(x) for x in ci(boot_bal_acc)],
            "auc": [float(x) for x in ci(boot_auc)],
            "auprc": [float(x) for x in ci(boot_auprc)],
            "f1": [float(x) for x in ci(boot_f1)],
        },
        "raw_bootstrap": {
            "bal_acc": [float(x) for x in boot_bal_acc],
            "auc": [float(x) for x in boot_auc],
            "auprc": [float(x) for x in boot_auprc],
            "f1": [float(x) for x in boot_f1],
        },
        "n_bootstrap_valid": len(boot_auc),
        "n_bootstrap_requested": N_BOOTSTRAP
    }

    print(f"  Balanced Accuracy: {point_bal_acc:.4f} (95% CI: {results['95_ci']['bal_acc'][0]:.4f}-{results['95_ci']['bal_acc'][1]:.4f})")
    print(f"  AUROC:             {point_auc:.4f} (95% CI: {results['95_ci']['auc'][0]:.4f}-{results['95_ci']['auc'][1]:.4f})")
    print(f"  AUPRC:              {point_auprc:.4f} (95% CI: {results['95_ci']['auprc'][0]:.4f}-{results['95_ci']['auprc'][1]:.4f})")
    print(f"  F1 (macro):         {point_f1:.4f} (95% CI: {results['95_ci']['f1'][0]:.4f}-{results['95_ci']['f1'][1]:.4f})")

    return results
# --- end verbatim block ---


def get_fusion_test_predictions(task, fc_dim, dropout, exclude_clinical_cols=None):
    """Runs inference ONCE on the test set using the trained fusion model
    (frozen ABMIL z ++ clinical features -> FusionClassifier). Mirrors
    bootstrap_ci.py's get_test_predictions, adapted for the two-part fusion
    input instead of a single embedding forward pass."""
    arch = TASK_ARCH[task]
    num_classes = arch["num_classes"]

    abmil_model = ABMIL(EMBED_DIM, arch["attn_dim"], arch["fc_dim"], num_classes, 0.0).to(DEVICE)
    abmil_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, arch["model_file"]), map_location=DEVICE))
    abmil_model.eval()

    clinical_lookup, clinical_cols = load_clinical_features(exclude_cols=exclude_clinical_cols)
    input_dim = EMBED_DIM + len(clinical_cols)

    # Load the SAME train-derived z-normalization stats saved during
    # fusion_train.py's run -- must match exactly, not be recomputed here,
    # since these test patients must never influence their own normalization.
    norm_stats_path = os.path.join(FUSION_RESULTS_DIR, f"z_norm_stats_{task}.npz")
    norm_stats = np.load(norm_stats_path)
    z_mean, z_std = norm_stats["mean"], norm_stats["std"]

    fusion_ckpt_suffix = ("_excl_" + "_".join(exclude_clinical_cols)) if exclude_clinical_cols else ""
    fusion_ckpt_path = os.path.join(FUSION_RESULTS_DIR, f"best_fusion_{task}{fusion_ckpt_suffix}.pt")
    fusion_model = FusionClassifier(input_dim, fc_dim, num_classes, dropout).to(DEVICE)
    fusion_model.load_state_dict(torch.load(fusion_ckpt_path, map_location=DEVICE))
    fusion_model.eval()

    splits_df = pd.read_csv(SPLITS_PATH)
    if task == "multiclass":
        label_map = {"BRAF_V600E": 0, "RAS": 1, "Other": 2}
        splits_df = splits_df[splits_df["mutation_label"].isin(label_map)].reset_index(drop=True)
        splits_df["label"] = splits_df["mutation_label"].map(label_map)
    else:
        splits_df["label"] = splits_df["RET"]

    test_df = splits_df[splits_df["split"] == "test"].copy()
    embedding_index = build_embedding_index()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            patient = row["patient"]
            emb_file = embedding_index.get(patient)
            if emb_file is None or patient not in clinical_lookup:
                continue
            with h5py.File(emb_file, "r") as f:
                features = torch.tensor(f["features"][:], dtype=torch.float32).to(DEVICE)
            _, _, z = abmil_model(features)
            z = z.squeeze().cpu().numpy()
            z = (z - z_mean) / z_std
            x = np.concatenate([z, clinical_lookup[patient]])
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            logits = fusion_model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
            all_labels.append(row["label"])

    return np.array(all_probs), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["multiclass", "ret_binary"], required=True)
    parser.add_argument("--fc_dim", type=int, required=True,
                         help="Must match the winning config's fc_dim from fusion_hparam_search.py")
    parser.add_argument("--dropout", type=float, default=0.0,
                         help="Only affects layer construction, not loaded weights (Dropout has no "
                              "learnable params) -- safe to leave at default.")
    parser.add_argument("--exclude_clinical_cols", type=str, default=None,
                         help="Must match whatever was used for the corresponding fusion_train.py run "
                              "(e.g. for the site-confound diagnostic checkpoint).")
    args = parser.parse_args()

    exclude_cols = args.exclude_clinical_cols.split(",") if args.exclude_clinical_cols else None

    print(f"Loading fusion model and computing test predictions for {args.task}...")
    probs, labels = get_fusion_test_predictions(args.task, args.fc_dim, args.dropout, exclude_cols)
    print(f"Test set: {len(labels)} patients")

    suffix = ("_excl_" + "_".join(exclude_cols)) if exclude_cols else ""
    all_results = {}

    if args.task == "multiclass":
        print("=" * 60)
        print("Fusion -- Aim a) Multiclass")
        print("=" * 60)
        results = run_bootstrap("fusion_multiclass", probs, labels, 3, compute_metrics_multiclass)
        all_results["multiclass"] = results
    else:
        # Read the fusion model's OWN validation-selected threshold -- NOT
        # the base model's 0.13, which was calibrated for a different model.
        fusion_results_path = os.path.join(FUSION_RESULTS_DIR, f"fusion_results_ret_binary{suffix}.json")
        with open(fusion_results_path) as f:
            fusion_results = json.load(f)
        threshold = fusion_results["threshold_calibration"]["threshold"]
        print(f"Using fusion model's own validation-selected threshold: {threshold}")

        print("=" * 60)
        print("Fusion -- Aim b) RET binary")
        print("=" * 60)

        results_by_threshold = run_bootstrap_binary(
            "fusion_ret_binary", probs, labels,
            thresholds={"default_0.5": 0.5, f"optimal_{threshold}": threshold}
        )
        all_results["ret_binary_default_threshold"] = results_by_threshold["default_0.5"]
        all_results["ret_binary_optimal_threshold"] = results_by_threshold[f"optimal_{threshold}"]

    out_path = os.path.join(FUSION_RESULTS_DIR, f"fusion_bootstrap_ci_{args.task}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
