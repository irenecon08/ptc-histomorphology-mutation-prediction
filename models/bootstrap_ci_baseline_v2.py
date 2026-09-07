import os
import numpy as np
import pandas as pd
import h5py
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score

EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
SPLITS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"
RESULTS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2"
N_BOOTSTRAP = 2000
RANDOM_SEED = 42
RET_OPTIMAL_THRESHOLD = 0.11

np.random.seed(RANDOM_SEED)


def get_embedding_file(patient):
    for f in os.listdir(EMBEDDINGS_DIR):
        if f.startswith(patient) and f.endswith(".h5"):
            return os.path.join(EMBEDDINGS_DIR, f)
    return None


def mean_pool_embedding(emb_file):
    with h5py.File(emb_file, "r") as f:
        features = f["features"][:]
    return features.mean(axis=0)


def build_feature_matrix(splits_df, split, task):
    data = splits_df[splits_df["split"] == split].reset_index(drop=True)
    if task == "multiclass":
        label_map = {"BRAF_V600E": 0, "RAS": 1, "Other": 2}
        data = data[data["mutation_label"].isin(label_map)]
        labels = data["mutation_label"].map(label_map).values
    else:
        labels = data["RET"].values
    X, y = [], []
    for patient, label in zip(data["patient"].values, labels):
        emb_file = get_embedding_file(patient)
        if emb_file is None:
            continue
        X.append(mean_pool_embedding(emb_file))
        y.append(label)
    return np.array(X), np.array(y)


def stratified_bootstrap_indices(labels, n_classes):
    """Resample WITHIN each class separately, preserving class proportions."""
    indices = []
    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        if len(class_idx) == 0:
            continue
        sampled = np.random.choice(class_idx, size=len(class_idx), replace=True)
        indices.extend(sampled)
    return np.array(indices)


def generate_all_bootstrap_indices(labels, n_classes, n_bootstrap):
    """Generate ALL bootstrap resamples ONCE, to be reused across every metric
    and every threshold - ensures AUROC/AUPRC (threshold-independent) get exactly
    one consistent CI, and BalAcc/F1 at different thresholds are computed on the
    SAME underlying resamples for direct comparability."""
    return [stratified_bootstrap_indices(labels, n_classes) for _ in range(n_bootstrap)]


def ci(arr):
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def compute_full_ci_multiclass(probs, labels, bootstrap_indices, num_classes=3):
    """Threshold-free: computes CI for all 4 metrics using shared bootstrap indices."""
    preds = np.argmax(probs, axis=1)
    point_bal_acc = balanced_accuracy_score(labels, preds)
    point_f1 = f1_score(labels, preds, average="macro")
    point_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    point_auprc = np.mean([
        average_precision_score((labels == c).astype(int), probs[:, c]) for c in range(num_classes)
    ])

    boot_bal_acc, boot_f1, boot_auc, boot_auprc = [], [], [], []
    for idx in bootstrap_indices:
        b_probs, b_labels = probs[idx], labels[idx]
        b_preds = np.argmax(b_probs, axis=1)
        try:
            boot_bal_acc.append(balanced_accuracy_score(b_labels, b_preds))
            boot_f1.append(f1_score(b_labels, b_preds, average="macro"))
            boot_auc.append(roc_auc_score(b_labels, b_probs, multi_class="ovr", average="macro"))
            boot_auprc.append(np.mean([
                average_precision_score((b_labels == c).astype(int), b_probs[:, c]) for c in range(num_classes)
            ]))
        except Exception:
            continue

    return {
        "point_estimate": {"bal_acc": float(point_bal_acc), "auc": float(point_auc),
                            "auprc": float(point_auprc), "f1": float(point_f1)},
        "95_ci": {"bal_acc": list(ci(boot_bal_acc)), "auc": list(ci(boot_auc)),
                  "auprc": list(ci(boot_auprc)), "f1": list(ci(boot_f1))},
        "n_bootstrap_valid": len(boot_auc)
    }


def compute_threshold_independent_ci_binary(probs, labels, bootstrap_indices):
    """AUROC/AUPRC only - computed ONCE, reused for any threshold."""
    pos_probs = probs[:, 1]
    point_auc = roc_auc_score(labels, pos_probs)
    point_auprc = average_precision_score(labels, pos_probs)

    boot_auc, boot_auprc = [], []
    for idx in bootstrap_indices:
        b_pos_probs, b_labels = pos_probs[idx], labels[idx]
        try:
            boot_auc.append(roc_auc_score(b_labels, b_pos_probs))
            boot_auprc.append(average_precision_score(b_labels, b_pos_probs))
        except Exception:
            continue

    return {
        "point_estimate": {"auc": float(point_auc), "auprc": float(point_auprc)},
        "95_ci": {"auc": list(ci(boot_auc)), "auprc": list(ci(boot_auprc))},
        "n_bootstrap_valid": len(boot_auc)
    }


def compute_threshold_dependent_ci_binary(probs, labels, bootstrap_indices, threshold):
    """BalAcc/F1 only, at a SPECIFIC threshold - uses the SAME bootstrap indices
    as the threshold-independent metrics, so results are directly comparable
    across thresholds and consistent with the AUROC/AUPRC CI."""
    pos_probs = probs[:, 1]
    preds = (pos_probs >= threshold).astype(int)
    point_bal_acc = balanced_accuracy_score(labels, preds)
    point_f1 = f1_score(labels, preds, average="macro")

    boot_bal_acc, boot_f1 = [], []
    for idx in bootstrap_indices:
        b_pos_probs, b_labels = pos_probs[idx], labels[idx]
        b_preds = (b_pos_probs >= threshold).astype(int)
        try:
            boot_bal_acc.append(balanced_accuracy_score(b_labels, b_preds))
            boot_f1.append(f1_score(b_labels, b_preds, average="macro"))
        except Exception:
            continue

    return {
        "threshold": threshold,
        "point_estimate": {"bal_acc": float(point_bal_acc), "f1": float(point_f1)},
        "95_ci": {"bal_acc": list(ci(boot_bal_acc)), "f1": list(ci(boot_f1))},
        "n_bootstrap_valid": len(boot_bal_acc)
    }


if __name__ == "__main__":
    splits_df = pd.read_csv(SPLITS_PATH)
    all_results = {}

    print("="*60 + "\nBaseline Aim a) Multiclass\n" + "="*60)
    X_train, y_train = build_feature_matrix(splits_df, "train", "multiclass")
    X_test, y_test = build_feature_matrix(splits_df, "test", "multiclass")
    clf_mc = LogisticRegression(max_iter=2000, multi_class="ovr")
    clf_mc.fit(X_train, y_train)
    probs_mc = clf_mc.predict_proba(X_test)

    print(f"Generating {N_BOOTSTRAP} shared bootstrap resamples...")
    bootstrap_idx_mc = generate_all_bootstrap_indices(y_test, 3, N_BOOTSTRAP)
    results_mc = compute_full_ci_multiclass(probs_mc, y_test, bootstrap_idx_mc, num_classes=3)
    all_results["multiclass"] = results_mc
    print(f"  BalAcc: {results_mc['point_estimate']['bal_acc']:.4f} "
          f"(95% CI: {results_mc['95_ci']['bal_acc'][0]:.4f}-{results_mc['95_ci']['bal_acc'][1]:.4f})")
    print(f"  AUROC:  {results_mc['point_estimate']['auc']:.4f} "
          f"(95% CI: {results_mc['95_ci']['auc'][0]:.4f}-{results_mc['95_ci']['auc'][1]:.4f})")
    print(f"  AUPRC:  {results_mc['point_estimate']['auprc']:.4f} "
          f"(95% CI: {results_mc['95_ci']['auprc'][0]:.4f}-{results_mc['95_ci']['auprc'][1]:.4f})")
    print(f"  F1:     {results_mc['point_estimate']['f1']:.4f} "
          f"(95% CI: {results_mc['95_ci']['f1'][0]:.4f}-{results_mc['95_ci']['f1'][1]:.4f})")

    print("\n" + "="*60 + "\nBaseline Aim b) RET binary\n" + "="*60)
    X_train_r, y_train_r = build_feature_matrix(splits_df, "train", "ret_binary")
    X_test_r, y_test_r = build_feature_matrix(splits_df, "test", "ret_binary")
    clf_ret = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf_ret.fit(X_train_r, y_train_r)
    probs_ret = clf_ret.predict_proba(X_test_r)

    print(f"Generating {N_BOOTSTRAP} shared bootstrap resamples (used for ALL RET metrics/thresholds)...")
    bootstrap_idx_ret = generate_all_bootstrap_indices(y_test_r, 2, N_BOOTSTRAP)

    print("\n--- Threshold-independent metrics (computed ONCE) ---")
    ret_indep = compute_threshold_independent_ci_binary(probs_ret, y_test_r, bootstrap_idx_ret)
    print(f"  AUROC: {ret_indep['point_estimate']['auc']:.4f} "
          f"(95% CI: {ret_indep['95_ci']['auc'][0]:.4f}-{ret_indep['95_ci']['auc'][1]:.4f})")
    print(f"  AUPRC: {ret_indep['point_estimate']['auprc']:.4f} "
          f"(95% CI: {ret_indep['95_ci']['auprc'][0]:.4f}-{ret_indep['95_ci']['auprc'][1]:.4f})")

    print("\n--- At DEFAULT threshold (0.5) ---")
    ret_default = compute_threshold_dependent_ci_binary(probs_ret, y_test_r, bootstrap_idx_ret, 0.5)
    print(f"  BalAcc: {ret_default['point_estimate']['bal_acc']:.4f} "
          f"(95% CI: {ret_default['95_ci']['bal_acc'][0]:.4f}-{ret_default['95_ci']['bal_acc'][1]:.4f})")
    print(f"  F1:     {ret_default['point_estimate']['f1']:.4f} "
          f"(95% CI: {ret_default['95_ci']['f1'][0]:.4f}-{ret_default['95_ci']['f1'][1]:.4f})")

    print(f"\n--- At OPTIMAL threshold ({RET_OPTIMAL_THRESHOLD}) ---")
    ret_optimal = compute_threshold_dependent_ci_binary(probs_ret, y_test_r, bootstrap_idx_ret, RET_OPTIMAL_THRESHOLD)
    print(f"  BalAcc: {ret_optimal['point_estimate']['bal_acc']:.4f} "
          f"(95% CI: {ret_optimal['95_ci']['bal_acc'][0]:.4f}-{ret_optimal['95_ci']['bal_acc'][1]:.4f})")
    print(f"  F1:     {ret_optimal['point_estimate']['f1']:.4f} "
          f"(95% CI: {ret_optimal['95_ci']['f1'][0]:.4f}-{ret_optimal['95_ci']['f1'][1]:.4f})")

    all_results["ret_binary"] = {
        "threshold_independent": ret_indep,
        "default_threshold": ret_default,
        "optimal_threshold": ret_optimal,
    }

    with open(os.path.join(RESULTS_DIR, "bootstrap_ci_baseline_results_v2.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results_v2/bootstrap_ci_baseline_results_v2.json")
