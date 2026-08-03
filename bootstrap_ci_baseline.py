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
RET_THRESHOLD = 0.11  # validation-selected optimal threshold for baseline RET

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


def stratified_bootstrap_indices(labels, n_classes):
    indices = []
    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        if len(class_idx) == 0:
            continue
        sampled = np.random.choice(class_idx, size=len(class_idx), replace=True)
        indices.extend(sampled)
    return np.array(indices)


def run_bootstrap(task, probs, labels, num_classes, compute_fn, **kwargs):
    print(f"\nRunning {N_BOOTSTRAP} stratified bootstrap iterations for baseline {task}...")
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
        "point_estimate": {"bal_acc": float(point_bal_acc), "auc": float(point_auc),
                            "auprc": float(point_auprc), "f1": float(point_f1)},
        "95_ci": {"bal_acc": [float(x) for x in ci(boot_bal_acc)], "auc": [float(x) for x in ci(boot_auc)],
                  "auprc": [float(x) for x in ci(boot_auprc)], "f1": [float(x) for x in ci(boot_f1)]},
        "n_bootstrap_valid": len(boot_auc)
    }
    print(f"  Balanced Accuracy: {point_bal_acc:.4f} (95% CI: {results['95_ci']['bal_acc'][0]:.4f}-{results['95_ci']['bal_acc'][1]:.4f})")
    print(f"  AUROC:             {point_auc:.4f} (95% CI: {results['95_ci']['auc'][0]:.4f}-{results['95_ci']['auc'][1]:.4f})")
    print(f"  AUPRC:              {point_auprc:.4f} (95% CI: {results['95_ci']['auprc'][0]:.4f}-{results['95_ci']['auprc'][1]:.4f})")
    print(f"  F1 (macro):         {point_f1:.4f} (95% CI: {results['95_ci']['f1'][0]:.4f}-{results['95_ci']['f1'][1]:.4f})")
    return results


if __name__ == "__main__":
    splits_df = pd.read_csv(SPLITS_PATH)
    all_results = {}

    print("="*60 + "\nBaseline Aim a) Multiclass\n" + "="*60)
    X_train, y_train = build_feature_matrix(splits_df, "train", "multiclass")
    X_test, y_test = build_feature_matrix(splits_df, "test", "multiclass")
    clf_mc = LogisticRegression(max_iter=2000, multi_class="ovr")
    clf_mc.fit(X_train, y_train)
    probs_mc = clf_mc.predict_proba(X_test)
    all_results["multiclass"] = run_bootstrap("multiclass", probs_mc, y_test, 3, compute_metrics_multiclass)

    print("\n" + "="*60 + "\nBaseline Aim b) RET binary\n" + "="*60)
    X_train_r, y_train_r = build_feature_matrix(splits_df, "train", "ret_binary")
    X_test_r, y_test_r = build_feature_matrix(splits_df, "test", "ret_binary")
    clf_ret = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf_ret.fit(X_train_r, y_train_r)
    probs_ret = clf_ret.predict_proba(X_test_r)

    print("\n--- At DEFAULT threshold (0.5) ---")
    all_results["ret_binary_default_threshold"] = run_bootstrap(
        "ret_binary (default 0.5)", probs_ret, y_test_r, 2, compute_metrics_binary, threshold=0.5
    )

    print("\n--- At OPTIMAL threshold (0.11) ---")
    all_results["ret_binary_optimal_threshold"] = run_bootstrap(
        "ret_binary (optimal 0.11)", probs_ret, y_test_r, 2, compute_metrics_binary, threshold=RET_THRESHOLD
    )

    with open(os.path.join(RESULTS_DIR, "bootstrap_ci_baseline_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results_v2/bootstrap_ci_baseline_results.json")
