import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
import json
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score

EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
SPLITS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"
RESULTS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2"
DEVICE = "cpu"
EMBED_DIM = 1536
N_BOOTSTRAP = 2000
RANDOM_SEED = 42
RET_THRESHOLD = 0.13

TASK_ARCH = {
    "multiclass": {"attn_dim": 512, "fc_dim": 512, "model_file": "best_multiclass_v2_retuned.pt"},
    "ret_binary": {"attn_dim": 512, "fc_dim": 256, "model_file": "best_ret_binary_v2_retuned.pt"},
}

np.random.seed(RANDOM_SEED)


class GatedAttention(nn.Module):
    def __init__(self, embed_dim=1536, hidden_dim=512, dropout=0.0):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Sigmoid())
        self.w = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        A = self.w(self.drop(self.V(x) * self.U(x)))
        A_softmax = torch.softmax(A, dim=0)
        return torch.mm(A_softmax.T, x), A_softmax


class ABMIL(nn.Module):
    def __init__(self, embed_dim=1536, attn_dim=512, fc_dim=256, num_classes=3, dropout=0.0):
        super().__init__()
        self.attention = GatedAttention(embed_dim, attn_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, fc_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(fc_dim, num_classes)
        )

    def forward(self, x):
        z, A = self.attention(x)
        return self.classifier(z), A


def get_embedding_file(patient):
    for f in os.listdir(EMBEDDINGS_DIR):
        if f.startswith(patient) and f.endswith(".h5"):
            return os.path.join(EMBEDDINGS_DIR, f)
    return None


def get_test_predictions(model, test_df, num_classes):
    all_probs, all_labels = [], []
    for _, row in test_df.iterrows():
        patient = row["patient"]
        emb_file = get_embedding_file(patient)
        if emb_file is None:
            continue
        with h5py.File(emb_file, "r") as f:
            features = torch.tensor(f["features"][:], dtype=torch.float32)
        with torch.no_grad():
            logits, _ = model(features)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        all_probs.append(probs)
        all_labels.append(row["label"])
    return np.array(all_probs), np.array(all_labels)


def stratified_bootstrap_indices(labels, n_classes):
    indices = []
    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        if len(class_idx) == 0:
            continue
        sampled = np.random.choice(class_idx, size=len(class_idx), replace=True)
        indices.extend(sampled)
    return np.array(indices)


def generate_all_bootstrap_indices(labels, n_classes, n_bootstrap):
    return [stratified_bootstrap_indices(labels, n_classes) for _ in range(n_bootstrap)]


def ci(arr):
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def compute_full_ci_multiclass(probs, labels, bootstrap_indices, num_classes=3):
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
    test_df_full = splits_df[splits_df["split"] == "test"].copy()
    all_results = {}

    print("="*60 + "\nAim a) Multiclass (final retuned model)\n" + "="*60)
    label_map = {"BRAF_V600E": 0, "RAS": 1, "Other": 2}
    test_df_mc = test_df_full[test_df_full["mutation_label"].isin(label_map)].copy()
    test_df_mc["label"] = test_df_mc["mutation_label"].map(label_map)

    arch_mc = TASK_ARCH["multiclass"]
    model_mc = ABMIL(EMBED_DIM, arch_mc["attn_dim"], arch_mc["fc_dim"], 3, 0.0).to(DEVICE)
    model_mc.load_state_dict(torch.load(os.path.join(RESULTS_DIR, arch_mc["model_file"]), map_location=DEVICE))
    model_mc.eval()

    probs_mc, labels_mc = get_test_predictions(model_mc, test_df_mc, 3)
    print(f"Generating {N_BOOTSTRAP} shared bootstrap resamples...")
    bootstrap_idx_mc = generate_all_bootstrap_indices(labels_mc, 3, N_BOOTSTRAP)
    results_mc = compute_full_ci_multiclass(probs_mc, labels_mc, bootstrap_idx_mc, num_classes=3)
    all_results["multiclass"] = results_mc
    print(f"  BalAcc: {results_mc['point_estimate']['bal_acc']:.4f} "
          f"(95% CI: {results_mc['95_ci']['bal_acc'][0]:.4f}-{results_mc['95_ci']['bal_acc'][1]:.4f})")
    print(f"  AUROC:  {results_mc['point_estimate']['auc']:.4f} "
          f"(95% CI: {results_mc['95_ci']['auc'][0]:.4f}-{results_mc['95_ci']['auc'][1]:.4f})")
    print(f"  AUPRC:  {results_mc['point_estimate']['auprc']:.4f} "
          f"(95% CI: {results_mc['95_ci']['auprc'][0]:.4f}-{results_mc['95_ci']['auprc'][1]:.4f})")
    print(f"  F1:     {results_mc['point_estimate']['f1']:.4f} "
          f"(95% CI: {results_mc['95_ci']['f1'][0]:.4f}-{results_mc['95_ci']['f1'][1]:.4f})")

    print("\n" + "="*60 + "\nAim b) RET binary (final retuned model)\n" + "="*60)
    test_df_ret = test_df_full.copy()
    test_df_ret["label"] = test_df_ret["RET"]

    arch_ret = TASK_ARCH["ret_binary"]
    model_ret = ABMIL(EMBED_DIM, arch_ret["attn_dim"], arch_ret["fc_dim"], 2, 0.0).to(DEVICE)
    model_ret.load_state_dict(torch.load(os.path.join(RESULTS_DIR, arch_ret["model_file"]), map_location=DEVICE))
    model_ret.eval()

    probs_ret, labels_ret = get_test_predictions(model_ret, test_df_ret, 2)
    print(f"Generating {N_BOOTSTRAP} shared bootstrap resamples (used for ALL RET metrics/thresholds)...")
    bootstrap_idx_ret = generate_all_bootstrap_indices(labels_ret, 2, N_BOOTSTRAP)

    print("\n--- Threshold-independent metrics (computed ONCE) ---")
    ret_indep = compute_threshold_independent_ci_binary(probs_ret, labels_ret, bootstrap_idx_ret)
    print(f"  AUROC: {ret_indep['point_estimate']['auc']:.4f} "
          f"(95% CI: {ret_indep['95_ci']['auc'][0]:.4f}-{ret_indep['95_ci']['auc'][1]:.4f})")
    print(f"  AUPRC: {ret_indep['point_estimate']['auprc']:.4f} "
          f"(95% CI: {ret_indep['95_ci']['auprc'][0]:.4f}-{ret_indep['95_ci']['auprc'][1]:.4f})")

    print("\n--- At DEFAULT threshold (0.5) ---")
    ret_default = compute_threshold_dependent_ci_binary(probs_ret, labels_ret, bootstrap_idx_ret, 0.5)
    print(f"  BalAcc: {ret_default['point_estimate']['bal_acc']:.4f} "
          f"(95% CI: {ret_default['95_ci']['bal_acc'][0]:.4f}-{ret_default['95_ci']['bal_acc'][1]:.4f})")
    print(f"  F1:     {ret_default['point_estimate']['f1']:.4f} "
          f"(95% CI: {ret_default['95_ci']['f1'][0]:.4f}-{ret_default['95_ci']['f1'][1]:.4f})")

    print(f"\n--- At OPTIMAL threshold ({RET_THRESHOLD}) ---")
    ret_optimal = compute_threshold_dependent_ci_binary(probs_ret, labels_ret, bootstrap_idx_ret, RET_THRESHOLD)
    print(f"  BalAcc: {ret_optimal['point_estimate']['bal_acc']:.4f} "
          f"(95% CI: {ret_optimal['95_ci']['bal_acc'][0]:.4f}-{ret_optimal['95_ci']['bal_acc'][1]:.4f})")
    print(f"  F1:     {ret_optimal['point_estimate']['f1']:.4f} "
          f"(95% CI: {ret_optimal['95_ci']['f1'][0]:.4f}-{ret_optimal['95_ci']['f1'][1]:.4f})")

    all_results["ret_binary"] = {
        "threshold_independent": ret_indep,
        "default_threshold": ret_default,
        "optimal_threshold": ret_optimal,
    }

    with open(os.path.join(RESULTS_DIR, "bootstrap_ci_results_v2.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results_v2/bootstrap_ci_results_v2.json")
