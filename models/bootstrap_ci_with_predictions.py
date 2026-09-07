import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
import json
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
)

EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
SPLITS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"
RESULTS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2"
DEVICE = "cpu"
EMBED_DIM = 1536
N_BOOTSTRAP = 2000
RANDOM_SEED = 42
RET_THRESHOLD = 0.13  # validation-selected optimal threshold for the retuned RET model

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
    """Run inference ONCE on the full test set. Returns probs (N, num_classes), true labels (N,), and patient IDs (N,)."""
    all_probs, all_labels, all_patients = [], [], []
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
        all_patients.append(patient)
    return np.array(all_probs), np.array(all_labels), np.array(all_patients)


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
    """Resample WITHIN each class separately, preserving class proportions in every bootstrap sample."""
    indices = []
    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        if len(class_idx) == 0:
            continue
        sampled = np.random.choice(class_idx, size=len(class_idx), replace=True)
        indices.extend(sampled)
    return np.array(indices)


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

    # --- raw per-iteration bootstrap arrays (added previously) ---
    safe_name = task.replace(" ", "_").replace("(", "").replace(")", "")
    raw_path = os.path.join(RESULTS_DIR, f"bootstrap_raw_{safe_name}.npz")
    np.savez(
        raw_path,
        bal_acc=np.array(boot_bal_acc),
        auc=np.array(boot_auc),
        auprc=np.array(boot_auprc),
        f1=np.array(boot_f1),
    )
    print(f"  Saved raw bootstrap arrays: {raw_path}")
    # --- end ---

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
        "n_bootstrap_valid": len(boot_auc),
        "n_bootstrap_requested": N_BOOTSTRAP
    }

    print(f"  Balanced Accuracy: {point_bal_acc:.4f} (95% CI: {results['95_ci']['bal_acc'][0]:.4f}-{results['95_ci']['bal_acc'][1]:.4f})")
    print(f"  AUROC:             {point_auc:.4f} (95% CI: {results['95_ci']['auc'][0]:.4f}-{results['95_ci']['auc'][1]:.4f})")
    print(f"  AUPRC:              {point_auprc:.4f} (95% CI: {results['95_ci']['auprc'][0]:.4f}-{results['95_ci']['auprc'][1]:.4f})")
    print(f"  F1 (macro):         {point_f1:.4f} (95% CI: {results['95_ci']['f1'][0]:.4f}-{results['95_ci']['f1'][1]:.4f})")

    return results


# --- NEW: save raw per-patient test-set predictions (for confusion matrices, ROC/PR curves) ---
def save_predictions_csv(path, patients, labels, probs, class_names):
    """One row per test patient: true label plus predicted probability for each class."""
    df = pd.DataFrame({"patient": patients, "true_label": labels})
    for i, name in enumerate(class_names):
        df[f"prob_{name}"] = probs[:, i]
    df.to_csv(path, index=False)
    print(f"  Saved per-patient predictions: {path}")
# --- end NEW ---


if __name__ == "__main__":
    splits_df = pd.read_csv(SPLITS_PATH)
    test_df_full = splits_df[splits_df["split"] == "test"].copy()

    all_results = {}

    # === Aim a) Multiclass ===
    print("="*60)
    print("Aim a) Multiclass")
    print("="*60)
    label_map = {"BRAF_V600E": 0, "RAS": 1, "Other": 2}
    test_df_mc = test_df_full[test_df_full["mutation_label"].isin(label_map)].copy()
    test_df_mc["label"] = test_df_mc["mutation_label"].map(label_map)

    model_mc = ABMIL(EMBED_DIM, 512, 512, 3, 0.0).to(DEVICE)
    model_mc.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_multiclass_v2_retuned.pt"), map_location=DEVICE))
    model_mc.eval()

    probs_mc, labels_mc, patients_mc = get_test_predictions(model_mc, test_df_mc, 3)
    save_predictions_csv(
        os.path.join(RESULTS_DIR, "test_predictions_abmil_multiclass.csv"),
        patients_mc, labels_mc, probs_mc, ["BRAF_V600E", "RAS", "Other"]
    )
    results_mc = run_bootstrap("multiclass", probs_mc, labels_mc, 3, compute_metrics_multiclass)
    all_results["multiclass"] = results_mc

    # === Aim b) RET binary ===
    print("\n" + "="*60)
    print("Aim b) RET binary")
    print("="*60)
    test_df_ret = test_df_full.copy()
    test_df_ret["label"] = test_df_ret["RET"]

    model_ret = ABMIL(EMBED_DIM, 512, 256, 2, 0.0).to(DEVICE)
    model_ret.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_ret_binary_v2_retuned.pt"), map_location=DEVICE))
    model_ret.eval()

    probs_ret, labels_ret, patients_ret = get_test_predictions(model_ret, test_df_ret, 2)
    save_predictions_csv(
        os.path.join(RESULTS_DIR, "test_predictions_abmil_ret_binary.csv"),
        patients_ret, labels_ret, probs_ret, ["RET_negative", "RET_positive"]
    )

    print("\n--- At DEFAULT threshold (0.5) ---")
    results_ret_default = run_bootstrap("ret_binary (default 0.5)", probs_ret, labels_ret, 2, compute_metrics_binary, threshold=0.5)
    all_results["ret_binary_default_threshold"] = results_ret_default

    print("\n--- At OPTIMAL threshold (0.13) ---")
    results_ret_optimal = run_bootstrap("ret_binary (optimal 0.13)", probs_ret, labels_ret, 2, compute_metrics_binary, threshold=RET_THRESHOLD)
    all_results["ret_binary_optimal_threshold"] = results_ret_optimal

    with open(os.path.join(RESULTS_DIR, "bootstrap_ci_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results_v2/bootstrap_ci_results.json")
