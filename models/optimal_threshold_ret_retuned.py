import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
import json
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    average_precision_score, classification_report
)

EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
SPLITS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"
RESULTS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2"
DEVICE = "cpu"
EMBED_DIM = 1536
ATTN_DIM, FC_DIM = 512, 256  # new RET binary architecture


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
    def __init__(self, embed_dim=1536, attn_dim=512, fc_dim=256, num_classes=2, dropout=0.0):
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


def get_probs_and_labels(model, df):
    probs, labels = [], []
    for _, row in df.iterrows():
        patient = row["patient"]
        true_label = row["RET"]
        emb_file = get_embedding_file(patient)
        if emb_file is None:
            continue
        with h5py.File(emb_file, "r") as f:
            features = torch.tensor(f["features"][:], dtype=torch.float32)
        with torch.no_grad():
            logits, A = model(features)
            p = torch.softmax(logits, dim=1).numpy()[0]
        probs.append(p[1])
        labels.append(true_label)
    return np.array(probs), np.array(labels)


model = ABMIL(EMBED_DIM, ATTN_DIM, FC_DIM, 2, 0.0).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_ret_binary_v2_retuned.pt"), map_location=DEVICE))
model.eval()

splits_df = pd.read_csv(SPLITS_PATH)
val_df = splits_df[splits_df["split"] == "val"].copy()
test_df = splits_df[splits_df["split"] == "test"].copy()

print("Computing probabilities on VALIDATION set (to select threshold)...")
val_probs, val_labels = get_probs_and_labels(model, val_df)
print("Computing probabilities on TEST set (final evaluation only)...")
test_probs, test_labels = get_probs_and_labels(model, test_df)

best_thresh, best_bal_acc = 0.5, 0
for t in np.arange(0.01, 0.55, 0.01):
    preds = (val_probs >= t).astype(int)
    bal_acc = balanced_accuracy_score(val_labels, preds)
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_thresh = t

print(f"\nOptimal threshold (max val balanced accuracy): {best_thresh:.2f} (val BalAcc={best_bal_acc:.4f})")

test_preds = (test_probs >= best_thresh).astype(int)
test_bal_acc = balanced_accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds, average="macro")
test_auc = roc_auc_score(test_labels, test_probs)
test_auprc = average_precision_score(test_labels, test_probs)

print(f"\n=== FINAL TEST RESULTS (v2_retuned, threshold={best_thresh:.2f}) ===")
print(f"Balanced Accuracy: {test_bal_acc:.4f}")
print(f"AUROC: {test_auc:.4f}")
print(f"AUPRC: {test_auprc:.4f}")
print(f"F1 (macro): {test_f1:.4f}")
print(classification_report(test_labels, test_preds, target_names=["RET_negative", "RET_positive"]))

results = {
    "task": "ret_binary", "model": "ABMIL_v2_retuned_optimal_threshold",
    "threshold_selected_on": "validation_set", "optimal_threshold": float(best_thresh),
    "test": {"bal_acc": float(test_bal_acc), "auc": float(test_auc), "auprc": float(test_auprc), "f1": float(test_f1)}
}
with open(os.path.join(RESULTS_DIR, "abmil_ret_binary_v2_retuned_optimal_threshold.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results_v2/abmil_ret_binary_v2_retuned_optimal_threshold.json")
