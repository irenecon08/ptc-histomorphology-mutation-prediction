import os
import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, classification_report
)

EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
SPLITS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"


def get_embedding_file(patient):
    for f in os.listdir(EMBEDDINGS_DIR):
        if f.startswith(patient) and f.endswith(".h5"):
            return os.path.join(EMBEDDINGS_DIR, f)
    return None


def mean_pool_embedding(emb_file):
    with h5py.File(emb_file, "r") as f:
        features = f["features"][:]
    return features.mean(axis=0)


def build_feature_matrix(splits_df, split):
    data = splits_df[splits_df["split"] == split].reset_index(drop=True)
    labels = data["RET"].values
    X, y = [], []
    for patient, label in zip(data["patient"].values, labels):
        emb_file = get_embedding_file(patient)
        if emb_file is None:
            continue
        X.append(mean_pool_embedding(emb_file))
        y.append(label)
    return np.array(X), np.array(y)


print("Building train/val/test feature matrices (RET binary, baseline mean-pooling)...")
splits_df = pd.read_csv(SPLITS_PATH)
X_train, y_train = build_feature_matrix(splits_df, "train")
X_val, y_val = build_feature_matrix(splits_df, "val")
X_test, y_test = build_feature_matrix(splits_df, "test")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

print("\nTraining logistic regression (class_weight='balanced'), same as original baseline...")
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train, y_train)

val_probs = clf.predict_proba(X_val)[:, 1]
test_probs = clf.predict_proba(X_test)[:, 1]

# === Check DEFAULT 0.5 threshold (what was originally reported) ===
print("\n=== DEFAULT THRESHOLD (0.5) — as originally reported ===")
test_preds_default = (test_probs >= 0.5).astype(int)
bal_acc_default = balanced_accuracy_score(y_test, test_preds_default)
f1_default = f1_score(y_test, test_preds_default, average="macro")
auc = roc_auc_score(y_test, test_probs)
auprc = average_precision_score(y_test, test_probs)
print(f"Balanced Accuracy: {bal_acc_default:.4f}")
print(f"AUROC: {auc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"F1: {f1_default:.4f}")
print(classification_report(y_test, test_preds_default, target_names=["RET_negative", "RET_positive"]))

# === Search for OPTIMAL threshold on VALIDATION set only ===
print("\n=== SEARCHING FOR OPTIMAL THRESHOLD (validation set only) ===")
best_thresh, best_bal_acc = 0.5, 0
for t in np.arange(0.01, 0.99, 0.01):
    preds = (val_probs >= t).astype(int)
    bal_acc = balanced_accuracy_score(y_val, preds)
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_thresh = t
print(f"Optimal threshold: {best_thresh:.2f} (val BalAcc={best_bal_acc:.4f})")

print(f"\n=== RESULTS AT OPTIMAL THRESHOLD ({best_thresh:.2f}) — applied once to test ===")
test_preds_opt = (test_probs >= best_thresh).astype(int)
bal_acc_opt = balanced_accuracy_score(y_test, test_preds_opt)
f1_opt = f1_score(y_test, test_preds_opt, average="macro")
print(f"Balanced Accuracy: {bal_acc_opt:.4f}")
print(f"AUROC: {auc:.4f} (unchanged - threshold independent)")
print(f"AUPRC: {auprc:.4f} (unchanged - threshold independent)")
print(f"F1: {f1_opt:.4f}")
print(classification_report(y_test, test_preds_opt, target_names=["RET_negative", "RET_positive"]))

print("\n=== SUMMARY: was the originally-reported baseline threshold-miscalibrated? ===")
print(f"Default (0.5):  BalAcc={bal_acc_default:.4f}, F1={f1_default:.4f}")
print(f"Optimal ({best_thresh:.2f}): BalAcc={bal_acc_opt:.4f}, F1={f1_opt:.4f}")
if abs(bal_acc_opt - bal_acc_default) > 0.02:
    print(">>> MEANINGFUL DIFFERENCE FOUND. Original baseline report may need updating.")
else:
    print(">>> No meaningful difference. Original baseline report (default threshold) stands as reported.")
