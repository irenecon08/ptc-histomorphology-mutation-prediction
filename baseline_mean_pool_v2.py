import os
import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, classification_report
import json

# === CONFIG (pointing at CORRECTED embeddings) ===
EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
SPLITS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"
OUTPUT_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_embedding_file(patient):
    for f in os.listdir(EMBEDDINGS_DIR):
        if f.startswith(patient) and f.endswith(".h5"):
            return os.path.join(EMBEDDINGS_DIR, f)
    return None


def mean_pool_embedding(emb_file):
    with h5py.File(emb_file, "r") as f:
        features = f["features"][:]
    return features.mean(axis=0)


def build_feature_matrix(splits_df, split, task="multiclass"):
    data = splits_df[splits_df["split"] == split].reset_index(drop=True)

    if task == "multiclass":
        label_map = {"BRAF_V600E": 0, "RAS": 1, "Other": 2}
        data = data[data["mutation_label"].isin(label_map)]
        labels = data["mutation_label"].map(label_map).values
    elif task == "ret_binary":
        labels = data["RET"].values

    X, y, missing = [], [], []
    for patient, label in zip(data["patient"].values, labels):
        emb_file = get_embedding_file(patient)
        if emb_file is None:
            missing.append(patient)
            continue
        X.append(mean_pool_embedding(emb_file))
        y.append(label)

    if missing:
        print(f"  WARNING: {len(missing)} patients missing embeddings: {missing}")

    return np.array(X), np.array(y)


def evaluate(y_true, y_pred, y_prob, num_classes):
    acc = (y_true == y_pred).mean()
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            auprc = average_precision_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            auprc = np.mean([
                average_precision_score((np.array(y_true) == c).astype(int), y_prob[:, c])
                for c in range(num_classes)
            ])
    except Exception:
        auc, auprc = 0.0, 0.0

    return acc, bal_acc, auc, auprc, f1


def run_baseline(task, num_classes, target_names, class_weight=None):
    print(f"\n{'='*50}")
    print(f"BASELINE v2 (corrected embeddings): {task}")
    print(f"{'='*50}")

    splits_df = pd.read_csv(SPLITS_PATH)

    print("Building train features...")
    X_train, y_train = build_feature_matrix(splits_df, "train", task)
    print("Building val features...")
    X_val, y_val = build_feature_matrix(splits_df, "val", task)
    print("Building test features...")
    X_test, y_test = build_feature_matrix(splits_df, "test", task)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    clf = LogisticRegression(
        max_iter=2000,
        class_weight=class_weight,
        multi_class="ovr" if num_classes > 2 else "auto"
    )
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    val_prob = clf.predict_proba(X_val)
    val_acc, val_bal_acc, val_auc, val_auprc, val_f1 = evaluate(y_val, val_pred, val_prob, num_classes)
    print(f"\nVal  | Acc: {val_acc:.4f} BalAcc: {val_bal_acc:.4f} AUC: {val_auc:.4f} AUPRC: {val_auprc:.4f} F1: {val_f1:.4f}")

    test_pred = clf.predict(X_test)
    test_prob = clf.predict_proba(X_test)
    test_acc, test_bal_acc, test_auc, test_auprc, test_f1 = evaluate(y_test, test_pred, test_prob, num_classes)
    print(f"Test | Acc: {test_acc:.4f} BalAcc: {test_bal_acc:.4f} AUC: {test_auc:.4f} AUPRC: {test_auprc:.4f} F1: {test_f1:.4f}")

    print(f"\nClassification Report (Test):")
    print(classification_report(y_test, test_pred, target_names=target_names))

    results = {
        "task": task,
        "model": "mean_pool_logreg_v2_corrected",
        "val": {"acc": val_acc, "bal_acc": val_bal_acc, "auc": val_auc, "auprc": val_auprc, "f1": val_f1},
        "test": {"acc": test_acc, "bal_acc": test_bal_acc, "auc": test_auc, "auprc": test_auprc, "f1": test_f1}
    }
    with open(os.path.join(OUTPUT_DIR, f"baseline_results_{task}_v2.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_baseline(
        task="multiclass",
        num_classes=3,
        target_names=["BRAF_V600E", "RAS", "Other"],
        class_weight=None
    )

    run_baseline(
        task="ret_binary",
        num_classes=2,
        target_names=["RET_negative", "RET_positive"],
        class_weight="balanced"
    )

    print("\nBaseline v2 (corrected) experiments complete!")
