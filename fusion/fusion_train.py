"""
Multimodal fusion training: frozen ABMIL slide-level embeddings (from the
corrected, leakage-free retuned checkpoints) concatenated with clinical
features (9 variables: Diagnosis Age, Sex, Race Category, Ethnicity
Category, Overall Disease Stage, T Stage, N Stage, M Stage, Histological
Subtype -- see chat for the full derivation: molecular-burden, hypoxia,
survival/outcome, and treatment variables were explicitly excluded per
Petru's guidance), feeding a newly trained classifier head.

The ABMIL attention module and its original classifier are both frozen --
only the fusion classifier head is trained. This mirrors the "frozen
encoder, train the head" logic already used for UNI2-h in the base pipeline,
one level up.

Reuses the exact GatedAttention/ABMIL class definitions and slide-embedding
extraction logic from embedding_analysis_v3.py, so the z vectors here are
identical to what that script already produces -- this script just adds the
clinical concatenation and a training loop on top.

Usage:
    python fusion_train.py --task multiclass
    python fusion_train.py --task ret_binary
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              balanced_accuracy_score, classification_report)
import h5py

# === PATHS (match embedding_analysis_v3.py / train_abmil_v2_retrained.py) ===
EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
SPLITS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"
RESULTS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2"
CLINICAL_FEATURES_PATH = "/cs/student/project_msc/2025/aibh/iconstan/results_v2/clinical_features.csv"
FUSION_RESULTS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2/fusion"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 1536
NUM_EPOCHS = 100
PATIENCE = 10
WEIGHT_DECAY = 1e-3
FUSION_LR = 1e-3          # new head only, trained on fixed-length vectors -- can afford a higher lr than the attention training
FUSION_FC_DIM = 128
FUSION_DROPOUT = 0.2

# Per-task ABMIL architecture -- MUST match train_abmil_v2_retrained.py
# exactly, or load_state_dict will fail / silently mismatch.
TASK_ARCH = {
    "multiclass": {"attn_dim": 512, "fc_dim": 512, "model_file": "best_multiclass_v2_retuned.pt", "num_classes": 3},
    "ret_binary": {"attn_dim": 512, "fc_dim": 256, "model_file": "best_ret_binary_v2_retuned.pt", "num_classes": 2},
}

os.makedirs(FUSION_RESULTS_DIR, exist_ok=True)


# --- Reused verbatim from embedding_analysis_v3.py / train_abmil_v2_retrained.py ---
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
        z = torch.mm(A_softmax.T, x)
        return z, A_softmax


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
        logits = self.classifier(z)
        return logits, A, z
# --- end reused block ---


def build_embedding_index():
    index = {}
    for f in os.listdir(EMBEDDINGS_DIR):
        if f.endswith(".h5"):
            index[f[:12]] = os.path.join(EMBEDDINGS_DIR, f)
    return index


def load_clinical_features(exclude_cols=None):
    """Loads clinical_features.csv, returns (patient -> np.array) dict and
    the ordered list of feature column names (for reporting).

    exclude_cols: optional list of column names to drop before building the
    feature vector -- used for the site-confound diagnostic (see chat):
    Race/Ethnicity missingness tracks the institutional split almost
    perfectly (test = site EL, near-zero missing; train/val = other sites,
    ~20-27% missing), so a model could in principle learn 'high
    race_missing/ethnicity_missing' as a proxy for 'not site EL' rather
    than learning from genuine clinical signal. To check whether this is
    actually happening, run once with exclude_cols=None (full feature set,
    as instructed by Petru) and once with
    exclude_cols=['race_missing','ethnicity_missing'] (flags removed, the
    underlying imputed category values kept), and compare test performance.
    A large drop when the flags are removed would suggest the model was
    leaning on them; little change is reassuring.
    """
    df = pd.read_csv(CLINICAL_FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in ("patient", "split")]
    if exclude_cols:
        missing = set(exclude_cols) - set(feature_cols)
        if missing:
            print(f"  WARNING: exclude_cols not found in clinical_features.csv, ignoring: {missing}")
        feature_cols = [c for c in feature_cols if c not in exclude_cols]
    lookup = {row["patient"]: row[feature_cols].values.astype(np.float32) for _, row in df.iterrows()}
    return lookup, feature_cols


def extract_all_slide_embeddings(abmil_model, embedding_index, patients):
    """Runs the frozen ABMIL attention forward pass once per patient and
    caches the resulting z vector -- since the attention is frozen, this
    only needs to happen once, not every training epoch."""
    z_lookup = {}
    abmil_model.eval()
    with torch.no_grad():
        for i, patient in enumerate(patients):
            emb_file = embedding_index.get(patient)
            if emb_file is None:
                continue
            with h5py.File(emb_file, "r") as f:
                features = torch.tensor(f["features"][:], dtype=torch.float32).to(DEVICE)
            _, _, z = abmil_model(features)
            z_lookup[patient] = z.squeeze().cpu().numpy()
            if (i + 1) % 100 == 0:
                print(f"  Extracted embeddings for {i + 1}/{len(patients)} patients...")
    return z_lookup


class FusionDataset(Dataset):
    """Fixed-length fusion vectors (frozen ABMIL z ++ clinical features) --
    unlike the base ABMIL training, this does NOT need batch_size=1 or a
    variable-length bag collate function, since pooling already happened."""
    def __init__(self, patients, labels, z_lookup, clinical_lookup):
        valid_idx = [i for i, p in enumerate(patients)
                     if p in z_lookup and p in clinical_lookup]
        self.patients = [patients[i] for i in valid_idx]
        self.labels = [labels[i] for i in valid_idx]
        self.vectors = [np.concatenate([z_lookup[p], clinical_lookup[p]]) for p in self.patients]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        return torch.tensor(self.vectors[idx], dtype=torch.float32), self.labels[idx]


class FusionClassifier(nn.Module):
    """New classifier head only. Input = ABMIL z (1536-dim, frozen) ++
    clinical features. No attention mechanism here -- the attention pooling
    already happened upstream in the frozen ABMIL model."""
    def __init__(self, input_dim, fc_dim=FUSION_FC_DIM, num_classes=3, dropout=FUSION_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, fc_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(fc_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, num_classes):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())
    all_probs = np.array(all_probs)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
            auprc = average_precision_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
            auprc = np.mean([
                average_precision_score((np.array(all_labels) == c).astype(int), all_probs[:, c])
                for c in range(num_classes)
            ])
    except Exception:
        auc, auprc = 0.0, 0.0
    return bal_acc, auc, auprc, f1, all_preds, all_labels, all_probs


def find_optimal_threshold(labels, probs_pos):
    """Same convention as the base RET model: threshold selected on
    validation set only, by maximising balanced accuracy."""
    best_thresh, best_bal_acc = 0.5, 0.0
    for thresh in np.arange(0.01, 1.0, 0.01):
        preds = (probs_pos >= thresh).astype(int)
        bal_acc = balanced_accuracy_score(labels, preds)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thresh = thresh
    return best_thresh, best_bal_acc


def setup_task(task, exclude_clinical_cols=None):
    """One-time, expensive setup: load frozen ABMIL, extract slide
    embeddings, load clinical features, build per-split datasets and class
    weights. Call this ONCE per task, then reuse the returned dict across
    as many hyperparameter configs as you like via train_one_config() --
    avoids recomputing the frozen forward pass once per grid point.

    exclude_clinical_cols: passed through to load_clinical_features() --
    see that function's docstring for the site-confound diagnostic this
    supports (e.g. exclude_clinical_cols=['race_missing','ethnicity_missing'])."""
    arch = TASK_ARCH[task]
    num_classes = arch["num_classes"]

    print(f"\n{'='*60}\nSetting up task: {task}\n{'='*60}")

    abmil_model = ABMIL(EMBED_DIM, arch["attn_dim"], arch["fc_dim"], num_classes, 0.0).to(DEVICE)
    abmil_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, arch["model_file"]), map_location=DEVICE))
    for p in abmil_model.parameters():
        p.requires_grad = False
    abmil_model.eval()

    splits_df = pd.read_csv(SPLITS_PATH)
    if task == "multiclass":
        label_map = {"BRAF_V600E": 0, "RAS": 1, "Other": 2}
        splits_df = splits_df[splits_df["mutation_label"].isin(label_map)].reset_index(drop=True)
        labels_all = splits_df["mutation_label"].map(label_map).values
    else:
        labels_all = splits_df["RET"].values

    embedding_index = build_embedding_index()
    print("Extracting frozen ABMIL slide embeddings (once, cached for all configs)...")
    z_lookup = extract_all_slide_embeddings(abmil_model, embedding_index, splits_df["patient"].tolist())

    # --- Z-normalization of the ABMIL embedding, TRAIN-set stats only ---
    # Without this, z's natural scale (whatever the attention-weighted sum
    # of UNI2-h embeddings happens to sit at) can dominate the much smaller
    # clinical block (26 dims of mostly 0/1 indicators + one z-scored Age
    # column) at the fusion classifier's first layer, purely because of
    # scale rather than informativeness. Standard multimodal fusion fix:
    # z-score z per-dimension using TRAIN patients only, same
    # leakage-avoidance principle as every other train-only statistic in
    # this pipeline. Stats are saved to disk so fusion_bootstrap_ci.py can
    # apply the identical transform at inference time later.
    train_patients_for_norm = splits_df.loc[splits_df["split"] == "train", "patient"].tolist()
    train_z_matrix = np.stack([z_lookup[p] for p in train_patients_for_norm if p in z_lookup])
    z_mean = train_z_matrix.mean(axis=0)
    z_std = train_z_matrix.std(axis=0) + 1e-8  # epsilon avoids divide-by-zero on any constant dimension
    z_lookup = {p: (z - z_mean) / z_std for p, z in z_lookup.items()}

    norm_stats_path = os.path.join(FUSION_RESULTS_DIR, f"z_norm_stats_{task}.npz")
    np.savez(norm_stats_path, mean=z_mean, std=z_std)
    print(f"  Z-normalized ABMIL embeddings using {len(train_patients_for_norm)} train patients' stats "
          f"(saved to {os.path.basename(norm_stats_path)})")

    clinical_lookup, clinical_cols = load_clinical_features(exclude_cols=exclude_clinical_cols)
    print(f"Clinical feature columns ({len(clinical_cols)}): {clinical_cols}")
    if exclude_clinical_cols:
        print(f"  (diagnostic run -- excluded: {exclude_clinical_cols})")
    input_dim = EMBED_DIM + len(clinical_cols)

    datasets = {}
    for split in ["train", "val", "test"]:
        mask = splits_df["split"] == split
        patients = splits_df.loc[mask, "patient"].tolist()
        labels = labels_all[mask.values].tolist()
        datasets[split] = FusionDataset(patients, labels, z_lookup, clinical_lookup)
        print(f"  {split}: {len(datasets[split])} patients "
              f"(dropped {mask.sum() - len(datasets[split])} without embedding/clinical match)")

    class_weights = None
    if task == "ret_binary":
        train_labels = np.array(datasets["train"].labels)
        n_neg = (train_labels == 0).sum()
        n_pos = (train_labels == 1).sum()
        total = n_neg + n_pos
        class_weights = torch.tensor(
            [total / (2 * n_neg), total / (2 * n_pos)], dtype=torch.float32
        ).to(DEVICE)
        print(f"  Class weights: neg={class_weights[0]:.3f}, pos={class_weights[1]:.3f}")

    return {
        "task": task, "arch": arch, "num_classes": num_classes,
        "datasets": datasets, "input_dim": input_dim,
        "clinical_cols": clinical_cols, "class_weights": class_weights,
    }


def train_one_config(setup, fc_dim, dropout, lr, batch_size=16,
                      weight_decay=WEIGHT_DECAY, max_epochs=NUM_EPOCHS,
                      patience=PATIENCE, ckpt_path=None, verbose=True,
                      evaluate_test=False):
    """Trains a single fusion classifier config using the datasets/class
    weights already prepared by setup_task(). Returns best validation
    metrics (and test metrics only if evaluate_test=True -- keep this False
    during hyperparameter search so test is never touched until the final,
    single chosen config)."""
    task = setup["task"]
    num_classes = setup["num_classes"]
    datasets = setup["datasets"]
    input_dim = setup["input_dim"]
    class_weights = setup["class_weights"]

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=32, shuffle=False)

    fusion_model = FusionClassifier(input_dim, fc_dim, num_classes, dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=patience // 2)

    best_val_auc, patience_counter, history = 0, 0, []
    if ckpt_path is None:
        ckpt_path = os.path.join(FUSION_RESULTS_DIR, f"_tmp_fusion_{task}.pt")
    best_val_metrics = None

    for epoch in range(max_epochs):
        fusion_model.train()
        total_loss, correct, total = 0, 0, 0
        for x, labels in train_loader:
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = fusion_model(x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)

        val_bal_acc, val_auc, val_auprc, val_f1, _, _, _ = evaluate(fusion_model, val_loader, num_classes)
        scheduler.step(val_auc)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / total, "val_auc": val_auc, "val_bal_acc": val_bal_acc})
        if verbose:
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss/total:.4f} Acc: {correct/total:.3f} | "
                  f"Val BalAcc: {val_bal_acc:.3f} AUC: {val_auc:.3f} AUPRC: {val_auprc:.3f} F1: {val_f1:.3f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_metrics = {"bal_acc": val_bal_acc, "auc": val_auc, "auprc": val_auprc, "f1": val_f1}
            torch.save(fusion_model.state_dict(), ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    result = {"fc_dim": fc_dim, "dropout": dropout, "lr": lr, "batch_size": batch_size,
              "best_val": best_val_metrics, "n_epochs_trained": len(history), "ckpt_path": ckpt_path}

    if evaluate_test:
        fusion_model.load_state_dict(torch.load(ckpt_path))
        test_loader = DataLoader(datasets["test"], batch_size=32, shuffle=False)

        threshold_info = None
        if task == "ret_binary":
            _, _, _, _, _, val_labels, val_probs = evaluate(fusion_model, val_loader, num_classes)
            best_thresh, best_val_bal_acc = find_optimal_threshold(np.array(val_labels), val_probs[:, 1])
            print(f"Optimal threshold (selected on val only): {best_thresh:.2f} (val BalAcc: {best_val_bal_acc:.3f})")
            threshold_info = {"threshold": float(best_thresh), "val_bal_acc_at_threshold": float(best_val_bal_acc)}

        test_bal_acc, test_auc, test_auprc, test_f1, test_preds, test_labels, test_probs = evaluate(fusion_model, test_loader, num_classes)
        target_names = ["BRAF_V600E", "RAS", "Other"] if task == "multiclass" else ["RET_negative", "RET_positive"]
        print(f"\nTEST (default threshold) | BalAcc: {test_bal_acc:.4f} AUC: {test_auc:.4f} AUPRC: {test_auprc:.4f} F1: {test_f1:.4f}")
        print(classification_report(test_labels, test_preds, target_names=target_names))

        result["test_default_threshold"] = {"bal_acc": test_bal_acc, "auc": test_auc, "auprc": test_auprc, "f1": test_f1}
        result["threshold_calibration"] = threshold_info
        if threshold_info is not None:
            test_preds_cal = (test_probs[:, 1] >= threshold_info["threshold"]).astype(int)
            result["test_calibrated_threshold"] = {
                "bal_acc": balanced_accuracy_score(test_labels, test_preds_cal),
                "f1": f1_score(test_labels, test_preds_cal, average="macro"),
            }
    result["history"] = history
    return result


def run_fusion(task, fc_dim=FUSION_FC_DIM, dropout=FUSION_DROPOUT, lr=FUSION_LR,
               batch_size=16, exclude_clinical_cols=None, results_suffix=""):
    """Single end-to-end run (setup + train + test evaluation) at one fixed
    config -- use this for the FINAL run only, after hyperparameter search
    (see fusion_hparam_search.py) has picked fc_dim/dropout/lr on validation
    data. Test is evaluated exactly once here.

    exclude_clinical_cols / results_suffix: use together for the site-confound
    diagnostic (see load_clinical_features docstring) -- e.g.
        run_fusion(task, ..., exclude_clinical_cols=['race_missing','ethnicity_missing'],
                   results_suffix='_no_race_ethnicity_missing_flags')
    writes to a separate checkpoint/results filename so it never overwrites
    the main run."""
    setup = setup_task(task, exclude_clinical_cols=exclude_clinical_cols)
    ckpt_path = os.path.join(FUSION_RESULTS_DIR, f"best_fusion_{task}{results_suffix}.pt")
    result = train_one_config(setup, fc_dim, dropout, lr, batch_size=batch_size,
                               ckpt_path=ckpt_path, verbose=True, evaluate_test=True)

    results = {
        "task": task, "model": "Fusion_ABMILz_plus_clinical",
        "clinical_features_used": setup["clinical_cols"],
        "excluded_clinical_cols": exclude_clinical_cols,
        "abmil_checkpoint": setup["arch"]["model_file"],
        "fusion_hyperparameters": {"fc_dim": fc_dim, "dropout": dropout, "lr": lr, "batch_size": batch_size},
        "test_default_threshold": result["test_default_threshold"],
        "threshold_calibration": result["threshold_calibration"],
        "history": result["history"],
    }
    if "test_calibrated_threshold" in result:
        results["test_calibrated_threshold"] = result["test_calibrated_threshold"]

    out_path = os.path.join(FUSION_RESULTS_DIR, f"fusion_results_{task}{results_suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {os.path.basename(out_path)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["multiclass", "ret_binary", "both"], default="both")
    parser.add_argument("--fc_dim", type=int, default=FUSION_FC_DIM)
    parser.add_argument("--dropout", type=float, default=FUSION_DROPOUT)
    parser.add_argument("--lr", type=float, default=FUSION_LR)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--exclude_clinical_cols", type=str, default=None,
                         help="Comma-separated clinical feature columns to exclude, e.g. "
                              "'race_missing,ethnicity_missing' for the site-confound diagnostic "
                              "(see load_clinical_features docstring). Results are saved with a "
                              "matching suffix so they don't overwrite the main run.")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print("NOTE: this runs a SINGLE fixed config and touches test once.")
    print("Run fusion_hparam_search.py first to select fc_dim/dropout/lr via validation only.\n")
    tasks = ["multiclass", "ret_binary"] if args.task == "both" else [args.task]

    exclude_cols = args.exclude_clinical_cols.split(",") if args.exclude_clinical_cols else None
    suffix = ("_excl_" + "_".join(exclude_cols)) if exclude_cols else ""

    for t in tasks:
        run_fusion(t, fc_dim=args.fc_dim, dropout=args.dropout, lr=args.lr,
                   batch_size=args.batch_size, exclude_clinical_cols=exclude_cols,
                   results_suffix=suffix)

    print("\nFusion training complete.")
