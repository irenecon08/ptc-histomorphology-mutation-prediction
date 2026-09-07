import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, classification_report
import h5py
import json
from itertools import product
import time

# === CONFIG (pointing at CORRECTED v2 embeddings) ===
EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
SPLITS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"
OUTPUT_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2/tuning"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 1536
NUM_EPOCHS = 100
PATIENCE = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Same 9-point grid as original tuning
LR_VALUES = [1e-5, 5e-5, 1e-4]
MODEL_SIZES = [(256, 128), (512, 256), (512, 512)]

# Same small-LR extension as original
SMALL_LR_VALUES = [5e-6, 1e-6]
SMALL_LR_MODEL_SIZE = (512, 256)  # matches best/literature-default size
SMALL_LR_EPOCHS = 150
SMALL_LR_PATIENCE = 15


class SlideDataset(Dataset):
    def __init__(self, splits_df, split, task):
        self.task = task
        self.data = splits_df[splits_df["split"] == split].reset_index(drop=True)
        if task == "multiclass":
            label_map = {"BRAF_V600E": 0, "RAS": 1, "Other": 2}
            self.data = self.data[self.data["mutation_label"].isin(label_map)]
            self.labels = self.data["mutation_label"].map(label_map).values
            self.num_classes = 3
        else:
            self.labels = self.data["RET"].values
            self.num_classes = 2
        self.patients = self.data["patient"].values

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        label = self.labels[idx]
        emb_file = None
        for f in os.listdir(EMBEDDINGS_DIR):
            if f.startswith(patient) and f.endswith(".h5"):
                emb_file = os.path.join(EMBEDDINGS_DIR, f)
                break
        if emb_file is None:
            return None, label
        with h5py.File(emb_file, "r") as f:
            features = torch.tensor(f["features"][:], dtype=torch.float32)
        return features, label


def collate_fn(batch):
    batch = [(f, l) for f, l in batch if f is not None]
    if not batch:
        return None, None
    features, labels = zip(*batch)
    return list(features), torch.tensor(labels, dtype=torch.long)


class GatedAttention(nn.Module):
    def __init__(self, embed_dim=1536, hidden_dim=512, dropout=0.0):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Sigmoid())
        self.w = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        A = self.w(self.drop(self.V(x) * self.U(x)))
        A = torch.softmax(A, dim=0)
        return torch.mm(A.T, x), A


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


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for features_list, labels in loader:
        if features_list is None:
            continue
        labels = labels.to(DEVICE)
        for i, features in enumerate(features_list):
            features = features.to(DEVICE)
            label = labels[i].unsqueeze(0)
            optimizer.zero_grad()
            logits, _ = model(features)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == label).sum().item()
            total += 1
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, num_classes):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for features_list, labels in loader:
            if features_list is None:
                continue
            labels = labels.to(DEVICE)
            for i, features in enumerate(features_list):
                features = features.to(DEVICE)
                label = labels[i].unsqueeze(0)
                logits, _ = model(features)
                all_preds.append(logits.argmax(dim=1).cpu().item())
                all_labels.append(label.cpu().item())
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
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
    return bal_acc, auc, auprc, f1, all_preds, all_labels


def run_config(task, num_classes, lr, attn_dim, fc_dim, class_weights=None,
               num_epochs=NUM_EPOCHS, patience=PATIENCE):
    config_name = f"lr{lr}_attn{attn_dim}_fc{fc_dim}"
    print(f"\n--- Config: {config_name} | Task: {task} ---", flush=True)

    splits_df = pd.read_csv(SPLITS_PATH)
    train_ds = SlideDataset(splits_df, "train", task)
    val_ds = SlideDataset(splits_df, "val", task)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = ABMIL(EMBED_DIM, attn_dim, fc_dim, num_classes, 0.0).to(DEVICE)
    if class_weights:
        weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=patience // 2)

    best_val_auc = 0
    patience_counter = 0
    best_path = os.path.join(OUTPUT_DIR, f"best_{task}_{config_name}.pt")

    start = time.time()
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_bal_acc, val_auc, val_auprc, val_f1, _, _ = evaluate(model, val_loader, criterion, num_classes)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (elapsed {(time.time()-start)/60:.1f}min)", flush=True)
            break
    else:
        print(f"  Completed {num_epochs} epochs (elapsed {(time.time()-start)/60:.1f}min)", flush=True)

    return {"config": config_name, "lr": lr, "attn_dim": attn_dim, "fc_dim": fc_dim, "best_val_auc": best_val_auc}


def run_full_tuning(task, num_classes, class_weights=None):
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER RETUNING ON CORRECTED (v2) DATA: {task}")
    print(f"Phase 1: {len(LR_VALUES)} LRs x {len(MODEL_SIZES)} sizes = {len(LR_VALUES)*len(MODEL_SIZES)} configs")
    print(f"Selection based on VALIDATION AUC ONLY. Test set is not touched during selection.")
    print(f"{'='*60}")

    all_results = []
    for lr, (attn_dim, fc_dim) in product(LR_VALUES, MODEL_SIZES):
        result = run_config(task, num_classes, lr, attn_dim, fc_dim, class_weights)
        all_results.append(result)

    print(f"\n{'='*60}")
    print(f"Phase 2: small-LR extension at literature-default model size {SMALL_LR_MODEL_SIZE}")
    print(f"{'='*60}")
    for lr in SMALL_LR_VALUES:
        result = run_config(task, num_classes, lr, SMALL_LR_MODEL_SIZE[0], SMALL_LR_MODEL_SIZE[1],
                             class_weights, num_epochs=SMALL_LR_EPOCHS, patience=SMALL_LR_PATIENCE)
        all_results.append(result)

    # === Selection based on VALIDATION AUC ONLY ===
    all_results_sorted_by_val = sorted(all_results, key=lambda x: x["best_val_auc"], reverse=True)
    best_by_val = all_results_sorted_by_val[0]

    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS TABLE: {task} (13 configs, corrected v2 data)")
    print(f"{'='*60}")
    print(f"{'Config':<28} {'Val AUC':>9}")
    print("-" * 40)
    for r in all_results_sorted_by_val:
        marker = " <-- SELECTED" if r["config"] == best_by_val["config"] else ""
        print(f"  {r['config']:<26} {r['best_val_auc']:>9.4f}{marker}")

    literature_default = next((r for r in all_results if r["config"] == "lr1e-05_attn512_fc256"), None)
    if literature_default:
        lit_rank = [r["config"] for r in all_results_sorted_by_val].index("lr1e-05_attn512_fc256") + 1
        print(f"\nLiterature default (lr=1e-5, [512,256]) ranked #{lit_rank}/13 by validation AUC "
              f"(val_auc={literature_default['best_val_auc']:.4f})")

    if best_by_val["config"] == "lr1e-05_attn512_fc256":
        print("\n*** CONFIRMED: literature-default configuration is also the validation-selected configuration "
              "on corrected (v2) data. ***")
        print("*** Existing best_multiclass_v2.pt / best_ret_binary_v2.pt (from train_abmil_v2.py) remain final. ***")
        print("*** No test-set evaluation needed here - official test results already exist from train_abmil_v2.py. ***")
    else:
        print(f"\n*** NOTE: validation-selected config differs from literature default. ***")
        print(f"*** Update train_abmil_v2.py hyperparameters to: {best_by_val['config']} and re-run it. ***")
        print(f"*** train_abmil_v2.py will then produce the official, single test-set evaluation. ***")
        print(f"*** This WILL require re-running: threshold calibration, heatmaps, PCA/UMAP, TIIC analysis. ***")

    with open(os.path.join(OUTPUT_DIR, f"tuning_v2_results_{task}.json"), "w") as f:
        json.dump({
            "task": task,
            "all_configs_validation_only": [
                {"config": r["config"], "lr": r["lr"], "attn_dim": r["attn_dim"],
                 "fc_dim": r["fc_dim"], "best_val_auc": r["best_val_auc"]}
                for r in all_results
            ],
            "selected_config": best_by_val["config"],
            "literature_default_was_selected": best_by_val["config"] == "lr1e-05_attn512_fc256",
            "note": "Test set was NOT evaluated in this script. Official test performance comes from train_abmil_v2.py only."
        }, f, indent=2)
    print(f"\nSaved results to tuning_v2_results_{task}.json")

    return all_results, best_by_val


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Using CORRECTED embeddings from: {EMBEDDINGS_DIR}")

    run_full_tuning("multiclass", num_classes=3)

    ret_neg, ret_pos = 319, 21
    total = ret_neg + ret_pos
    weights = [total / (2 * ret_neg), total / (2 * ret_pos)]
    run_full_tuning("ret_binary", num_classes=2, class_weights=weights)

    print("\n" + "="*60)
    print("ALL RETUNING ON CORRECTED (v2) DATA COMPLETE!")
    print("="*60)
