import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
import h5py
import json
import time

# === CONFIG (pointing at CORRECTED v2 embeddings) ===
EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
SPLITS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"
OUTPUT_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2/tuning"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 1536
NUM_EPOCHS = 150   # more epochs since smaller LR may need longer to converge
PATIENCE = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)

# NEW: two additional, even smaller learning rates, continuing the same pattern
# as the original small-LR extension (5e-6, 1e-6), at the same fixed architecture size
EXTRA_LR_VALUES = [5e-7, 1e-7]
FIXED_ARCH = (512, 256)  # matches the original small-LR extension's fixed size


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
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0
    return bal_acc, auc, f1


def run_config(task, num_classes, lr, attn_dim, fc_dim, class_weights=None):
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=PATIENCE // 2)

    best_val_auc = 0
    patience_counter = 0
    start = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_bal_acc, val_auc, val_f1 = evaluate(model, val_loader, criterion, num_classes)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (elapsed {(time.time()-start)/60:.1f}min)", flush=True)
            break
    else:
        print(f"  Completed {NUM_EPOCHS} epochs (elapsed {(time.time()-start)/60:.1f}min)", flush=True)

    return {"config": config_name, "lr": lr, "attn_dim": attn_dim, "fc_dim": fc_dim, "best_val_auc": best_val_auc}


def run_extra_configs_and_merge(task, num_classes, class_weights=None):
    print(f"\n{'='*60}")
    print(f"EXTRA SMALL-LR CONFIGS (completing 13-config search): {task}")
    print(f"Testing lr in {EXTRA_LR_VALUES} at fixed architecture {FIXED_ARCH}")
    print(f"Selection based on VALIDATION AUC ONLY. Test set is not touched.")
    print(f"{'='*60}")

    new_results = []
    for lr in EXTRA_LR_VALUES:
        result = run_config(task, num_classes, lr, FIXED_ARCH[0], FIXED_ARCH[1], class_weights)
        new_results.append(result)

    # Load existing 11-config results and merge
    results_path = os.path.join(OUTPUT_DIR, f"tuning_v2_results_{task}.json")
    with open(results_path, "r") as f:
        existing = json.load(f)

    existing_configs = existing["all_configs_validation_only"]
    all_configs = existing_configs + new_results

    # Re-sort and re-select
    all_configs_sorted = sorted(all_configs, key=lambda x: x["best_val_auc"], reverse=True)
    best = all_configs_sorted[0]
    literature_default = next((r for r in all_configs if r["config"] == "lr1e-05_attn512_fc256"), None)

    print(f"\n{'='*60}")
    print(f"UPDATED FULL RESULTS TABLE: {task} (13 configs total, corrected v2 data)")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'Config':<28} {'Val AUC':>9}")
    for i, r in enumerate(all_configs_sorted):
        marker = " <-- SELECTED" if r["config"] == best["config"] else ""
        print(f"  {i+1:<3} {r['config']:<26} {r['best_val_auc']:>9.4f}{marker}")

    if literature_default:
        lit_rank = [r["config"] for r in all_configs_sorted].index("lr1e-05_attn512_fc256") + 1
        print(f"\nLiterature default ranked #{lit_rank}/13 (val_auc={literature_default['best_val_auc']:.4f})")

    if best["config"] == "lr1e-05_attn512_fc256":
        print("\n*** Literature-default configuration selected (unchanged from before). ***")
    else:
        print(f"\n*** Selected config: {best['config']} (may be unchanged or different from the 11-config result — check) ***")

    # Save updated, complete 13-config record
    updated = {
        "task": task,
        "all_configs_validation_only": all_configs,
        "selected_config": best["config"],
        "literature_default_was_selected": best["config"] == "lr1e-05_attn512_fc256",
        "note": "Test set was NOT evaluated in this script. Official test performance comes from train_abmil_v2_retrained.py only.",
        "n_configs_total": len(all_configs)
    }
    with open(results_path, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"\nUpdated and saved: {results_path}")

    return all_configs_sorted, best


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Using CORRECTED embeddings from: {EMBEDDINGS_DIR}")

    run_extra_configs_and_merge("multiclass", num_classes=3)

    ret_neg, ret_pos = 319, 21
    total = ret_neg + ret_pos
    weights = [total / (2 * ret_neg), total / (2 * ret_pos)]
    run_extra_configs_and_merge("ret_binary", num_classes=2, class_weights=weights)

    print("\n" + "="*60)
    print("EXTRA CONFIGS COMPLETE - 13-config search now finalized for both tasks!")
    print("="*60)
