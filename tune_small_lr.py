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

EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings"
SPLITS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"
OUTPUT_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results/tuning"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 1536
NUM_EPOCHS = 150   # more epochs since smaller LR may need longer to converge
PATIENCE = 15      # more patience for the same reason

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test smaller learning rates at the best model size [512, 256]
LR_VALUES = [5e-6, 1e-6]
ATTN_DIM, FC_DIM = 512, 256

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

def run_config(task, num_classes, lr, class_weights=None):
    config_name = f"lr{lr}_attn{ATTN_DIM}_fc{FC_DIM}"
    print(f"\n--- Config: {config_name} | Task: {task} ---")

    splits_df = pd.read_csv(SPLITS_PATH)
    train_ds = SlideDataset(splits_df, "train", task)
    val_ds = SlideDataset(splits_df, "val", task)
    test_ds = SlideDataset(splits_df, "test", task)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = ABMIL(EMBED_DIM, ATTN_DIM, FC_DIM, num_classes, 0.0).to(DEVICE)
    if class_weights:
        weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=PATIENCE // 2)

    best_val_auc = 0
    patience_counter = 0
    best_path = os.path.join(OUTPUT_DIR, f"best_{task}_{config_name}.pt")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_bal_acc, val_auc, val_auprc, val_f1, _, _ = evaluate(model, val_loader, criterion, num_classes)
        scheduler.step(val_auc)
        print(f"  Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.3f} BalAcc: {val_bal_acc:.3f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Test eval
    model.load_state_dict(torch.load(best_path))
    test_bal_acc, test_auc, test_auprc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, num_classes)
    target_names = ["BRAF_V600E", "RAS", "Other"] if task == "multiclass" else ["RET_negative", "RET_positive"]
    print(f"  TEST | BalAcc: {test_bal_acc:.4f} AUC: {test_auc:.4f} AUPRC: {test_auprc:.4f} F1: {test_f1:.4f}")
    print(classification_report(test_labels, test_preds, target_names=target_names))

    return {
        "config": config_name, "lr": lr,
        "best_val_auc": best_val_auc,
        "test": {"bal_acc": test_bal_acc, "auc": test_auc, "auprc": test_auprc, "f1": test_f1}
    }

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print("Testing smaller learning rates (5e-6, 1e-6) at model size [512, 256]")

    all_results = {"multiclass": [], "ret_binary": []}

    for lr in LR_VALUES:
        all_results["multiclass"].append(run_config("multiclass", 3, lr))

    ret_neg, ret_pos = 319, 21
    total = ret_neg + ret_pos
    weights = [total / (2 * ret_neg), total / (2 * ret_pos)]
    for lr in LR_VALUES:
        all_results["ret_binary"].append(run_config("ret_binary", 2, lr, class_weights=weights))

    with open(os.path.join(OUTPUT_DIR, "small_lr_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== SUMMARY: smaller learning rates ===")
    for task, results in all_results.items():
        print(f"\n{task}:")
        for r in results:
            t = r["test"]
            print(f"  {r['config']}: Test AUC={t['auc']:.4f} AUPRC={t['auprc']:.4f} BalAcc={t['bal_acc']:.4f} F1={t['f1']:.4f}")

    print("\nDone! Compare against original (lr1e-5): aim a AUC=0.783, aim b AUC=0.901")
