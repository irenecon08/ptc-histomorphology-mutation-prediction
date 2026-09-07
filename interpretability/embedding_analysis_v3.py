import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json

# === PATHS (final retuned models) ===
EMBEDDINGS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embeddings_v2"
SPLITS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"
RESULTS_DIR = "/cs/student/project_msc/2025/aibh/iconstan/results_v2"
OUTPUT_DIR = "/cs/student/project_msc/2025/aibh/iconstan/embedding_analysis_v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM = 1536

# Per-task architecture (now DIFFERENT between tasks, per retuning results)
TASK_ARCH = {
    "multiclass": {"attn_dim": 512, "fc_dim": 512, "model_file": "best_multiclass_v2_retuned.pt"},
    "ret_binary": {"attn_dim": 512, "fc_dim": 256, "model_file": "best_ret_binary_v2_retuned.pt"},
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def build_embedding_index():
    index = {}
    for f in os.listdir(EMBEDDINGS_DIR):
        if f.endswith(".h5"):
            patient_prefix = f[:12]
            index[patient_prefix] = os.path.join(EMBEDDINGS_DIR, f)
    return index


EMBEDDING_INDEX = build_embedding_index()


def get_embedding_file(patient):
    return EMBEDDING_INDEX.get(patient, None)


def extract_slide_embeddings(model, patients_df):
    embeddings = []
    valid_rows = []
    total = len(patients_df)

    for i, (_, row) in enumerate(patients_df.iterrows()):
        patient = row["patient"]
        emb_file = get_embedding_file(patient)
        if emb_file is None:
            continue
        with h5py.File(emb_file, "r") as f:
            features = torch.tensor(f["features"][:], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            _, _, z = model(features)
        embeddings.append(z.squeeze().cpu().numpy())
        valid_rows.append(row)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Processed {i + 1}/{total} slides...", flush=True)

    embeddings = np.array(embeddings)
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    return embeddings, valid_df


def plot_projection(coords, labels, title, out_path, label_colors=None, label_order=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    unique_labels = label_order if label_order else sorted(set(labels))
    if label_colors is None:
        cmap = plt.get_cmap("tab10")
        label_colors = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    for lab in unique_labels:
        mask = np.array(labels) == lab
        ax.scatter(coords[mask, 0], coords[mask, 1], label=str(lab), alpha=0.7, s=40,
                   color=label_colors.get(lab))

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def run_analysis(task, label_column, label_map=None, target_names=None):
    print(f"\n{'='*50}\nEmbedding analysis (v3, final retuned model): {task}\n{'='*50}")

    arch = TASK_ARCH[task]
    num_classes = 3 if task == "multiclass" else 2

    splits_df = pd.read_csv(SPLITS_PATH)
    if task == "multiclass":
        splits_df = splits_df[splits_df["mutation_label"].isin(label_map)]

    model = ABMIL(EMBED_DIM, arch["attn_dim"], arch["fc_dim"], num_classes, 0.0).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, arch["model_file"]), map_location=DEVICE))
    model.eval()

    print("Extracting slide-level embeddings for all slides (train+val+test)...")
    embeddings, valid_df = extract_slide_embeddings(model, splits_df)
    print(f"Extracted embeddings shape: {embeddings.shape}")

    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(embeddings)
    print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}")

    print("Running UMAP...")
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_coords = reducer.fit_transform(embeddings)

    labels = valid_df[label_column].values
    if label_map:
        inv_label_map = {v: k for k, v in label_map.items()}
        label_names = [inv_label_map.get(l, l) if isinstance(l, int) else l for l in labels]
    else:
        label_names = ["Positive" if l == 1 else "Negative" for l in labels]

    plot_projection(
        pca_coords, label_names,
        f"PCA of Slide-Level Embeddings ({task}, final model)\nExplained variance: {pca.explained_variance_ratio_.sum():.1%}",
        os.path.join(OUTPUT_DIR, f"pca_{task}_by_label.png"),
        label_order=target_names
    )
    plot_projection(
        umap_coords, label_names,
        f"UMAP of Slide-Level Embeddings ({task}, final model)",
        os.path.join(OUTPUT_DIR, f"umap_{task}_by_label.png"),
        label_order=target_names
    )
    plot_projection(
        umap_coords, valid_df["split"].values,
        f"UMAP of Slide-Level Embeddings ({task}, final model) — coloured by split",
        os.path.join(OUTPUT_DIR, f"umap_{task}_by_split.png"),
        label_order=["train", "val", "test"]
    )
    if "site" in valid_df.columns:
        plot_projection(
            umap_coords, valid_df["site"].values,
            f"UMAP of Slide-Level Embeddings ({task}, final model) — coloured by site",
            os.path.join(OUTPUT_DIR, f"umap_{task}_by_site.png")
        )

    out_df = valid_df.copy()
    out_df["pca_x"] = pca_coords[:, 0]
    out_df["pca_y"] = pca_coords[:, 1]
    out_df["umap_x"] = umap_coords[:, 0]
    out_df["umap_y"] = umap_coords[:, 1]
    out_df.to_csv(os.path.join(OUTPUT_DIR, f"embedding_coords_{task}.csv"), index=False)
    print(f"  Saved coordinates to embedding_coords_{task}.csv")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    run_analysis(
        task="multiclass",
        label_column="mutation_label",
        label_map={"BRAF_V600E": 0, "RAS": 1, "Other": 2},
        target_names=["BRAF_V600E", "RAS", "Other"]
    )

    run_analysis(
        task="ret_binary",
        label_column="RET",
        label_map=None,
        target_names=["Negative", "Positive"]
    )

    print("\nAll embedding analyses (v3, final models) complete!")
