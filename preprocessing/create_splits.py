import pandas as pd
import numpy as np
import os

# === PATHS ===
LABELS_PATH = "/cs/student/project_msc/2025/aibh/iconstan/final_labels.csv"
TILES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/tiles"
OUTPUT_PATH = "/cs/student/project_msc/2025/aibh/iconstan/splits.csv"

np.random.seed(42)  # reproducibility

# === LOAD LABELS ===
labels = pd.read_csv(LABELS_PATH)

# === ADD 14 UNLABELLED CASES AS OTHER ===
unlabelled = [
    "TCGA-BJ-A0Z5", "TCGA-BJ-A190", "TCGA-BJ-A28W", "TCGA-BJ-A291",
    "TCGA-BJ-A45C", "TCGA-EL-A3ZM", "TCGA-ET-A25L", "TCGA-ET-A25M",
    "TCGA-ET-A25N", "TCGA-ET-A25P", "TCGA-FE-A238", "TCGA-FE-A239",
    "TCGA-FY-A4B0", "TCGA-IM-A41Z"
]
unlabelled_df = pd.DataFrame({
    "patient": unlabelled,
    "BRAF_V600E": 0,
    "RAS": 0,
    "RET": 0,
    "mutation_label": "Other"
})
labels = pd.concat([labels, unlabelled_df], ignore_index=True)
labels = labels.drop_duplicates(subset=["patient"])

# === FILTER TO ONLY PATIENTS WITH TILES ===
tiled = set()
for f in os.listdir(TILES_DIR):
    if f.endswith(".h5"):
        tiled.add(f[:12])
labels = labels[labels["patient"].isin(tiled)].copy()

# === EXTRACT SITE CODE ===
labels["site"] = labels["patient"].str[5:7]

# === ASSIGN TEST SET (site EL) ===
labels["split"] = "train"
labels.loc[labels["site"] == "EL", "split"] = "test"

# === 80/20 TRAIN/VAL SPLIT FOR REMAINING ===
train_val = labels[labels["split"] == "train"].copy()
val_idx = train_val.sample(frac=0.2, random_state=42).index
labels.loc[val_idx, "split"] = "val"

# === SUMMARY ===
print("=== SPLIT SUMMARY ===")
print(labels["split"].value_counts())
print("\n=== MUTATION LABELS PER SPLIT ===")
print(labels.groupby(["split", "mutation_label"]).size().unstack(fill_value=0))
print("\n=== RET CASES PER SPLIT ===")
print(labels.groupby("split")["RET"].sum())

# === SAVE ===
labels.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")
