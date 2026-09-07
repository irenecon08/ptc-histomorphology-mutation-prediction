"""
Paired bootstrap comparison: ABMIL (WSI-only) vs Fusion.

Unlike bootstrap_ci.py / fusion_bootstrap_ci_updated.py, which bootstrap
each model INDEPENDENTLY (a different random resample per model per
iteration), this draws ONE stratified resample per iteration and applies it
to BOTH models' predictions on the SAME patients, then computes the
difference (Fusion - ABMIL) in each metric for that resample. The 95% CI of
this difference distribution is a direct, statistically sharper test of
whether the two models differ than comparing two independently-bootstrapped
CIs for overlap: if the difference CI excludes 0, that is a genuine,
paired-test-confirmed difference.

Because the fusion model can drop slightly more test patients than the base
model (it additionally requires clinical data to be present), predictions
are aligned by patient ID (inner join) before pairing -- resampling is only
ever done over patients both models actually scored.

Usage:
    python paired_bootstrap.py --task multiclass --fusion_fc_dim 64
    python paired_bootstrap.py --task ret_binary --fusion_fc_dim 64
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression
import h5py

from fusion_train import (
    ABMIL, FusionClassifier, TASK_ARCH, RESULTS_DIR, FUSION_RESULTS_DIR,
    SPLITS_PATH, EMBED_DIM, build_embedding_index, load_clinical_features,
)

DEVICE = "cpu"
N_BOOTSTRAP = 2000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def get_abmil_predictions(task):
    """ABMIL (WSI-only) test predictions, WITH patient IDs for alignment
    (bootstrap_ci.py does not return these, so this mirrors its logic but
    keeps the patient list)."""
    arch = TASK_ARCH[task]
    num_classes = arch["num_classes"]

    model = ABMIL(EMBED_DIM, arch["attn_dim"], arch["fc_dim"], num_classes, 0.0).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, arch["model_file"]), map_location=DEVICE))
    model.eval()

    splits_df = pd.read_csv(SPLITS_PATH)
    if task == "multiclass":
        label_map = {"BRAF_V600E": 0, "RAS": 1, "Other": 2}
        splits_df = splits_df[splits_df["mutation_label"].isin(label_map)].reset_index(drop=True)
        splits_df["label"] = splits_df["mutation_label"].map(label_map)
    else:
        splits_df["label"] = splits_df["RET"]

    test_df = splits_df[splits_df["split"] == "test"].copy()
    embedding_index = build_embedding_index()

    patients, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            patient = row["patient"]
            emb_file = embedding_index.get(patient)
            if emb_file is None:
                continue
            with h5py.File(emb_file, "r") as f:
                features = torch.tensor(f["features"][:], dtype=torch.float32)
            logits, _, _ = model(features)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            patients.append(patient)
            all_probs.append(probs)
            all_labels.append(row["label"])

    return patients, np.array(all_probs), np.array(all_labels)


def get_baseline_predictions(task):
    """Loads the saved baseline test predictions rather than refitting.
    An earlier version of this function retrained the classifier on the
    assumption that LogisticRegression's fit is deterministic given fixed
    inputs; this was found not to hold in this environment (refitting
    produced per-patient probability differences of up to 0.22 from the
    originally saved predictions, and a materially different test AUC).
    The saved predictions are therefore loaded directly, guaranteeing this
    comparison uses the exact model reported in baseline_results_*.json
    and Tables 3-4 of the thesis, rather than an approximate refit."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, f"test_predictions_baseline_{task}.csv"))
    patients = df["patient"].tolist()
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    probs = df[prob_cols].values
    labels = df["true_label"].values
    return patients, probs, labels


def get_fusion_predictions(task, fc_dim, dropout=0.0):
    arch = TASK_ARCH[task]
    num_classes = arch["num_classes"]

    abmil_model = ABMIL(EMBED_DIM, arch["attn_dim"], arch["fc_dim"], num_classes, 0.0).to(DEVICE)
    abmil_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, arch["model_file"]), map_location=DEVICE))
    abmil_model.eval()

    clinical_lookup, clinical_cols = load_clinical_features()
    input_dim = EMBED_DIM + len(clinical_cols)

    norm_stats = np.load(os.path.join(FUSION_RESULTS_DIR, f"z_norm_stats_{task}.npz"))
    z_mean, z_std = norm_stats["mean"], norm_stats["std"]

    fusion_ckpt_path = os.path.join(FUSION_RESULTS_DIR, f"best_fusion_{task}.pt")
    fusion_model = FusionClassifier(input_dim, fc_dim, num_classes, dropout).to(DEVICE)
    fusion_model.load_state_dict(torch.load(fusion_ckpt_path, map_location=DEVICE))
    fusion_model.eval()

    splits_df = pd.read_csv(SPLITS_PATH)
    if task == "multiclass":
        label_map = {"BRAF_V600E": 0, "RAS": 1, "Other": 2}
        splits_df = splits_df[splits_df["mutation_label"].isin(label_map)].reset_index(drop=True)
        splits_df["label"] = splits_df["mutation_label"].map(label_map)
    else:
        splits_df["label"] = splits_df["RET"]

    test_df = splits_df[splits_df["split"] == "test"].copy()
    embedding_index = build_embedding_index()

    patients, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            patient = row["patient"]
            emb_file = embedding_index.get(patient)
            if emb_file is None or patient not in clinical_lookup:
                continue
            with h5py.File(emb_file, "r") as f:
                features = torch.tensor(f["features"][:], dtype=torch.float32)
            _, _, z = abmil_model(features)
            z = z.squeeze().numpy()
            z = (z - z_mean) / z_std
            x = np.concatenate([z, clinical_lookup[patient]])
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            logits = fusion_model(x_tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            patients.append(patient)
            all_probs.append(probs)
            all_labels.append(row["label"])

    return patients, np.array(all_probs), np.array(all_labels)


def align_by_patient(patients_a, probs_a, labels_a, patients_b, probs_b, labels_b):
    """Inner join on patient ID -- only pair on patients BOTH models scored.
    Also sanity-checks that labels agree for every shared patient (they
    should, since both come from the same splits.csv)."""
    common = sorted(set(patients_a) & set(patients_b))
    idx_a = {p: i for i, p in enumerate(patients_a)}
    idx_b = {p: i for i, p in enumerate(patients_b)}

    aligned_probs_a = np.array([probs_a[idx_a[p]] for p in common])
    aligned_probs_b = np.array([probs_b[idx_b[p]] for p in common])
    aligned_labels_a = np.array([labels_a[idx_a[p]] for p in common])
    aligned_labels_b = np.array([labels_b[idx_b[p]] for p in common])

    mismatches = (aligned_labels_a != aligned_labels_b).sum()
    if mismatches > 0:
        raise ValueError(f"{mismatches} patients have disagreeing labels between the two "
                          f"prediction sets -- something is wrong with the alignment.")

    print(f"Aligned on {len(common)} shared test patients "
          f"(model A had {len(patients_a)}, model B had {len(patients_b)})")
    return aligned_probs_a, aligned_probs_b, aligned_labels_a


def stratified_bootstrap_indices(labels, n_classes):
    indices = []
    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        if len(class_idx) == 0:
            continue
        sampled = np.random.choice(class_idx, size=len(class_idx), replace=True)
        indices.extend(sampled)
    return np.array(indices)


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


def paired_bootstrap(probs_abmil, probs_fusion, labels, num_classes, compute_fn,
                      threshold_abmil=None, threshold_fusion=None):
    """Draws ONE resample per iteration, applies it to BOTH models, and
    collects the distribution of (fusion_metric - abmil_metric)."""
    kwargs_abmil = {"threshold": threshold_abmil} if threshold_abmil is not None else {}
    kwargs_fusion = {"threshold": threshold_fusion} if threshold_fusion is not None else {}

    point_abmil = compute_fn(probs_abmil, labels, **kwargs_abmil)
    point_fusion = compute_fn(probs_fusion, labels, **kwargs_fusion)
    point_diff = tuple(f - a for f, a in zip(point_fusion, point_abmil))
    print(f"  [79-patient ABMIL point estimate] bal_acc={point_abmil[0]:.4f}  auc={point_abmil[1]:.4}  auc={point_abmil[1]:.4f}  auprc={point_abmil[2]:.4f}  f1={point_abmil[3]:.4f}")

    diffs = {"bal_acc": [], "auc": [], "auprc": [], "f1": []}
    metric_names = ["bal_acc", "auc", "auprc", "f1"]

    for i in range(N_BOOTSTRAP):
        idx = stratified_bootstrap_indices(labels, num_classes)
        b_labels = labels[idx]
        try:
            m_abmil = compute_fn(probs_abmil[idx], b_labels, **kwargs_abmil)
            m_fusion = compute_fn(probs_fusion[idx], b_labels, **kwargs_fusion)
            if any(np.isnan(x) for x in m_abmil) or any(np.isnan(x) for x in m_fusion):
                continue
            for name, a, f in zip(metric_names, m_abmil, m_fusion):
                diffs[name].append(f - a)
        except Exception:
            continue

    def ci(arr):
        arr = np.array(arr)
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    results = {}
    for name, point in zip(metric_names, point_diff):
        lo, hi = ci(diffs[name])
        significant = (lo > 0) or (hi < 0)
        results[name] = {"point_diff": float(point), "95_ci": [float(lo), float(hi)],
                          "significant": bool(significant)}
        sig_str = "SIGNIFICANT" if significant else "not significant"
        print(f"  {name:10s}: diff = {point:+.4f}  95% CI = [{lo:+.4f}, {hi:+.4f}]  ({sig_str})")

    results["n_valid"] = len(diffs["auc"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["multiclass", "ret_binary"], required=True)
    parser.add_argument("--fusion_fc_dim", type=int, required=True)
    parser.add_argument("--fusion_dropout", type=float, default=0.0)
    parser.add_argument("--baseline_threshold", type=float, default=0.11,
                         help="Baseline's own calibrated optimal RET threshold (default 0.11, "
                              "per Table 5.1.2 -- update if this changes).")
    parser.add_argument("--compare", choices=["fusion", "baseline", "both"], default="both",
                         help="Which model(s) to compare against ABMIL.")
    args = parser.parse_args()

    print(f"Loading ABMIL predictions for {args.task}...")
    p_a, probs_a, labels_a = get_abmil_predictions(args.task)

    all_results = {}

    if args.compare in ("fusion", "both"):
        print(f"Loading Fusion predictions for {args.task}...")
        p_f, probs_f, labels_f = get_fusion_predictions(args.task, args.fusion_fc_dim, args.fusion_dropout)
        probs_abmil, probs_fusion, labels = align_by_patient(p_a, probs_a, labels_a, p_f, probs_f, labels_f)

        if args.task == "multiclass":
            print(f"\n{'='*60}\nPaired bootstrap: ABMIL vs Fusion -- Aim (a) Multiclass\n{'='*60}")
            all_results["abmil_vs_fusion_multiclass"] = paired_bootstrap(
                probs_abmil, probs_fusion, labels, 3, compute_metrics_multiclass)
        else:
            with open(os.path.join(FUSION_RESULTS_DIR, "fusion_results_ret_binary.json")) as f:
                fusion_threshold = json.load(f)["threshold_calibration"]["threshold"]
            print(f"\n{'='*60}\nPaired bootstrap: ABMIL vs Fusion -- Aim (b) RET "
                  f"(each model's own calibrated optimal threshold: "
                  f"ABMIL=0.13, Fusion={fusion_threshold})\n{'='*60}")
            all_results["abmil_vs_fusion_ret_optimal"] = paired_bootstrap(
                probs_abmil, probs_fusion, labels, 2, compute_metrics_binary,
                threshold_abmil=0.13, threshold_fusion=fusion_threshold)

    if args.compare in ("baseline", "both"):
        print(f"Loading Baseline predictions for {args.task}...")
        p_b, probs_b, labels_b = get_baseline_predictions(args.task)
        probs_abmil2, probs_baseline, labels2 = align_by_patient(p_a, probs_a, labels_a, p_b, probs_b, labels_b)

        if args.task == "multiclass":
            print(f"\n{'='*60}\nPaired bootstrap: ABMIL vs Baseline -- Aim (a) Multiclass\n{'='*60}")
            all_results["abmil_vs_baseline_multiclass"] = paired_bootstrap(
                probs_abmil2, probs_baseline, labels2, 3, compute_metrics_multiclass)
        else:
            # Baseline DOES have its own calibrated threshold (0.11, per
            # your corrected Table 5.1.2) -- exposed as a CLI arg rather
            # than hardcoded, since this script cannot itself calibrate it
            # (that calibration happens in whichever script produced
            # Table 5.1.2, not baseline_mean_pool_v2.py).
            print(f"\n{'='*60}\nPaired bootstrap: ABMIL vs Baseline -- Aim (b) RET "
                  f"(ABMIL at calibrated optimal=0.13, Baseline at calibrated "
                  f"optimal={args.baseline_threshold})\n{'='*60}")
            all_results["abmil_vs_baseline_ret"] = paired_bootstrap(
                probs_abmil2, probs_baseline, labels2, 2, compute_metrics_binary,
                threshold_abmil=0.13, threshold_fusion=args.baseline_threshold)

    out_path = os.path.join(FUSION_RESULTS_DIR, f"paired_bootstrap_{args.task}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
