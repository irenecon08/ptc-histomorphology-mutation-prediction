"""
Fusion classifier head hyperparameter search.

Mirrors the base ABMIL model's search protocol (Section 4.8 / Supplementary
Table X): an initial grid ranked by validation AUROC, followed by a smaller
extension around the winning config. Test set is NEVER touched during this
search -- only after a single final config is chosen should fusion_train.py
be run (as a plain CLI call with --lr/--fc_dim/--dropout set to the winning
values) to get the one official test result. There is no separate
"--evaluate_test" flag -- every CLI run of fusion_train.py evaluates test
exactly once by design, which is why hyperparameter search must go through
this script instead, never fusion_train.py directly.

Grid:
    Stage 1 (12 configs): lr in {1e-4, 1e-3, 1e-2} x fc_dim in {64, 128, 256, 512},
                          dropout fixed at 0.0 (matches base model's
                          dropout=0.0 as the starting point). 512 is included
                          because it's the base multiclass model's own
                          winning fc_dim (Supplementary Table X) -- no reason
                          to exclude testing a wider head here too.
    Stage 2 (2 configs):  best (lr, fc_dim) from Stage 1, dropout in {0.2, 0.4}
                          (0.0 already covered in Stage 1)

Total: 14 configs per task, ranked by validation AUROC.

Known limitations of this search (documented for the Methods write-up):
    - batch_size (16) and weight_decay (1e-3) are held fixed, not searched
    - this is a GREEDY/sequential search: dropout is only tested around the
      single Stage 1 winner, not against all 12 (lr, fc_dim) combos, so a
      dropout/architecture interaction could be missed. This mirrors the
      same limitation in the base ABMIL model's own extension step
      (Section 4.8), so it is at least consistent with existing precedent,
      not a new weakness introduced here.

Usage:
    python fusion_hparam_search.py --task multiclass
    python fusion_hparam_search.py --task ret_binary
    python fusion_hparam_search.py --task both
"""

import os
import argparse
import json
import pandas as pd

from fusion_train import setup_task, train_one_config, FUSION_RESULTS_DIR

LR_GRID = [1e-4, 1e-3, 1e-2]
FC_DIM_GRID = [64, 128, 256, 512]  # 512 included: base model's own winning fc_dim (multiclass)
DROPOUT_EXTENSION = [0.2, 0.4]  # 0.0 already covered in the main grid


def run_search(task):
    print(f"\n{'#'*60}\nHyperparameter search: {task}\n{'#'*60}")
    setup = setup_task(task)  # expensive setup done ONCE, reused for all 14 configs

    results = []

    # --- Stage 1: 3x3 grid, dropout=0.0 ---
    print(f"\n--- Stage 1: {len(LR_GRID)}x{len(FC_DIM_GRID)} grid (lr x fc_dim), dropout=0.0 ---")
    for lr in LR_GRID:
        for fc_dim in FC_DIM_GRID:
            print(f"\n[Stage 1] lr={lr}, fc_dim={fc_dim}, dropout=0.0")
            r = train_one_config(setup, fc_dim=fc_dim, dropout=0.0, lr=lr,
                                  verbose=False, evaluate_test=False)
            r["stage"] = 1
            results.append(r)
            print(f"  -> best val AUC: {r['best_val']['auc']:.4f} "
                  f"(bal_acc={r['best_val']['bal_acc']:.4f}, {r['n_epochs_trained']} epochs)")

    # --- Pick Stage 1 winner by validation AUROC ---
    stage1_results = [r for r in results if r["stage"] == 1]
    best_stage1 = max(stage1_results, key=lambda r: r["best_val"]["auc"])
    print(f"\nStage 1 winner: lr={best_stage1['lr']}, fc_dim={best_stage1['fc_dim']} "
          f"(val AUC={best_stage1['best_val']['auc']:.4f})")

    # --- Stage 2: dropout extension around the Stage 1 winner ---
    print(f"\n--- Stage 2: dropout extension at lr={best_stage1['lr']}, fc_dim={best_stage1['fc_dim']} ---")
    for dropout in DROPOUT_EXTENSION:
        print(f"\n[Stage 2] lr={best_stage1['lr']}, fc_dim={best_stage1['fc_dim']}, dropout={dropout}")
        r = train_one_config(setup, fc_dim=best_stage1["fc_dim"], dropout=dropout,
                              lr=best_stage1["lr"], verbose=False, evaluate_test=False)
        r["stage"] = 2
        results.append(r)
        print(f"  -> best val AUC: {r['best_val']['auc']:.4f} "
              f"(bal_acc={r['best_val']['bal_acc']:.4f}, {r['n_epochs_trained']} epochs)")

    # --- Rank all 14 configs together ---
    table = pd.DataFrame([{
        "rank": None, "stage": r["stage"], "lr": r["lr"], "fc_dim": r["fc_dim"],
        "dropout": r["dropout"], "val_auc": r["best_val"]["auc"],
        "val_bal_acc": r["best_val"]["bal_acc"], "val_f1": r["best_val"]["f1"],
        "n_epochs": r["n_epochs_trained"],
    } for r in results])
    table = table.sort_values("val_auc", ascending=False).reset_index(drop=True)
    table["rank"] = table.index + 1

    out_csv = os.path.join(FUSION_RESULTS_DIR, f"fusion_hparam_search_{task}.csv")
    table.to_csv(out_csv, index=False)
    print(f"\n{table.to_string(index=False)}")
    print(f"\nSaved full search table to {out_csv}")

    winner = table.iloc[0]
    print(f"\n>>> RECOMMENDED CONFIG for {task}: "
          f"lr={winner['lr']}, fc_dim={int(winner['fc_dim'])}, dropout={winner['dropout']} "
          f"(val AUC={winner['val_auc']:.4f})")
    print(f">>> To get the official test result, run:")
    print(f">>>   python fusion_train.py --task {task} --lr {winner['lr']} "
          f"--fc_dim {int(winner['fc_dim'])} --dropout {winner['dropout']}")

    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["multiclass", "ret_binary", "both"], default="both")
    args = parser.parse_args()

    os.makedirs(FUSION_RESULTS_DIR, exist_ok=True)
    tasks = ["multiclass", "ret_binary"] if args.task == "both" else [args.task]
    for t in tasks:
        run_search(t)

    print("\nHyperparameter search complete. Test set was NOT touched -- "
          "run fusion_train.py with the recommended config to get final test results.")
