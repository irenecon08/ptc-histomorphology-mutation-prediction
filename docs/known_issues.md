# Known issues and differences from the dissertation code

The refactor keeps the methods of the dissertation. This file lists every place where the
current code behaves differently from the code at the
[`dissertation-submission`](https://github.com/irenecon08/ptc-histomorphology-mutation-prediction/tree/dissertation-submission)
tag, why, and what effect to expect, followed by limitations of the study itself.

Re-running the pipeline will not reproduce the dissertation checkpoints bit-for-bit in any
case: GPU training is not fully deterministic, and the original ABMIL and fusion training
runs were not seeded. Results should agree within the reported confidence intervals.

## The tiling correction (part of the dissertation)

The first tiling run assumed that OpenSlide level 1 is a 2× downsample of level 0. On many
TCGA-THCA slides it is not, so tiles were read from the wrong positions and covered only about
53% of the tissue. `analyses/diagnostics/audit_downsamples.py` found this. Before the final
experiments every slide was re-tiled using each slide's actual downsample factor, and
embeddings, training, tuning, evaluation and all analyses were repeated. All dissertation
results, and this code, use the corrected tiling.

## Behaviour changes

### 1. One rule for patients with more than one slide

A few patients have several diagnostic slides; one slide per patient is used. The original
scripts did not choose consistently:

- most scripts used the first matching file in directory-listing order (`os.listdir`), which
  depends on the filesystem;
- the fusion, paired-bootstrap and embedding-analysis scripts built a dictionary from the
  listing, so they used the *last* matching file;
- the analysis scripts looked up the `.svs`, tile and embedding files separately, so for the
  same patient they could combine attention from one slide with pixels from another.

Now every step uses the **alphabetically first** slide (`data/embedding_index.py`), and the
`.svs` and tile files are found by that slide's name (`data/slide_files.py`), so all three
always match. Multi-slide patients are logged as a warning. For those patients results can
differ from the dissertation; for everyone else nothing changes.

### 2. Training is seeded

`train`, `tune` and `fusion-train` seed Python, NumPy and PyTorch from `seed` in
`configs/experiment.yaml` (42). The original training scripts set no seed. This only makes
re-runs repeatable; it does not change the method.

### 3. Bootstrap resamples are shared across thresholds, for every model

For the RET task, metrics are reported at the default (0.5) and the calibrated threshold. The
original ABMIL and baseline bootstrap scripts drew a new set of 2,000 resamples for each
threshold, so the threshold-independent AUROC and AUPRC intervals came out slightly different
for the two thresholds. This was noticed and fixed in `fusion_bootstrap_ci_fix.py` for the
fusion model only. The fix is now applied to every model: one set of resamples per model and
task, shared across thresholds. Each model and task also starts its own random stream from
seed 42 (the originals continued one stream from the multiclass task into the RET task).

Effect: three-class intervals are unchanged (a test checks the draws match the original
code). RET intervals for ABMIL and the baseline may differ from Tables 4 and 6 in the second or
third decimal place.

### 4. Baseline classifier

The three-class baseline used `LogisticRegression(multi_class="ovr")`. That argument has been
removed from scikit-learn, so the same model is now built as
`OneVsRestClassifier(LogisticRegression())`, which fits one binary model per class and
normalises their probabilities in the same way. The RET baseline is unchanged (binary
logistic regression with balanced class weights).

The dissertation text describes the three-class baseline as "multinomial"; the code (and now
this repository) uses one-vs-rest.

### 5. Thresholds

- **ABMIL (0.13):** selected on validation balanced accuracy over 0.01–0.54, as in the original
  script. `ptc-mil calibrate --model abmil` reproduces the procedure.
- **Baseline (0.11):** the script that selected this threshold was not kept in the original
  repository. `calibrate --model baseline` applies the same procedure as for ABMIL (same grid),
  and warns if the result differs from the 0.11 in `experiment.yaml`, which is the value used
  for evaluation.
- **Fusion (0.02):** selected by `calibrate --model fusion` over 0.01–0.99, as in the original
  fusion script, and read from its output when evaluating.

### 6. Fusion hyperparameters

The original repository did not record which fusion configuration won the search (it printed a
recommendation to pass by hand to the training script). `fusion-train` now takes the
configuration from `fusion.selected` in `experiment.yaml` if it is set, otherwise from the top
row of the search table written by `fusion-search`.

### 7. Hyperparameter search size

The dissertation reports 11 ABMIL configurations: 3 learning rates × 3 model sizes, plus two
smaller learning rates at the default size (`tune_abmil_v2_full.py`). A later script
(`tune_abmil_v2_extra_lr.py`) added two even smaller learning rates, giving 13; the previous
README described 13. Both select the same configurations. The default `ptc-mil tune` runs the
11 reported in the dissertation; `--extended` adds the other two.

### 8. Superseded scripts not carried over

Available at the tag, but not in the current code:

- `models/tune_abmil.py`: the first tuning run, which selected models using the test set
  (test-set leakage). It was superseded before the final results and is not used.
- `bootstrap_ci_v2.py`, `bootstrap_ci_baseline_v2.py`: earlier versions without saved
  predictions.
- `figures/extract_figure_assets.py`, `figures/extract_background_and_grid_v3.py`: earlier drafts
  of the tiling figure (`analyses/figures/tiling_figure_assets.py` is the final one).
- The "v5 vs v6" comparison in the TIIC script, which compared the final results with a
  superseded run that used the wrong checkpoints.

### 9. Analysis scripts adapted to the final pipeline

- `analyses/tiic/tiic_subcrop_validation.py` imported `tiic_correlation_v4`, which was not in
  the repository. It validated the earlier 2×2 sub-crop design (25% of each tile) by
  re-running some slides at full 4×4 coverage. The final analysis uses 4×4, so the script now
  runs the comparison the other way round: it re-runs the same tiles at 2×2 and compares with
  the final results.
- `analyses/diagnostics/tiic_sanity_check.py` loaded an early checkpoint
  (`best_multiclass_v2.pt`, before re-tuning). It now uses the final multiclass model.
- The evaluation plots read the new metrics and comparison files instead of the old JSON
  layout.

## Things kept as they were, worth knowing

- **RET loss weights** for ABMIL come from fixed class counts (319 negative, 21 positive →
  weights 0.53 and 8.10), as in the original. The fusion model computes its weights from its
  own training patients, also as in the original.
- **Undefined validation AUROC** (a validation set with one class) counts as 0 for early
  stopping, as in the original. It does not occur with the real split.
- **Spatial distances** in the Hypothesis 2 analysis use an approximate 0.99 µm per pixel at
  level 1 rather than each slide's own resolution.
- **Embedding separation analysis:** the docstring describes eight permutation tests in the
  primary battery, but the code runs four (UMAP purity by site and by split, for each task);
  purity by genotype or RET status is tested in the k-sweep. The code is kept as it was run.
- **Fusion search is greedy:** dropout is only tried around the stage-1 winner, and batch size
  and weight decay are fixed.

## Limitations of the study

- **Small positive class:** 33 of 505 patients are RET-positive, six in the test set, so RET
  confidence intervals are wide.
- **Single external institution:** the test set is one institution (site EL); performance on
  other institutions is not established.
- **Missing script for one supporting analysis:** an analysis for the immune-molecular
  hypothesis (H3 in the dissertation) was run and its outputs kept, but the script that
  produced it was not preserved.
- **No data or weights in the repository:** TCGA slides, tiles, embeddings and trained
  checkpoints are not included because of their size and the TCGA data-use conditions.
