# Design decisions

Why the code is organised the way it is. The scientific methods are described in the
dissertation; this note covers the engineering.

## A package plus scripts, not only scripts

The dissertation code was about forty standalone scripts. Each one redefined the ABMIL model,
the metrics, the way a patient's embedding file is found and the cluster paths, so a change in
one place had to be copied by hand into the others, and copies drifted (for example, two
scripts chose a different slide than the rest for patients with several slides).

Now everything that more than one step needs lives once in `src/ptc_mil/`:

| Concern | Module |
| --- | --- |
| Label encodings per task | `data/labels.py` |
| Which slide a patient uses | `data/embedding_index.py`, `data/slide_files.py` |
| Train/validation/test split | `data/splits.py` |
| ABMIL model | `models/abmil.py` |
| Metrics, thresholds, bootstrap | `evaluation/` |

The downstream analyses stay as scripts in `analyses/`. They are one-off scientific analyses
that are read top to bottom, so they keep their original structure and only take their data
access and model loading from the package.

## Steps communicate through files

Every `ptc-mil` step reads its inputs from and writes its outputs to `results_dir`
(`checkpoints/`, `predictions/`, `metrics/`, ...). This means:

- any step can be re-run alone, e.g. re-evaluating without retraining;
- each model's test predictions are computed **once**, saved as a CSV, and every later step
  (bootstrap, paired comparison, figures) reads that file. In the dissertation code, inference
  was repeated in several scripts, and refitting the baseline in one of them gave different
  probabilities from the saved ones.

## Configuration in two files

- `configs/experiment.yaml` (committed) holds every setting that changes the results: the
  selected hyperparameters, the tuning grid, thresholds, bootstrap settings and seeds. A
  reviewer can see all of them in one place.
- `configs/paths.yaml` (not committed) holds where the data is on a given machine.

Keeping them apart means the experiment definition is version-controlled while nobody's local
paths are.

## Validation-only model selection is enforced by structure

Tuning and training only ever receive the training and validation datasets. Test predictions
are written by `predict` after training has finished, and thresholds are selected from the
validation prediction file. There is no code path in which a test-set number influences a
choice.

## Faithful reimplementation, checked by tests

The goal of the refactor was to change the engineering, not the methods. Where it matters,
tests pin the new code to the original behaviour. For example, `tests/test_bootstrap.py`
checks that the bootstrap draws exactly the same resamples as the dissertation's script for the
same seed. Every place where the new code deliberately differs is listed in
[known_issues.md](known_issues.md).

## Tests on synthetic data

The TCGA data cannot be redistributed, so `tests/conftest.py` builds a small synthetic dataset
(random tile embeddings with a class-dependent shift, named like real TCGA slides) and the
end-to-end test runs every step from the split to the paired comparison on it, followed by the
analysis scripts that need neither OpenSlide nor HoVer-Net. This catches broken wiring between
steps, which unit tests alone do not.

## Optional dependency groups

Training and evaluation only need PyTorch, scikit-learn and h5py. OpenSlide, timm and
TIAToolbox are large, platform-specific and only needed for tiling, feature extraction and
the nucleus-level analyses, so they are optional extras (`[wsi]`, `[analysis]`) and imported
only inside the functions that use them.
