# Predicting driver alterations in papillary thyroid carcinoma from H&E slides

[![CI](https://github.com/irenecon08/ptc-histomorphology-mutation-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/irenecon08/ptc-histomorphology-mutation-prediction/actions/workflows/ci.yml)

Code for my MSc dissertation (Artificial Intelligence for Biomedicine and Healthcare, UCL, 2026):
predicting BRAF V600E, RAS and RET-fusion status in papillary thyroid carcinoma (PTC) from
whole-slide H&E images, and characterising the tissue the model attends to.

The pipeline tiles each slide, embeds every tile with the frozen
[UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) pathology foundation model, and aggregates
the tile embeddings with gated attention-based multiple instance learning (ABMIL,
[Ilse et al., 2018](https://arxiv.org/abs/1802.04712)). It is compared with a mean-pooling
baseline and with a multimodal model that adds routine clinical variables, on an institutional
hold-out test set.

Author: Irene Constantinidou · Supervisors: Dr Petru Manescu (UCL), Dr Karen Sayal (UCLH)

---

## Results

TCGA-THCA, 505 labelled patients; all 80 patients from one institution (site EL) are held out
as the test set. Point estimates with stratified bootstrap 95% CIs (2,000 resamples). RET rows
use the decision threshold selected on the validation set.

| Task | Model | Balanced acc. | AUROC | AUPRC | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| BRAF V600E / RAS / Other | Mean-pool baseline | 0.616 (0.481–0.760) | 0.773 (0.664–0.877) | 0.677 (0.557–0.821) | 0.650 (0.492–0.791) |
| | **ABMIL** | 0.692 (0.568–0.819) | **0.838** (0.740–0.930) | **0.766** (0.647–0.897) | 0.742 (0.591–0.856) |
| | ABMIL + clinical | 0.729 (0.600–0.853) | 0.835 (0.733–0.927) | 0.760 (0.638–0.888) | 0.762 (0.634–0.874) |
| RET fusion | Mean-pool baseline (t = 0.11) | 0.842 (0.662–0.953) | 0.838 (0.550–0.998) | 0.690 (0.340–0.976) | 0.684 (0.572–0.812) |
| | **ABMIL** (t = 0.13) | 0.842 (0.669–0.953) | **0.948** (0.831–1.000) | **0.868** (0.597–1.000) | 0.684 (0.577–0.792) |
| | ABMIL + clinical (t = 0.02) | 0.813 (0.632–0.986) | 0.838 (0.575–1.000) | 0.660 (0.328–1.000) | 0.791 (0.624–0.951) |

- In a paired bootstrap on the same patients, ABMIL beat the baseline on the three-class task
  for AUROC (+0.065, 95% CI 0.014–0.128) and AUPRC (+0.089, 0.009–0.181). No other
  difference was significant; the RET test set has only six positives.
- Adding clinical variables did not measurably help either task.
- Attention is negatively correlated with immune-cell density, but this is almost entirely
  explained by attention concentrating on epithelium-dense tumour regions (partial
  correlation controlling for epithelial density: r = −0.010, p = 0.19 for the three-class
  model).

The fusion rows use the 79 test patients with clinical data.

---

## Repository layout

```text
src/ptc_mil/            the installable package
  data/                 labels, splits, slide/embedding lookup, datasets, clinical features
  preprocessing/        tiling and UNI2-h embedding extraction
  models/               ABMIL and the mean-pooling baseline
  training/             ABMIL training and hyperparameter search
  fusion/               multimodal fusion head, training and search
  evaluation/           metrics, threshold selection, bootstrap CIs, prediction tables
  analysis/             shared context used by the scripts in analyses/
  pipeline.py, cli.py   the steps of the experiment and the `ptc-mil` command
analyses/               downstream analyses from the dissertation (TIIC, interpretability,
                        clinical, figures), run as scripts; see analyses/README.md
configs/                experiment settings (committed) and local paths (not committed)
tests/                  unit tests and an end-to-end run on synthetic data
docs/                   known issues, original environment
```

---

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/irenecon08/ptc-histomorphology-mutation-prediction.git
cd ptc-histomorphology-mutation-prediction
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # training and evaluation on extracted embeddings
pip install -e ".[wsi]"            # + tiling and UNI2-h extraction (OpenSlide, timm)
pip install -e ".[analysis]"       # + downstream analyses (TIAToolbox, OpenCV, UMAP)
```

Check the installation (no data needed; runs the whole pipeline on synthetic slides):

```bash
pytest
```

---

## Reproducing the experiments

### Data

- Whole-slide images: TCGA-THCA diagnostic slides from the [GDC Data Portal](https://portal.gdc.cancer.gov/).
- BRAF and RAS labels from the GDC mutation annotation files; RET-fusion labels and clinical
  variables from cBioPortal (`thca_tcga_pan_can_atlas_2018`).
- Immune estimates for the molecular comparison from Thorsson et al. (2018), Table S1.

No patient-level data is redistributed here. Copy `configs/paths.example.yaml` to
`configs/paths.yaml` and point it at your copies.

### Steps

Each step reads and writes files under `results_dir`, so any step can be re-run on its own.
All hyperparameters are in `configs/experiment.yaml` and are set to the dissertation values.

```bash
ptc-mil tile                          # 256 px tiles at level 1, background removed
ptc-mil embed                         # UNI2-h embedding per tile (GPU)
ptc-mil split                         # site EL = test, rest split 80/20 (seed 42)

ptc-mil baseline                      # mean pooling + logistic regression
ptc-mil tune                          # 11-configuration ABMIL search on validation AUROC
ptc-mil train                         # final ABMIL model per task
ptc-mil predict                       # validation and test predictions, saved once
ptc-mil calibrate --model abmil       # RET threshold on validation balanced accuracy
ptc-mil calibrate --model baseline
ptc-mil evaluate --model abmil        # test metrics + bootstrap CIs
ptc-mil evaluate --model baseline

ptc-mil clinical                      # nine clinical variables, train-only imputation
ptc-mil fusion-search                 # 14-configuration fusion head search
ptc-mil fusion-train
ptc-mil calibrate --model fusion
ptc-mil evaluate --model fusion

ptc-mil compare --other baseline      # paired bootstrap vs ABMIL
ptc-mil compare --other fusion
```

Run `ptc-mil <step> --help` for options, e.g. `--task ret_binary` to run one task.
The downstream analyses are described in [analyses/README.md](analyses/README.md).

---

## Methods in brief

- **Tiling:** non-overlapping 256×256 tiles at OpenSlide level 1; a tile is background when
  more than 85% of its pixels are above 220 in every channel. Level coordinates are converted
  with each slide's actual downsample factor (see [docs/known_issues.md](docs/known_issues.md)).
- **Features:** frozen UNI2-h (ViT-H/14), 1536-dimensional embedding per tile.
- **ABMIL:** gated attention pooling, then a two-layer classifier. Adam (weight decay 1e-3),
  one slide per step, learning rate halved on a validation-AUROC plateau, early stopping on
  validation AUROC (patience 10). RET uses inverse-frequency class weights.
- **Model selection:** learning rate and model size chosen on validation AUROC only. The RET
  decision threshold is chosen on validation balanced accuracy and applied once to the test set.
- **Fusion:** the frozen ABMIL slide embedding, z-normalised on training patients, is
  concatenated with nine clinical variables (age, sex, race, ethnicity, overall/T/N/M stage,
  histological subtype) and passed to a new two-layer head.
- **Uncertainty:** stratified bootstrap (resampling within each class) for CIs, and a paired
  bootstrap on shared patients for model comparisons.

What differs from the code that produced the dissertation, and known limitations:
[docs/known_issues.md](docs/known_issues.md).

---

## History

The code as submitted with the dissertation is preserved at the
[`dissertation-submission`](https://github.com/irenecon08/ptc-histomorphology-mutation-prediction/tree/dissertation-submission)
tag. The current version reorganises it into a tested package without changing the methods.
The exact package versions of the original GPU environment are in
[docs/dissertation_environment.txt](docs/dissertation_environment.txt).

## Citation

> Constantinidou, I. (2026). *Predicting Driver Alterations and Revealing Their
> Histomorphological Correlates in Papillary Thyroid Carcinoma*. MSc Dissertation,
> University College London.

## License

MIT, see [LICENSE](LICENSE).
