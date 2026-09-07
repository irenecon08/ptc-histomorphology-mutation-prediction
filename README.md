# Predicting Driver Alterations and Revealing Their Histomorphological Correlates in Papillary Thyroid Carcinoma

MSc Dissertation Project — Artificial Intelligence for Biomedicine and Healthcare, University College London

This project investigates the prediction of driver alterations in papillary thyroid carcinoma (PTC) from haematoxylin and eosin (H&E)-stained whole-slide images (WSIs). The study focuses on BRAF V600E, RAS, and RET fusion status using a frozen UNI2-h pathology foundation model for tile-level feature extraction and gated attention-based multiple instance learning (ABMIL) for slide-level prediction.

Beyond predictive performance, the project examines the contribution of the modelling components and investigates the tissue characteristics selected by the attention mechanism. This includes comparisons with mean pooling, multimodal fusion with clinical variables, TIIC density analysis, and characterisation of epithelial nuclear morphology.

Author: Irene Constantinidou
Primary supervisor: Dr Petru Manescu
Co-supervisor: Dr Karen Sayal (UCLH)
Institution: University College London

---

## Research Aims

The project addresses three main research aims:

1. **Predict driver-alteration status from unannotated WSIs.**
   Develop a weakly supervised pipeline that aggregates frozen tile-level features using gated attention, without requiring regional annotations, and evaluate BRAF V600E/RAS/Other and RET fusion prediction under an institutional holdout setting.

2. **Assess the contribution of individual pipeline components.**
   Compare learned attention pooling with unweighted mean pooling, and evaluate the contribution of routinely available clinical variables through multimodal fusion.

3. **Characterise the tissue selected by the attention mechanism.**
   Quantify the relationship between attention and tumour-infiltrating immune cell (TIIC) density in the held-out cohort, assess whether this relationship is independent of tissue composition, and characterise epithelial nuclear morphology in attended tissue across driver genotypes.

---

## Repository Overview

This repository contains the reference implementation used to produce the results and figures reported in the dissertation. The code is organised to document the experimental workflow and preserve the implementation details underlying the reported analyses.

The TCGA whole-slide images must be obtained through the appropriate GDC data-access procedures, while the original experiments were conducted on UCL departmental infrastructure using environment-specific paths.

---

## Dataset

Experiments use the TCGA-THCA (Thyroid Carcinoma) cohort, comprising 519 slides from 505 labelled patients.

- BRAF and RAS labels were obtained from GDC Mutation Annotation Format (MAF) files.
- RET fusion labels were obtained from cBioPortal using the PanCancer Atlas 2018 thyroid carcinoma study.
- An institutional train/validation/test split was used, with one institution held out entirely as the test set.

No patient-level data are redistributed in this repository.

---

## Data and Pipeline Correction

An early version of the tiling pipeline used an incorrect downsample factor during tile extraction, resulting in only approximately 53% of the tissue area being covered.

The issue was identified and corrected before the final experiments. The corrected pipeline was subsequently used to:

1. re-tile all slides;
2. re-extract all tile embeddings;
3. repeat model training and hyperparameter selection;
4. regenerate evaluation results; and
5. repeat the downstream analyses.

All dissertation results are based on the corrected pipeline. Files containing `v2` or later in their filename correspond to analyses performed using this implementation.

`preprocessing/audit_downsamples.py` is retained as a diagnostic record of the original issue and is not part of the final execution pipeline.

---

## Repository Structure

```text
preprocessing/     Tiling, embedding extraction, and dataset splits
models/             Baseline and ABMIL training and hyperparameter tuning
evaluation/         Bootstrap confidence intervals and paired comparisons
interpretability/   Attention heatmaps, embeddings, and tumour morphology
tiic_analysis/      TIIC correlation analyses and hypothesis testing
fusion/             Multimodal clinical-variable and ABMIL embedding fusion
figures/            Scripts used to generate preprocessing figures
```

---

## Experimental Pipeline

The main workflow is:

1. **Create dataset splits**
   `preprocessing/create_splits.py`
   Constructs the institutional train/validation/test split using patient-level labels.

2. **Tile WSIs**
   `preprocessing/retile_slides_fixed.py`
   Extracts tiles at approximately 10× magnification using the corrected downsample factor.

3. **Extract tile embeddings**
   `preprocessing/extract_embeddings_fixed.py`
   Extracts frozen UNI2-h representations, with a dimensionality of 1536 per tile.

4. **Train the baseline**
   `models/baseline_mean_pool_v2.py`
   Mean-pools tile embeddings into a slide-level representation and applies logistic regression.

5. **Tune ABMIL**
   `models/tune_abmil_v2_full.py` followed by `models/tune_abmil_v2_extra_lr.py`
   Performs a 9-configuration search followed by a 4-configuration low-learning-rate extension, giving 13 configurations in total.

   The earlier `models/tune_abmil.py` is retained for transparency. It contains a documented test-set leakage issue in model selection and was not used to determine the final hyperparameters.

6. **Train final ABMIL models**
   `models/train_abmil_v2_retrained.py`
   Trains the final multiclass and RET binary models using the corrected pipeline and selected hyperparameters.

7. **Calibrate the RET threshold**
   `models/optimal_threshold_ret_retuned.py`
   Selects the RET decision threshold using the validation set only.

8. **Evaluate predictive performance**
   The `evaluation/` scripts calculate bootstrap confidence intervals and paired comparisons between the baseline, ABMIL, and multimodal models.

9. **Run downstream analyses**
   The interpretability, TIIC, tumour morphology, and multimodal fusion analyses operate on the trained models and can be run independently.

---

## Key Results

RET metrics are reported at the validation-set-calibrated decision threshold. AUROC and AUPRC are threshold-independent.

| Task                          | Model                   | BalAcc | AUROC | AUPRC |    F1 |
| ----------------------------- | ----------------------- | -----: | ----: | ----: | ----: |
| Multiclass (BRAF/RAS/Other)   | Baseline                |  0.615 | 0.773 | 0.677 | 0.650 |
| Multiclass                    | ABMIL                   |  0.692 | 0.838 | 0.766 | 0.742 |
| Multiclass                    | ABMIL + clinical fusion |  0.729 | 0.835 | 0.760 | 0.762 |
| RET binary (threshold = 0.13) | ABMIL                   |  0.842 | 0.948 | 0.868 | 0.684 |
| RET binary                    | ABMIL + clinical fusion |  0.813 | 0.838 | 0.660 | 0.790 |

Bootstrap 95% confidence intervals for the reported metrics are provided in the outputs under `evaluation/` and are reported in full in the dissertation.

Clinical fusion did not produce a consistent improvement across metrics: it improved balanced accuracy and F1 for the multiclass task, while AUROC and AUPRC were slightly lower. For RET prediction, F1 improved at the selected threshold, whereas AUROC and AUPRC decreased. Overall, the results indicate a trade-off between operating-point performance and ranking performance, rather than a uniform benefit from clinical fusion.

---

## Analyses of the Learned Representation

### Attention embeddings

Attention-derived slide representations show separation between BRAF-mutant slides and the RAS/Other groups in UMAP space, while RAS and Other samples exhibit substantial overlap.

For RET fusion status, no clear two-dimensional separation is observed, consistent with the small number of RET-positive cases (33/505).

---

## Attention-Based Tissue Analysis

### TIIC analysis

Attention scores show a negative correlation with local immune-cell density across 80 test slides, using full-coverage 4×4 sub-crop sampling at native 40× resolution.

After controlling for epithelial cell density, the association is no longer significant for the multiclass model:

```text
partial r = -0.0095, p = 0.190
```

and is borderline for the RET binary model:

```text
partial r = -0.0141, p = 0.051
```

These results suggest that the observed negative attention–immune correlation is largely explained by attention being concentrated in epithelial/tumour-dense regions, rather than by an independently learned immune-microenvironment signal.

### Epithelial morphology

The morphology analysis characterises epithelial nuclear features within attended tissue and examines their distribution across driver genotypes. This provides a complementary assessment of the histomorphological characteristics associated with regions selected by the learned attention mechanism.

---

## Known Limitations

- **Test-set leakage in `models/tune_abmil.py`:** this earlier tuning script contains a test-set leakage issue during model selection. It is retained for transparency but is superseded by the `v2` tuning scripts.
- **Unpreserved supporting analysis:** one supporting analysis associated with the immune-molecular hypothesis testing (H3 in the dissertation) was executed and its outputs retained, but the exact generating script was not preserved.
- **Large and restricted data:** trained model weights, TCGA WSIs, tile datasets, and extracted embeddings are not included due to their size and, for TCGA data, applicable data-use restrictions.
- **Infrastructure dependence:** the original experiments were run on UCL departmental infrastructure, so paths and some environment-specific configuration require modification on other systems.

---

## Data Access

Whole-slide images and associated clinical/genomic data are available through the GDC Data Portal under the applicable TCGA data-use conditions.

RET fusion labels were obtained from cBioPortal, using the `thca_tcga_pan_can_atlas_2018` study.

This repository does not redistribute patient-level data.

---

## Environment

The experiments were developed and executed on UCL Computer Science departmental GPU infrastructure using Rocky Linux 9 and Python 3.10 for the duration of GPU access.

Exact package versions are pinned in `requirements.txt`, recovered directly from the project's `venv310` virtual environment. The two most version-sensitive dependencies are:

- `torch==2.5.1+cu121`
- `tiatoolbox==2.0.1`

The latter is used with the `hovernet_fast-monusac` model weights for TIIC nucleus detection.

After GPU access ended, a small number of scripts with no GPU or deep-learning dependencies were run on the CPU-only login node instead, out of necessity rather than by design:

- `preprocessing/create_splits.py`
- `evaluation/genotype_stratified_slide_selection.py`
- `evaluation/plot_paired_forest.py`

The pinned environment therefore reflects the environment used for the main GPU-based experiments.

---

## Citation

> Constantinidou, I. (2026). *Predicting Driver Alterations and Revealing Their Histomorphological Correlates in Papillary Thyroid Carcinoma*. MSc Dissertation, University College London.

---

## License

MIT — see `LICENSE`.
