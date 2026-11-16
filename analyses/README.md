# Downstream analyses

Scripts for the analyses in the dissertation that go beyond predictive performance. They run
after the main pipeline (`ptc-mil train`, `predict`, `evaluate`, `compare`) and read its
outputs. Each script is run directly from the repository root:

```bash
pip install -e ".[wsi,analysis]"
python analyses/tiic/tiic_correlation.py --task both
```

The scripts find the data through the same configuration as the pipeline. By default they
read `configs/paths.yaml` and `configs/experiment.yaml`; set `PTC_MIL_PATHS` and
`PTC_MIL_EXPERIMENT` to use other files. Outputs go to `results/analyses/<name>/` (or the
`analyses_dir` set in `paths.yaml`). Every script's docstring describes its method and usage.

## TIIC: attention and immune cells (`tiic/`)

| Script | What it does | Needs |
| --- | --- | --- |
| `tiic_correlation.py` | Per-slide and pooled Spearman and partial correlations between attention and immune density on all test slides (HoVer-Net, full 4×4 sub-crop coverage). Resumable. | GPU, TIAToolbox |
| `summarise_tiic.py` | Summary of the correlations, and residuals by genotype and RET status | outputs above |
| `hypothesis1_checks.py`, `hypothesis1_compositional_checks.py` | Is the negative correlation a compositional artefact of immune and epithelial density sharing the same tile? | outputs above |
| `hypothesis4_checks.py` | Is the correlation a detection artefact (fewer nuclei found in dense epithelium)? | outputs above |
| `hypothesis2_select_slides.py` | Selects the slides with the strongest residual correlation | outputs above |
| `hypothesis2_spatial_analysis.py` | Are immune cells spatially excluded from high-attention regions? Every tile of the selected slides. | GPU, TIAToolbox |
| `hypothesis2_heatmap_figures.py` | Attention, immune and epithelial density heatmaps for those slides | OpenSlide |
| `genotype_stratified_slide_selection.py` | Genotype-balanced slide sample for the morphology and spatial analyses | splits |
| `raw_immune_density_by_genotype.py` | Immune density by genotype from HoVer-Net counts | TIIC outputs |
| `mutation_immune_comparison.py` | Molecular immune infiltration (Thorsson et al. 2018) by genotype | `thorsson_csv` |
| `tiic_subcrop_validation.py` | Sensitivity of the correlations to sub-crop coverage | GPU, TIAToolbox |

Run order: `tiic_correlation` → `summarise_tiic` / `hypothesis*_checks` →
`hypothesis2_select_slides` → `hypothesis2_spatial_analysis --stage nucleus`, then
`--stage distance` → `hypothesis2_heatmap_figures`.

## Interpretability (`interpretability/`)

| Script | What it does |
| --- | --- |
| `generate_heatmaps.py` | Attention heatmaps for correctly and incorrectly classified test slides |
| `extract_attention_crops.py` | Full-resolution crops of the highest- and lowest-attention tiles |
| `embedding_analysis.py` | PCA and UMAP of the attention-pooled slide embeddings |
| `embedding_separation_analysis.py` | Permutation-tested k-NN label purity of those projections |
| `shape_descriptor_analysis.py` | Epithelial nuclear shape (area, circularity, solidity, eccentricity) by attention level and by genotype |
| `plot_shape_attention.py`, `plot_shape_genotype.py`, `recompute_table15_and_figure.py` | Figures and Table 15 from the shape descriptors |

## Clinical variables (`clinical/`)

| Script | What it does |
| --- | --- |
| `clinical_correlation_analysis.py` | Association of each clinical variable with genotype and RET status (BH-corrected) |
| `eda_fusion_features.py` | Missingness, rare categories and split consistency of the nine fusion variables |

## Evaluation figures (`evaluation/`)

| Script | What it does |
| --- | --- |
| `plot_paired_forest.py` | Forest plot of the paired bootstrap differences |
| `plot_abmil_vs_fusion.py` | Bootstrap distributions of ABMIL and fusion metrics |

## Other

- `figures/tiling_figure_assets.py`: panels for the tiling figure.
- `diagnostics/audit_downsamples.py`: reports each slide's level-1 downsample factor.
- `diagnostics/tiic_sanity_check.py`: checks run on one slide before the full TIIC analysis.
