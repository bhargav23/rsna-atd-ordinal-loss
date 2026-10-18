# Ordinal Present-Label Upweighting for Multi-Organ Abdominal Trauma Severity Classification

Code for the M.Tech thesis of **Taraka Sriram Gavireddi** (guide: **Kompella Bhargava Kiran**) and
the paper *Ordinal Present-Label Upweighting for Multi-Organ Abdominal Trauma Severity
Classification* (10th International Conference on I-SMAC, 16–18 September 2026, held online;
accepted and presented, not yet published or indexed).

The work classifies the severity of abdominal trauma on CT scans from the
[RSNA 2023 Abdominal Trauma Detection](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)
competition. Every number in the thesis comes from the scripts in this repository, run as Kaggle
notebooks on a free-tier Tesla T4 GPU.

## Pipeline

| Phase | Script | What it does |
|---|---|---|
| 0 | `pipeline/phase0_preprocessing.py` | Builds a 1,200-patient subset, decodes and windows DICOM series, crops liver, spleen, kidney, and bowel through a mask-to-DICOM coordinate transform, saves 64-slice full volumes, and backfills full volumes for series that already had four crops |
| 1a/1b | `pipeline/phase1_organ_branch.py` | Organ-crop branch: EfficientNet-B0 with slice attention pooling, baseline cross-entropy and the ordinal present-label upweighting loss, patient-level 5-fold cross-validation, out-of-fold export |
| 1c | `pipeline/phase1c_fullvolume_branch.py` | Full-volume branch for bowel injury and extravasation: multi-label focal loss, 5-fold cross-validation, cross-fold logistic calibration |
| 2 | `pipeline/phase2_stacking.py` | Patient-level LightGBM stacking, the hybrid pipeline, and the competition-style weighted log loss |

Cohort sizes: 1,200 patients and 1,803 series after Phase 0; 206 organ-crop series from 128
patients; 1,554 full-volume series from 1,028 patients after the backfill.

## Repository layout

```
pipeline/    the four scripts above (reported experiments)
analysis/    eleven scripts for the checks, comparisons, tables, and figures
notebooks/   three early development notebooks (single-split and mixed-precision runs)
```

## Data

No data is included, and none may be redistributed: the images and labels belong to the
competition. Download them from the competition page under its rules. The scripts read the
competition files from `/kaggle/input/competitions/rsna-2023-abdominal-trauma-detection` and three
**derived** Kaggle datasets that the authors created from them and kept private:

| Dataset | Content | Made by |
|---|---|---|
| `rsna-atd-phase0-preprocessed` | `.npz` array per series, `manifest.csv`, `failed_log.csv`, `subset_labels.csv` | Phase 0 |
| `rsna-atd-backfilled-fullvolume` | Arrays with a full volume, `backfill_log.csv` | Phase 0 backfill |
| `rsna-atd-fv-oof-v3` | Out-of-fold predictions (`organ_oof_probabilities.csv`, `fv_oof_probabilities_v3.csv`, and the comparison files) | Phases 1 and 1c, analysis scripts |

To use your own copies, replace `<username>` in the paths (`grep -rn "<username>" .` lists them)
with your Kaggle username, or point the constants at local folders.

## Environment

The runs used Kaggle notebooks (Python 3.12.13, Tesla T4). The library versions were **not
recorded** at the time, so `requirements.txt` is unpinned; see Appendix C of the thesis. Phase 0
needs no GPU. Phases 1 and 1c need one (five epochs per fold, five folds).

```
pip install -r requirements.txt
```

## Running the pipeline

The scripts are written as sequences of notebook cells with a `__main__` block. In a Kaggle
notebook (internet on, derived datasets added as inputs):

```python
!git clone https://github.com/<your-account>/<this-repository>.git
%cd <this-repository>/pipeline
```

1. **Phase 0.** Install `pylibjpeg pylibjpeg-libjpeg pylibjpeg-rle python-gdcm nibabel`, then run
   `python phase0_preprocessing.py`. It resumes from `manifest.csv` and `failed_log.csv` if the
   session is interrupted. Call `run_full_volume_backfill` for the series that have four crops
   (see "Known gaps" for two lines you must add first).
2. **Phase 1a/1b.** `python phase1_organ_branch.py` writes `organ_oof_probabilities.csv` using the
   final configurations (liver: baseline; spleen: ordinal, α = 3.0; kidney: ordinal, α = 1.0, both
   with boost 1.0). Finished folds are reused if the output file exists.
3. **Phase 1c.** `python phase1c_fullvolume_branch.py` (imports from `phase1_organ_branch`, so run
   it from `pipeline/`). The reported cohort of 1,554 full-volume series needs the backfill:
   call `load_manifest_and_labels(backfill_dir=Path(".../rsna-atd-backfilled-fullvolume"))`
   instead of the default, which the `__main__` block uses.
4. **Phase 2.** `python phase2_stacking.py` prints the full-stack and hybrid scores.

## Analysis scripts

| Script | Purpose | Output |
|---|---|---|
| `generate_baseline_oof.py` | Baseline cross-entropy run for spleen and kidney (seed 42, five epochs) | `organ_baseline_oof.csv` |
| `generate_liver_ordinal_oof.py` | Liver run with the ordinal loss (α = 0.5, boost 1.5) | `liver_ordinal_oof.csv` |
| `generate_liver_baseline_seed2.py` | Liver baseline rerun with seed 123 | `liver_baseline_seed123_oof.csv` |
| `generate_bowel_organcrop_oof.py` | Bowel classifier on the organ crops | `bowel_organcrop_oof.csv` |
| `train_baseline_comparisons.py` | LDAM with deferred re-weighting and weighted focal loss | `organ_ldam_oof.csv`, `organ_weightedfocal_oof.csv` |
| `build_baseline_comparison_table.py` | Accuracy and recall of the four configurations | `baseline_comparison_table.csv` |
| `significance_testing.py` | Wilcoxon signed-rank test, paired by fold | `significance_test_results.csv` |
| `recompute_spleen_recall.py` | Spleen recalls per fold from the stored files | printed |
| `generate_fold_tables.py` | Class counts of each held-out fold | CSV files and a Markdown table |
| `verify_liver_counts.py` | Compares high-grade liver counts in the labels and the predictions | printed |
| `generate_figures.py` | Architecture diagram, reliability diagrams, confusion matrices | `figure1_architecture`, `figure2_calibration`, `figure3_confusion_matrices` (PNG, PDF) |

The five scripts that train models (`generate_baseline_oof`, `generate_liver_ordinal_oof`,
`generate_liver_baseline_seed2`, `generate_bowel_organcrop_oof`, `train_baseline_comparisons`) use
objects defined by `phase1_organ_branch.py` and must run in the same Python session. For example:

```python
import sys; sys.path.insert(0, "pipeline")
from phase1_organ_branch import *
manifest_df, labels_df, organ_manifest = load_manifest_and_labels()
organ_manifest_cv = make_patient_folds(labels_df, organ_manifest)

exec(open("analysis/generate_baseline_oof.py").read())
```

The other six scripts read the stored out-of-fold files and need no GPU.

## About the numbers

The reported results are from the final full-precision scripts. The development notebooks in
`notebooks/` (version 12: five-seed single-split confirmation; version 15: five-fold
cross-validation) use mixed precision (`torch.cuda.amp`) and give different means for the same
configurations, by up to 0.07 in accuracy (for example 0.458 against 0.530 for the kidney
baseline). They are different programs, not reruns, and are kept as a record. Single-split results
(version 9) were corrected by cross-validation and are reported in the thesis only to show the
size of that correction. See Chapter 6 and Appendix C of the thesis.

The scripts set all random seeds, but they do not enable deterministic GPU mode, so a rerun may
differ in the last printed digits.

## Known gaps

These are documented in Appendix C of the thesis and have **not** been patched here, so that the
code is what was run.

- The consolidated `phase0_preprocessing.py` does not save the label table that later phases read
  (the original notebook does). After `build_stratified_subset`, add:
  ```python
  subset_df.to_csv(OUTPUT_DIR / "subset_labels.csv", index=False)
  series_subset.to_csv(OUTPUT_DIR / "subset_series_meta.csv", index=False)
  ```
- Several docstrings quote values from earlier runs (for example the Phase 1c docstring and the
  closing message of `build_baseline_comparison_table.py`) that differ from the final results; and
  the Phase 1 docstring says "about 27" high-grade liver examples where the stored labels give
  31 series from 19 patients. The thesis reports the final values.
- The any-injury term of the weighted log loss is unweighted, although the official metric weights
  injured cases by 6, so scores here cannot be compared with the leaderboard.
- The Phase 0 configuration lists `image_level_labels_2024.csv`, `train_demographics_2024.csv`, and
  a blood-oriented CT window; the reported experiments do not use them.
- The configuration of the comparison losses (LDAM margin 0.5, scale 30, deferred weights from the
  fourth epoch; focal γ = 2.0) is in `train_baseline_comparisons.py`, not in Phase 1.

## Status and licence

The paper has been accepted and presented but is not yet published. No licence has been chosen
for this repository yet; until one is added, all rights are reserved by the authors.
