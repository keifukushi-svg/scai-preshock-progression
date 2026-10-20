# Early progression from SCAI stage A/B to C/D/E in ICU patients — analysis code

Analysis code for the manuscript "Early Progression From SCAI Stage A/B to C/D/E in ICU Patients and its Prediction Using Machine Learning: A MIMIC-IV Study" (Fukushi K, et al.). The repository contains code only; it does not contain MIMIC-IV data, derived patient-level files, or trained models.

## Data

MIMIC-IV version 3.1 (https://doi.org/10.13026/kpb9-mt58). Access requires PhysioNet credentialing, the required training, and the PhysioNet Credentialed Health Data Use Agreement 1.5.0. Set `DATADIR_HOSP` and `DATADIR_ICU` at the top of the scripts to the local `hosp` and `icu` directories. Run the scripts from `scripts/`; outputs are written to `scripts/output/`.

## Environment

Python 3.12.2 with the package versions in `requirements.txt`. Random seeds are fixed at 42. `catboost_sklearn_patch.py` makes CatBoost 1.2.x work with `CalibratedClassifierCV` in scikit-learn 1.6 or later.

## Run order

| Step | Script | Purpose |
|---|---|---|
| 1 | `01_cohort_selection.py` | Adult AMI/AHF admissions with a first ICU stay (2008–2019) |
| 2 | `02_extract_baseline_vitals_labs.py` | Vital signs and laboratory values, 0–6 h after ICU admission |
| 3 | `03_extract_baseline_treatments.py` | Vasoactive drugs, mechanical circulatory support, medications, PCI |
| 4 | `04_extract_comorbidities_demographics.py` | Demographics, comorbidities, anthropometrics |
| 5 | `05_scai_staging.py` | CSWG-SCAI stage at baseline |
| 6 | `06_extract_24h_variables.py` | SCAI stage at 24 h and the progression outcome |
| 7 | `07_feature_engineering.py` | Analytic cohort and A/B cohort |
| 8 | `08_progression_timing.py` | Time of first attainment of stage C/D/E (used in step 11) |
| 9 | `09_feature_selection_derivation_only.py` | Boruta and variance inflation factor screening in the 2008–2016 derivation cohort |
| 10 | `10_final_8var_pipeline.py` | Stratified 5-fold cross-validation with Optuna tuning and isotonic calibration; SHAP |
| 11 | `11_temporal_leakage_sensitivity_8var.py` | Sensitivity analysis masking post-progression values |
| 12 | `12_calibration_decision_curve_8var.py` | Calibration (ten equal-width bins) and decision curve analysis |
| 13 | `13_temporal_validation_full_metrics_8var.py` | Temporal validation (2008–2016 → 2017–2019) |
| 14 | `14_temporal_validation_primary_threshold_8var.py` | Temporal validation metrics at the primary-analysis thresholds |
| 15 | `15_bootstrap_ci_auroc.py` | Bootstrap 95% confidence intervals and paired AUROC differences |

Tables 1 and 2 used the Mann–Whitney U test for continuous variables and the χ² test (Table 1) or Fisher exact test (Table 2) for categorical variables (SciPy), with the files written in step 7.

## License

MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the article (details will be added on publication) and the MIMIC-IV dataset:
Johnson A, Bulgarelli L, Pollard T, Gow B, Moody B, Horng S, Celi LA, Mark R. MIMIC-IV (version 3.1). PhysioNet. 2024. https://doi.org/10.13026/kpb9-mt58.
