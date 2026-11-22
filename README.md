# Early progression from SCAI stage A/B to C/D/E in ICU patients — analysis code

Analysis code for the manuscript "Early Progression From SCAI Stage A/B to C/D/E in ICU Patients and its Prediction Using Machine Learning: A MIMIC-IV Study" (Fukushi K, et al.). The repository contains code only; it does not contain MIMIC-IV data, derived patient-level files, or trained models.

## Data

MIMIC-IV version 3.1 (https://doi.org/10.13026/kpb9-mt58). Access requires PhysioNet credentialing, the required training, and the PhysioNet Credentialed Health Data Use Agreement 1.5.0. Set `DATADIR_HOSP` and `DATADIR_ICU` at the top of the scripts to the local `hosp` and `icu` directories. Run the scripts from `scripts/`; outputs are written to `scripts/output/`.

## Environment

Python 3.12.2 with the package versions in `requirements.txt`. Random seeds are fixed (42; 43 for the bootstrap of the 2017–2019 cohort). `catboost_sklearn_patch.py` makes CatBoost 1.2.x work with `CalibratedClassifierCV` in scikit-learn 1.6 or later.

## Run order

| Step | Script | Purpose |
|---|---|---|
| 1 | `01_cohort_selection.py` | Adult AMI/AHF admissions with a first ICU stay outside the cardiac vascular ICU (2008–2019) |
| 2 | `02_extract_baseline_vitals_labs.py` | Vital signs and laboratory values, 0–24 h after ICU admission (baseline: first value within 6 h) |
| 3 | `03_extract_baseline_treatments.py` | Vasoactive drugs, mechanical circulatory support, medications, PCI |
| 4 | `04_extract_comorbidities_demographics.py` | Demographics, comorbidities, anthropometrics |
| 5 | `05_scai_staging.py` | CSWG-SCAI stage at baseline |
| 6 | `06_analytic_cohort.py` | Analytic cohort with the essential baseline measurements, and missing proportions of baseline variables in the stage A/B cohort |
| 7 | `07_landmark_dataset.py` | Prediction cohort at 6 hours (stage A/B, without stage C/D/E criteria, ICU discharge, or death within 6 h), 46 candidate predictors available at 6 h (the most recent value of each time-varying variable within 0–6 h, with demographics, comorbidities, and baseline medications), and progression between 6 and 24 h |
| 8 | `08_progression_24h.py` | Progression within 24 h and between 6 and 24 h, in-hospital mortality, the criteria met among progressions, and the 6–24 h staging measurements used in step 12 |
| 9 | `09_feature_selection.py` | Boruta and variance inflation factor screening in the 2008–2016 development cohort |
| 10 | `10_validation.py` | Temporal validation (2008–2016 → 2017–2019) and patient-grouped nested cross-validation with Optuna tuning, isotonic calibration, and thresholds selected in development data; bootstrap confidence intervals, calibration, and decision curves |
| 11 | `11_shap.py` | SHAP values for the CatBoost model developed on 2008–2016 |
| 12 | `12_sensitivity_analyses.py` | Physiology-based outcome, treatment-only progressions, and lactate, ALT, and pH measurement during follow-up |

`scai_criteria.py` holds the shared definitions (vasoactive drugs, mechanical circulatory support devices, plausible ranges, baseline variables, and the CSWG-SCAI staging function); `model_utils.py` holds the shared model fitting functions.

Tables 1 and 2 used the Mann–Whitney U test for continuous variables and the χ² test (Table 1) or Fisher exact test (Table 2) for categorical variables (SciPy), with the files written in steps 6 and 7.

## License

MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the article (details will be added on publication) and the MIMIC-IV dataset:
Johnson A, Bulgarelli L, Pollard T, Gow B, Moody B, Horng S, Celi LA, Mark R. MIMIC-IV (version 3.1). PhysioNet. 2024. https://doi.org/10.13026/kpb9-mt58.
