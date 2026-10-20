import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

SCAI_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR   = os.path.join(SCAI_DIR, "output")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

feat_df = pd.read_csv(os.path.join(OUTDIR, "07_features_AB.csv"))
cohort  = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"),
                      usecols=["stay_id", "anchor_year_group"])
df = feat_df.merge(cohort, on="stay_id", how="left")

train_groups = ["2008 - 2010", "2011 - 2013", "2014 - 2016"]
df_train = df[df["anchor_year_group"].isin(train_groups)].copy()

print(f"Derivation cohort only (2008-2016): n={len(df_train):,}")
print(f"Progression rate: {df_train['progressed'].mean()*100:.1f}%")

PREDICTOR_COLS = [
    "age", "gender_male", "weight_kg", "height_cm", "bmi",
    "hr", "sbp", "map", "rr", "temp_f", "spo2",
    "gcs_total", "ambulatory_aid",
    "ph", "pao2", "paco2", "lactate", "base_excess",
    "wbc", "plt", "hgb", "tbil", "albumin", "sodium", "potassium",
    "creatinine", "calcium", "ast", "alt", "bun", "aptt", "pt", "inr",
    "htn", "dm", "hyperlipidemia", "copd", "pneumonia", "ckd", "af",
    "ami_flag", "ahf_flag",
    "acei_arb_baseline", "betablocker_baseline",
    "furosemide_baseline", "spironolactone_baseline",
    "pci_flag",
]
TARGET = "progressed"

available = [c for c in PREDICTOR_COLS if c in df_train.columns]
print(f"Available features: {len(available)} / {len(PREDICTOR_COLS)}")

X = df_train[available].copy()
y = df_train[TARGET].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("\n" + "="*50)
print("STEP 1: Boruta Feature Selection (2008-2016 only)")
print("="*50)

from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                             n_jobs=-1, max_depth=5)
boruta = BorutaPy(rf, n_estimators='auto', verbose=0,
                  random_state=RANDOM_STATE, max_iter=50)
boruta.fit(X_imp.values, y)

selected_features  = [available[i] for i, s in enumerate(boruta.support_) if s]
tentative_features  = [available[i] for i, s in enumerate(boruta.support_weak_) if s]
boruta_features = selected_features + tentative_features

print(f"Confirmed: {selected_features}")
print(f"Tentative: {tentative_features}")
print(f"Total Boruta-retained: {len(boruta_features)}")

print("\n" + "="*50)
print("STEP 2: VIF check (remove VIF > 10)")
print("="*50)

from statsmodels.stats.outliers_influence import variance_inflation_factor

X_sel = X_imp[boruta_features].copy()
VIF_THRESHOLD = 10
while True:
    vifs = {col: variance_inflation_factor(X_sel.values, i)
            for i, col in enumerate(X_sel.columns)}
    max_vif = max(vifs.values())
    if max_vif > VIF_THRESHOLD:
        drop_col = max(vifs, key=vifs.get)
        print(f"  Removing {drop_col} (VIF={max_vif:.2f})")
        X_sel = X_sel.drop(columns=[drop_col])
    else:
        break

final_features_derivation_only = list(X_sel.columns)
print(f"\nFinal features (derivation-only, n=2008-2016): {len(final_features_derivation_only)}")
print(final_features_derivation_only)

out_path = os.path.join(OUTDIR, "09_boruta_vif_derivation_only.txt")
with open(out_path, "w") as f:
    f.write("final_features\n")
    for feat in final_features_derivation_only:
        f.write(f"{feat}\n")
print(f"\n✅ Saved: {out_path}")
