import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

SCAI_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR   = os.path.join(SCAI_DIR, "output")
DATADIR_ICU  = "/Users/fukushikei/Documents/mimic-iv-3.1/icu"
DATADIR_HOSP = "/Users/fukushikei/Documents/mimic-iv-3.1/hosp"
CHUNKSIZE = 2_000_000

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FEATURES_8 = ["lactate", "gcs_total", "pao2", "wbc", "ast", "weight_kg", "inr", "base_excess"]
TARGET = "progressed"

feat   = pd.read_csv(os.path.join(OUTDIR, "07_features_AB.csv"))
cohort = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"),
                      parse_dates=["icu_intime", "baseline_end"])
timing = pd.read_csv(os.path.join(OUTDIR, "08_progression_timing.csv"))

cohort_join = cohort[["stay_id", "icu_intime", "baseline_end"]]
df = feat.merge(cohort_join, on="stay_id", how="left").merge(
    timing[["stay_id", "h_to_first_CDE"]], on="stay_id", how="left")
print(f"A/B cohort: {len(df):,} rows; progressed with known CDE time: "
      f"{df['h_to_first_CDE'].notna().sum():,}")

stay_ids = set(df["stay_id"])
hadm_ids = set(df["hadm_id"])

CHART_ITEMS = {
    "gcs_eye": 220739, "gcs_verbal": 223900, "gcs_motor": 223901,
}
LAB_ITEMS = {
    50813: "lactate", 51301: "wbc", 50878: "ast", 51237: "inr",
    50821: "pao2", 50802: "base_excess",
}

print("\n[chartevents] reading charttime of GCS components...")
chart_records = []
chart_itemids = set(CHART_ITEMS.values())
for chunk in pd.read_csv(os.path.join(DATADIR_ICU, "chartevents.csv"),
                          usecols=["stay_id", "charttime", "itemid", "valuenum"],
                          parse_dates=["charttime"], chunksize=CHUNKSIZE):
    chunk = chunk[chunk["stay_id"].isin(stay_ids) & chunk["itemid"].isin(chart_itemids)]
    if len(chunk): chart_records.append(chunk)
chart_df = pd.concat(chart_records, ignore_index=True) if chart_records else pd.DataFrame()
print(f"  {len(chart_df):,} rows")

print("[labevents] reading charttime of 6 laboratory variables...")
lab_records = []
lab_itemids = set(LAB_ITEMS.keys())
for chunk in pd.read_csv(os.path.join(DATADIR_HOSP, "labevents.csv"),
                          usecols=["hadm_id", "charttime", "itemid", "valuenum"],
                          parse_dates=["charttime"], chunksize=CHUNKSIZE):
    chunk = chunk[chunk["hadm_id"].isin(hadm_ids) & chunk["itemid"].isin(lab_itemids)]
    if len(chunk): lab_records.append(chunk)
lab_df = pd.concat(lab_records, ignore_index=True) if lab_records else pd.DataFrame()
print(f"  {len(lab_df):,} rows")

df_idx = df.set_index("stay_id")

def first_value_hour(sub_df, time_col):
    if len(sub_df) == 0:
        return np.nan
    sub_df = sub_df.sort_values(time_col)
    return sub_df.iloc[0]["h"]

chart_df = chart_df.merge(cohort[["stay_id","icu_intime","baseline_end"]], on="stay_id", how="left")
chart_df = chart_df[(chart_df["charttime"] >= chart_df["icu_intime"]) &
                     (chart_df["charttime"] <= chart_df["baseline_end"]) &
                     chart_df["valuenum"].notna()]
chart_df["h"] = (chart_df["charttime"] - chart_df["icu_intime"]).dt.total_seconds() / 3600
item_to_gcsvar = {v: k for k, v in CHART_ITEMS.items()}
chart_df["var"] = chart_df["itemid"].map(item_to_gcsvar)

gcs_first_h = (chart_df.sort_values("h")
               .groupby(["stay_id", "var"])["h"].first()
               .unstack("var").max(axis=1))
gcs_first_h.name = "gcs_total_h"

lab_df = lab_df.merge(cohort[["hadm_id","stay_id","icu_intime","baseline_end"]], on="hadm_id", how="left")
lab_df = lab_df[(lab_df["charttime"] >= lab_df["icu_intime"]) &
                (lab_df["charttime"] <= lab_df["baseline_end"]) &
                lab_df["valuenum"].notna()]
lab_df["h"] = (lab_df["charttime"] - lab_df["icu_intime"]).dt.total_seconds() / 3600
lab_df["var"] = lab_df["itemid"].map(LAB_ITEMS)

lab_first_h = (lab_df.sort_values("h")
               .groupby(["stay_id", "var"])["h"].first()
               .unstack("var"))
lab_first_h.columns = [f"{c}_h" for c in lab_first_h.columns]

timing_h = gcs_first_h.to_frame().join(lab_first_h, how="outer")
df_idx = df_idx.join(timing_h, how="left")

print("\nMeasurement-hour columns available:", [c for c in df_idx.columns if c.endswith("_h")])

timing_record = df_idx[["h_to_first_CDE"] + [c for c in df_idx.columns if c.endswith("_h")]].copy()
timing_record.index.name = "stay_id"
timing_record_path = os.path.join(OUTDIR, "11_measurement_hours_8var.csv")
timing_record.to_csv(timing_record_path)
print(f"✅ Saved per-stay_id measurement-hour record: {timing_record_path}")

var_h_map = {
    "lactate": "lactate_h", "gcs_total": "gcs_total_h", "pao2": "pao2_h",
    "wbc": "wbc_h", "ast": "ast_h", "inr": "inr_h", "base_excess": "base_excess_h",
}
n_masked_total = 0
X = df_idx[FEATURES_8].copy()
mask_records = []
for var, hcol in var_h_map.items():
    if hcol not in df_idx.columns:
        print(f"  WARNING: {hcol} not found, skipping mask for {var}")
        continue
    mask = (df_idx["h_to_first_CDE"].notna()) & (df_idx[hcol] > df_idx["h_to_first_CDE"])
    n_masked = mask.sum()
    n_masked_total += n_masked
    masked_stays = df_idx.index[mask]
    for sid in masked_stays:
        mask_records.append({
            "stay_id": sid,
            "variable": var,
            "h_to_first_CDE": df_idx.loc[sid, "h_to_first_CDE"],
            "measurement_h": df_idx.loc[sid, hcol],
            "original_value": df_idx.loc[sid, var],
        })
    X.loc[mask, var] = np.nan
    print(f"  {var}: masked {n_masked} post-progression values")

print(f"\nTotal values masked (post-progression): {n_masked_total}")

mask_df = pd.DataFrame(mask_records)
mask_path = os.path.join(OUTDIR, "11_masked_values_8var.csv")
mask_df.to_csv(mask_path, index=False)
print(f"✅ Saved masked-value record (which stay_id/variable were post-progression): {mask_path}")

y = df_idx[TARGET].values

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_lr_leak = np.zeros(len(y))

for tr_idx, val_idx in cv.split(X, y):
    X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
    y_tr = y[tr_idx]

    imputer = SimpleImputer(strategy="median")
    X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=FEATURES_8)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=FEATURES_8)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_imp)
    X_val_sc = scaler.transform(X_val_imp)

    m = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                           class_weight="balanced", solver="lbfgs")
    m.fit(X_tr_sc, y_tr)
    oof_lr_leak[val_idx] = m.predict_proba(X_val_sc)[:, 1]

auroc_leak_controlled = roc_auc_score(y, oof_lr_leak)

orig_lr_path = os.path.join(OUTDIR, "10_oof_logisticregression_8var.npy")
if os.path.exists(orig_lr_path):
    oof_lr_orig = np.load(orig_lr_path)
    auroc_orig = roc_auc_score(y, oof_lr_orig)
else:
    auroc_orig = np.nan

print("\n" + "="*60)
print("RESULT: Temporal-leakage sensitivity analysis (8-variable model)")
print("="*60)
print(f"Logistic Regression AUROC (primary, 8-var, no masking)      : {auroc_orig:.3f}")
print(f"Logistic Regression AUROC (post-progression values masked)  : {auroc_leak_controlled:.3f}")
print(f"AUROC change                                                  : {auroc_leak_controlled - auroc_orig:+.3f}")

result_df = pd.DataFrame({
    "metric": ["auroc_primary_8var", "auroc_leakage_controlled_8var", "auroc_change", "n_values_masked"],
    "value": [auroc_orig, auroc_leak_controlled, auroc_leak_controlled - auroc_orig, n_masked_total],
})
out_path = os.path.join(OUTDIR, "11_temporal_leakage_sensitivity_8var.csv")
result_df.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")
