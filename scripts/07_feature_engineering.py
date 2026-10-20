import pandas as pd
import os
from scai_criteria import CANDIDATE_PREDICTORS, VASOACTIVE_CLASSES, MCS_CLASSES

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

print("Loading all datasets...")
cohort = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"),
                     parse_dates=["admittime","dischtime","deathtime",
                                  "icu_intime","icu_outtime","baseline_end","window_24h","dod"])
vl     = pd.read_csv(os.path.join(OUTDIR, "02_baseline_vitals_labs.csv"))
trt    = pd.read_csv(os.path.join(OUTDIR, "03_baseline_treatments.csv"))
demo   = pd.read_csv(os.path.join(OUTDIR, "04_comorbidities_demographics.csv"))
scai24 = pd.read_csv(os.path.join(OUTDIR, "06_scai_24h.csv"))

df = cohort.merge(vl, on=["hadm_id","stay_id"], how="left")
df = df.merge(trt, on=["hadm_id","stay_id"], how="left")

demo_cols = [c for c in demo.columns
             if c not in ["hadm_id","stay_id","subject_id","age","gender"]]
df = df.merge(demo[["hadm_id","stay_id"] + demo_cols],
              on=["hadm_id","stay_id"], how="left")

df = df.merge(scai24[["stay_id","scai_baseline","scai_24h","progressed",
                       "sbp_24h_min","map_24h_min","lactate_24h_max","alt_24h_max","ph_24h_min",
                       "vaso_count_24h","mcd_count_24h"]],
              on="stay_id", how="left")
print(f"Merged: {len(df):,} rows")

df["inhospital_death"] = df["hospital_expire_flag"].astype(int)

essential_mask = (
    (df["sbp"].notna() | df["map"].notna()) &
    df["hr"].notna()         &
    df["lactate"].notna()    &
    df["creatinine"].notna() &
    df["sodium"].notna()     &
    df["rr"].notna()         &
    df["spo2"].notna()
)
df_final = df[essential_mask].copy()
print(f"\nFinal analytic cohort: {len(df_final):,}")
print(f"Excluded: {len(df)-len(df_final):,} admissions (missing essential features)")

print("\nSCAI baseline distribution (final cohort):")
for stage in ["A","B","C","D","E","A (needs evaluation)"]:
    n = (df_final["scai_baseline"] == stage).sum()
    print(f"  Stage {stage}: {n:,}")

print("\nIn-hospital mortality by SCAI baseline stage:")
for stage in ["A","B","C","D","E"]:
    sub = df_final[df_final["scai_baseline"] == stage]
    if len(sub) > 0:
        mort = sub["inhospital_death"].mean() * 100
        n = len(sub)
        print(f"  Stage {stage}: {mort:.1f}%  (n={n:,})")

df_ab = df_final[df_final["scai_baseline"].isin(["A","B"])].copy()
df_ab["label"] = df_ab["progressed"]

print(f"\nA/B cohort: {len(df_ab):,}")
if len(df_ab) > 0:
    prog_n   = df_ab["progressed"].sum()
    prog_pct = df_ab["progressed"].mean() * 100
    print(f"Progressed to C/D/E within 24h: {prog_n:,} / {len(df_ab):,} = {prog_pct:.1f}%")

non_ab = df_final[~df_final["scai_baseline"].isin(["A","B","A (needs evaluation)"])]
print(f"\nC/D/E cohort: {len(non_ab):,}")
needs_eval = df_final[df_final["scai_baseline"]=="A (needs evaluation)"]
print(f"A (needs evaluation): {len(needs_eval):,}")

FEATURE_COLS = [
    "hadm_id", "stay_id", "subject_id",
    "scai_baseline", "scai_24h", "progressed", "inhospital_death", "label",
    "age", "gender_male", "weight_kg", "height_cm", "bmi",
    "ami_flag", "ahf_flag",
    "hr", "sbp", "map", "rr", "temp_f", "spo2",
    "gcs_total", "ambulatory_aid",
    "ph", "pao2", "paco2", "lactate", "base_excess",
    "wbc", "plt", "hgb", "tbil", "albumin", "sodium", "potassium",
    "creatinine", "calcium", "ast", "alt", "bun", "pt", "aptt", "inr",
    "htn", "dm", "hyperlipidemia", "copd", "pneumonia", "ckd", "af",
] + [f"{v}_baseline" for v in VASOACTIVE_CLASSES] + [
    "vaso_count_baseline", "mcd_count_baseline",
] + [f"{v}_baseline" for v in MCS_CLASSES] + [
    "acei_arb_baseline", "betablocker_baseline",
    "furosemide_baseline", "spironolactone_baseline",
    "fluid_input_6h", "pci_flag",
    "sbp_24h_min", "map_24h_min", "lactate_24h_max", "alt_24h_max", "ph_24h_min",
    "vaso_count_24h", "mcd_count_24h",
] + [c for c in vl.columns if c.endswith("_h")]

existing = [c for c in FEATURE_COLS if c in df_final.columns]

df_final[existing].to_csv(os.path.join(OUTDIR, "07_features_full.csv"), index=False)
df_ab[existing].to_csv(os.path.join(OUTDIR, "07_features_AB.csv"), index=False)
missing = (df_ab[CANDIDATE_PREDICTORS].isna().mean() * 100).round(1).rename("missing_pct")
missing.to_csv(os.path.join(OUTDIR, "07_missing_AB.csv"), index_label="variable")

print(f"\n✅ Saved: output/07_features_full.csv  ({len(df_final):,} rows, {len(existing)} cols)")
print(f"✅ Saved: output/07_features_AB.csv    ({len(df_ab):,} rows)")
print("\nFeature columns:", existing)
