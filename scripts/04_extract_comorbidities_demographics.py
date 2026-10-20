import pandas as pd
import os

DATADIR_HOSP = "/Users/fukushikei/Documents/mimic-iv-3.1/hosp"
OUTDIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

cohort = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"),
                     parse_dates=["icu_intime"])
hadm_ids    = set(cohort["hadm_id"])
subject_ids = set(cohort["subject_id"])
print(f"Cohort: {len(cohort):,} admissions")

result = cohort[["subject_id","hadm_id","stay_id","age","gender"]].copy()

print("\n[diagnoses_icd] Comorbidities...")

diag = pd.read_csv(
    os.path.join(DATADIR_HOSP, "diagnoses_icd.csv"),
    usecols=["hadm_id","icd_code","icd_version"]
)
diag = diag[diag["hadm_id"].isin(hadm_ids)].copy()

def flag_icd(diag_df, icd9_prefixes=None, icd10_prefixes=None):
    mask = pd.Series(False, index=diag_df.index)
    if icd9_prefixes:
        mask9 = (diag_df["icd_version"]==9) & diag_df["icd_code"].apply(
            lambda x: any(str(x).startswith(p) for p in icd9_prefixes))
        mask = mask | mask9
    if icd10_prefixes:
        mask10 = (diag_df["icd_version"]==10) & diag_df["icd_code"].apply(
            lambda x: any(str(x).startswith(p) for p in icd10_prefixes))
        mask = mask | mask10
    return set(diag_df[mask]["hadm_id"])

COMORBIDITIES = {
    "htn"           : dict(icd9_prefixes=["401","402","403","404","405"],
                           icd10_prefixes=["I10","I11","I12","I13","I15"]),
    "dm"            : dict(icd9_prefixes=["250"],
                           icd10_prefixes=["E10","E11","E12","E13","E14"]),
    "hyperlipidemia": dict(icd9_prefixes=["272"],
                           icd10_prefixes=["E78"]),
    "copd"          : dict(icd9_prefixes=["491","492","496"],
                           icd10_prefixes=["J44","J43","J41","J42"]),
    "pneumonia"     : dict(icd9_prefixes=["480","481","482","483","484","485","486"],
                           icd10_prefixes=["J12","J13","J14","J15","J16","J17","J18"]),
    "ckd"           : dict(icd9_prefixes=["585","586"],
                           icd10_prefixes=["N18","N19"]),
    "af"            : dict(icd9_prefixes=["42731"],
                           icd10_prefixes=["I48"]),
}

for name, kwargs in COMORBIDITIES.items():
    hadm_set = flag_icd(diag, **kwargs)
    result[name] = result["hadm_id"].isin(hadm_set).astype(int)
    print(f"  {name:20s}: {result[name].mean()*100:.1f}%")


print("\n[omr] Height, weight, BMI...")

omr = pd.read_csv(
    os.path.join(DATADIR_HOSP, "omr.csv"),
    usecols=["subject_id","chartdate","result_name","result_value"],
    parse_dates=["chartdate"]
)
omr = omr[omr["subject_id"].isin(subject_ids)].copy()

omr = omr.merge(cohort[["subject_id","icu_intime"]], on="subject_id", how="left")
omr["result_value_num"] = pd.to_numeric(omr["result_value"], errors="coerce")

wt_lbs = omr[
    (omr["result_name"].str.contains("Weight", case=False, na=False)) &
    (omr["chartdate"] <= omr["icu_intime"].dt.date.apply(pd.Timestamp)) &
    omr["result_value_num"].notna()
].copy()
wt_lbs["weight_kg"] = wt_lbs["result_value_num"] * 0.453592
wt_latest = (wt_lbs.sort_values("chartdate", ascending=False)
             .groupby("subject_id")["weight_kg"].first().reset_index())
result = result.merge(wt_latest, on="subject_id", how="left")

ht_in = omr[
    (omr["result_name"].str.contains("Height", case=False, na=False)) &
    (omr["chartdate"] <= omr["icu_intime"].dt.date.apply(pd.Timestamp)) &
    omr["result_value_num"].notna()
].copy()
ht_in["height_cm"] = ht_in["result_value_num"] * 2.54
ht_latest = (ht_in.sort_values("chartdate", ascending=False)
             .groupby("subject_id")["height_cm"].first().reset_index())
result = result.merge(ht_latest, on="subject_id", how="left")

bmi_val = omr[
    (omr["result_name"].str.contains("BMI", case=False, na=False)) &
    (omr["chartdate"] <= omr["icu_intime"].dt.date.apply(pd.Timestamp)) &
    omr["result_value_num"].notna()
].copy()
bmi_latest = (bmi_val.sort_values("chartdate", ascending=False)
              .groupby("subject_id")["result_value_num"].first().reset_index()
              .rename(columns={"result_value_num": "bmi"}))
result = result.merge(bmi_latest, on="subject_id", how="left")

mask = result["bmi"].isna() & result["weight_kg"].notna() & result["height_cm"].notna()
result.loc[mask, "bmi"] = (result.loc[mask, "weight_kg"] /
                            (result.loc[mask, "height_cm"]/100)**2)

print(f"  weight_kg : {result['weight_kg'].notna().mean()*100:.1f}% non-missing")
print(f"  height_cm : {result['height_cm'].notna().mean()*100:.1f}% non-missing")
print(f"  bmi       : {result['bmi'].notna().mean()*100:.1f}% non-missing")


result["gender_male"] = (result["gender"] == "M").astype(int)

out_path = os.path.join(OUTDIR, "04_comorbidities_demographics.csv")
result.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")
print(f"   Shape : {result.shape}")
