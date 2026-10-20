import pandas as pd
import os

DATADIR_HOSP = "/Users/fukushikei/Documents/mimic-iv-3.1/hosp"
DATADIR_ICU  = "/Users/fukushikei/Documents/mimic-iv-3.1/icu"
OUTDIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTDIR, exist_ok=True)

AMI_PREFIXES_ICD9  = ["410"]
AMI_PREFIXES_ICD10 = ["I21", "I22"]

AHF_PREFIXES_ICD9  = ["428"]
AHF_PREFIXES_ICD10 = ["I50"]

MAX_SEQ_NUM = 5


def starts_with_any(code, prefixes):
    return any(str(code).startswith(p) for p in prefixes)


print("=" * 60)
print("STEP 1: Load admissions and patients")
print("=" * 60)

adm = pd.read_csv(
    os.path.join(DATADIR_HOSP, "admissions.csv"),
    usecols=["subject_id", "hadm_id", "admittime", "dischtime",
             "deathtime", "hospital_expire_flag"],
    parse_dates=["admittime", "dischtime", "deathtime"]
)
pat = pd.read_csv(
    os.path.join(DATADIR_HOSP, "patients.csv"),
    usecols=["subject_id", "gender", "anchor_age", "anchor_year",
             "anchor_year_group", "dod"],
    parse_dates=["dod"]
)
print(f"  admissions: {len(adm):,} rows")
print(f"  patients  : {len(pat):,} rows")

adm = adm.merge(pat, on="subject_id", how="left")

adm["admit_year"] = adm["admittime"].dt.year
adm["age"] = adm["anchor_age"] + (adm["admit_year"] - adm["anchor_year"])

adm_period = adm[adm["anchor_year_group"] != "2020 - 2022"].copy()
print(f"\n  After 2008-2019 filter: {len(adm_period):,} admissions")

adm_adult = adm_period[adm_period["age"] >= 18].copy()
print(f"  After age>=18 filter: {len(adm_adult):,} admissions")


print("\n" + "=" * 60)
print("STEP 2: Load diagnoses (top-5 ICD codes per admission)")
print("=" * 60)

diag = pd.read_csv(
    os.path.join(DATADIR_HOSP, "diagnoses_icd.csv"),
    usecols=["subject_id", "hadm_id", "icd_code", "icd_version", "seq_num"]
)
diag_top5 = diag[diag["seq_num"] <= MAX_SEQ_NUM].copy()
print(f"  diagnoses (seq_num≤{MAX_SEQ_NUM}): {len(diag_top5):,} rows")

ami_hadm = set(diag_top5[
    ((diag_top5["icd_version"] == 9) &
     diag_top5["icd_code"].apply(lambda x: starts_with_any(x, AMI_PREFIXES_ICD9))) |
    ((diag_top5["icd_version"] == 10) &
     diag_top5["icd_code"].apply(lambda x: starts_with_any(x, AMI_PREFIXES_ICD10)))
]["hadm_id"])

ahf_hadm = set(diag_top5[
    ((diag_top5["icd_version"] == 9) &
     diag_top5["icd_code"].apply(lambda x: starts_with_any(x, AHF_PREFIXES_ICD9))) |
    ((diag_top5["icd_version"] == 10) &
     diag_top5["icd_code"].apply(lambda x: starts_with_any(x, AHF_PREFIXES_ICD10)))
]["hadm_id"])

ami_ahf_hadm = ami_hadm | ahf_hadm
print(f"  AMI hadm_id: {len(ami_hadm):,}")
print(f"  AHF hadm_id: {len(ahf_hadm):,}")
print(f"  AMI or AHF : {len(ami_ahf_hadm):,}")


print("\n" + "=" * 60)
print("STEP 3: Load ICU stays - first stay per hadm_id")
print("=" * 60)

icu = pd.read_csv(
    os.path.join(DATADIR_ICU, "icustays.csv"),
    usecols=["subject_id", "hadm_id", "stay_id",
             "first_careunit", "intime", "outtime", "los"],
    parse_dates=["intime", "outtime"]
)
print(f"  icustays: {len(icu):,} rows")

icu_first = (
    icu.sort_values("intime")
       .groupby("hadm_id", as_index=False)
       .first()
       .rename(columns={"subject_id": "subject_id_icu"})
)
icu_first = icu_first.drop(columns=["subject_id_icu"])
print(f"  First ICU stays per hadm_id: {len(icu_first):,}")


print("\n" + "=" * 60)
print("STEP 4: Build analytic cohort")
print("=" * 60)

adm_ami_ahf = adm_adult[adm_adult["hadm_id"].isin(ami_ahf_hadm)].copy()
print(f"  Adult AMI/AHF admissions (2008-2019): {len(adm_ami_ahf):,}")

cohort = adm_ami_ahf.merge(
    icu_first[["hadm_id", "stay_id", "first_careunit", "intime", "outtime", "los"]],
    on="hadm_id",
    how="inner"
)
print(f"  With first ICU stay: {len(cohort):,}")

cohort["ami_flag"] = cohort["hadm_id"].isin(ami_hadm).astype(int)
cohort["ahf_flag"] = cohort["hadm_id"].isin(ahf_hadm).astype(int)

cohort["hospital_expire_flag"] = cohort["hospital_expire_flag"].astype(int)

cohort["icu_intime"]  = cohort["intime"]
cohort["icu_outtime"] = cohort["outtime"]
cohort["baseline_end"] = cohort["icu_intime"] + pd.Timedelta(hours=6)
cohort["window_24h"]   = cohort["icu_intime"] + pd.Timedelta(hours=24)

print(f"\n  AMI only : {((cohort['ami_flag']==1) & (cohort['ahf_flag']==0)).sum():,}")
print(f"  AHF only : {((cohort['ahf_flag']==1) & (cohort['ami_flag']==0)).sum():,}")
print(f"  Both     : {((cohort['ami_flag']==1) & (cohort['ahf_flag']==1)).sum():,}")
print(f"  Median age: {cohort['age'].median():.1f}")
print(f"  Male     : {(cohort['gender']=='M').mean()*100:.1f}%")
print(f"  In-hospital mortality: {cohort['hospital_expire_flag'].mean()*100:.1f}%")

out_cols = [
    "subject_id", "hadm_id", "stay_id",
    "admittime", "dischtime", "deathtime", "dod",
    "gender", "age", "anchor_year_group",
    "first_careunit", "icu_intime", "icu_outtime", "los",
    "baseline_end", "window_24h",
    "hospital_expire_flag", "ami_flag", "ahf_flag"
]
cohort = cohort[out_cols]
out_path = os.path.join(OUTDIR, "01_cohort.csv")
cohort.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")
print(f"   Shape : {cohort.shape}")
