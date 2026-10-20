import pandas as pd
import os

DATADIR_HOSP = "/Users/fukushikei/Documents/mimic-iv-3.1/hosp"
DATADIR_ICU  = "/Users/fukushikei/Documents/mimic-iv-3.1/icu"
OUTDIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

cohort = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"),
                     parse_dates=["icu_intime", "icu_outtime",
                                  "baseline_end", "window_24h"])
stay_ids = set(cohort["stay_id"])
hadm_ids = set(cohort["hadm_id"])
subject_ids = set(cohort["subject_id"])
print(f"Cohort: {len(cohort):,} admissions")

result = cohort[["hadm_id","stay_id"]].copy()

print("\n[inputevents] Vasopressors & fluids...")

VASO_ITEMIDS = {
    "norepi"   : [221906],
    "epi"      : [221289, 229617],
    "vaso_drug": [222315],
    "dobuta"   : [221653],
    "milrino"  : [221986],
}
all_vaso_itemids = set()
vaso_item_to_var = {}
for var, ids in VASO_ITEMIDS.items():
    for iid in ids:
        all_vaso_itemids.add(iid)
        vaso_item_to_var[iid] = var

inp = pd.read_csv(
    os.path.join(DATADIR_ICU, "inputevents.csv"),
    usecols=["stay_id", "starttime", "endtime", "itemid",
             "amount", "amountuom", "totalamount"],
    parse_dates=["starttime", "endtime"]
)
inp = inp[inp["stay_id"].isin(stay_ids)].copy()
inp = inp.merge(cohort[["stay_id","icu_intime","baseline_end","window_24h"]],
                on="stay_id", how="left")

inp_base = inp[inp["starttime"] <= inp["baseline_end"]].copy()

vaso_base = inp_base[inp_base["itemid"].isin(all_vaso_itemids)].copy()
vaso_base["var"] = vaso_base["itemid"].map(vaso_item_to_var)

for var in VASO_ITEMIDS.keys():
    vaso_stay = set(vaso_base[vaso_base["var"]==var]["stay_id"])
    result[f"{var}_baseline"] = result["stay_id"].isin(vaso_stay).astype(int)

vaso_cols = [f"{v}_baseline" for v in VASO_ITEMIDS.keys()]
result["vaso_count_baseline"] = result[vaso_cols].sum(axis=1)

fluid_base = inp_base.copy()
fluid_base["amount_num"] = pd.to_numeric(fluid_base["amount"], errors="coerce")
fluid_total = fluid_base.groupby("stay_id")["amount_num"].sum().reset_index()
fluid_total.columns = ["stay_id", "fluid_input_6h"]
result = result.merge(fluid_total, on="stay_id", how="left")
result["fluid_input_6h"] = result["fluid_input_6h"].fillna(0)

print(f"  Vasopressors extracted. vaso_count: {result['vaso_count_baseline'].value_counts().sort_index().to_dict()}")


print("\n[procedureevents] MCS devices...")

MCS_ITEMIDS = {
    "iabp"   : [224272],
    "impella": [228169],
    "ecmo"   : [229529, 229530],
}
all_mcs_itemids = set()
mcs_item_to_var = {}
for var, ids in MCS_ITEMIDS.items():
    for iid in ids:
        all_mcs_itemids.add(iid)
        mcs_item_to_var[iid] = var

proc = pd.read_csv(
    os.path.join(DATADIR_ICU, "procedureevents.csv"),
    usecols=["stay_id", "starttime", "itemid"],
    parse_dates=["starttime"]
)
proc = proc[proc["stay_id"].isin(stay_ids)].copy()
proc = proc.merge(cohort[["stay_id","icu_intime","baseline_end"]],
                  on="stay_id", how="left")
proc_base = proc[
    (proc["itemid"].isin(all_mcs_itemids)) &
    (proc["starttime"] <= proc["baseline_end"])
].copy()
proc_base["var"] = proc_base["itemid"].map(mcs_item_to_var)

for var in MCS_ITEMIDS.keys():
    mcs_stay = set(proc_base[proc_base["var"]==var]["stay_id"])
    result[f"{var}_baseline"] = result["stay_id"].isin(mcs_stay).astype(int)

mcs_cols = [f"{v}_baseline" for v in MCS_ITEMIDS.keys()]
result["mcd_count_baseline"] = result[mcs_cols].sum(axis=1)

print(f"  MCS extracted. mcd_count: {result['mcd_count_baseline'].value_counts().sort_index().to_dict()}")


print("\n[prescriptions] Medications...")

presc = pd.read_csv(
    os.path.join(DATADIR_HOSP, "prescriptions.csv"),
    usecols=["hadm_id", "starttime", "drug"],
    parse_dates=["starttime"]
)
presc = presc[presc["hadm_id"].isin(hadm_ids)].copy()
presc = presc.merge(cohort[["hadm_id","icu_intime","baseline_end"]],
                    on="hadm_id", how="left")
presc["drug_lower"] = presc["drug"].str.lower().fillna("")

presc_base = presc[presc["starttime"] <= presc["baseline_end"]].copy()

acei_keywords = ["lisinopril","enalapril","captopril","ramipril",
                 "perindopril","quinapril","benazepril","trandolapril",
                 "fosinopril","moexipril","enalaprilat"]
acei_hadm = set(presc_base[presc_base["drug_lower"].str.contains(
    "|".join(acei_keywords), na=False)]["hadm_id"])
result["acei_baseline"] = result["hadm_id"].isin(acei_hadm).astype(int)

arb_keywords = ["losartan","valsartan","irbesartan","olmesartan",
                "candesartan","telmisartan","azilsartan","eprosartan"]
arb_hadm = set(presc_base[presc_base["drug_lower"].str.contains(
    "|".join(arb_keywords), na=False)]["hadm_id"])
result["arb_baseline"] = result["hadm_id"].isin(arb_hadm).astype(int)

result["acei_arb_baseline"] = ((result["acei_baseline"]==1) |
                                (result["arb_baseline"]==1)).astype(int)

bb_keywords = ["metoprolol","carvedilol","bisoprolol","atenolol",
               "labetalol","propranolol","nebivolol","sotalol",
               "acebutolol","nadolol","pindolol","timolol"]
bb_hadm = set(presc_base[presc_base["drug_lower"].str.contains(
    "|".join(bb_keywords), na=False)]["hadm_id"])
result["betablocker_baseline"] = result["hadm_id"].isin(bb_hadm).astype(int)

furo_hadm = set(presc_base[presc_base["drug_lower"].str.contains(
    "furosemide|lasix", na=False)]["hadm_id"])
result["furosemide_baseline"] = result["hadm_id"].isin(furo_hadm).astype(int)

spiro_hadm = set(presc_base[presc_base["drug_lower"].str.contains(
    "spironolactone|aldactone", na=False)]["hadm_id"])
result["spironolactone_baseline"] = result["hadm_id"].isin(spiro_hadm).astype(int)

print(f"  ACEI/ARB: {result['acei_arb_baseline'].mean()*100:.1f}%")
print(f"  BB      : {result['betablocker_baseline'].mean()*100:.1f}%")
print(f"  Furo    : {result['furosemide_baseline'].mean()*100:.1f}%")
print(f"  Spiro   : {result['spironolactone_baseline'].mean()*100:.1f}%")


print("\n[procedures_icd] PCI during admission...")

proc_icd = pd.read_csv(
    os.path.join(DATADIR_HOSP, "procedures_icd.csv"),
    usecols=["hadm_id", "icd_code", "icd_version"]
)
proc_icd = proc_icd[proc_icd["hadm_id"].isin(hadm_ids)].copy()

pci_icd9  = proc_icd[(proc_icd["icd_version"]==9) &
                      proc_icd["icd_code"].str.startswith(("3601", "3602", "3605", "3606", "0066"), na=False)]
pci_icd10 = proc_icd[(proc_icd["icd_version"]==10) &
                      proc_icd["icd_code"].str.startswith("027", na=False)]
pci_hadm  = set(pci_icd9["hadm_id"]) | set(pci_icd10["hadm_id"])
result["pci_flag"] = result["hadm_id"].isin(pci_hadm).astype(int)
print(f"  PCI during admission: {result['pci_flag'].mean()*100:.1f}%")


out_path = os.path.join(OUTDIR, "03_baseline_treatments.csv")
result.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")
print(f"   Shape : {result.shape}")
print(f"   Columns: {result.columns.tolist()}")
