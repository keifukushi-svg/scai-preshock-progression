import pandas as pd
import os
from scai_criteria import VASOACTIVE_ITEMIDS, MCS_ITEMIDS, classify_scai_cswg

DATADIR_ICU = "/Users/fukushikei/Documents/mimic-iv-3.1/icu"
OUTDIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

cohort = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"), parse_dates=["icu_intime", "window_24h"])
stay_ids = set(cohort["stay_id"])
print(f"Cohort: {len(cohort):,} admissions")

chart = pd.read_csv(os.path.join(OUTDIR, "02_chartevents_0_24h.csv.gz"))
lab = pd.read_csv(os.path.join(OUTDIR, "02_labevents_0_24h.csv.gz"))

result = cohort[["hadm_id", "stay_id"]].set_index("stay_id")
bp_min = chart[chart["var"].isin(["sbp", "map"])].groupby(["stay_id", "var"])["valuenum"].min().unstack("var")
result["sbp_24h_min"] = bp_min.get("sbp")
result["map_24h_min"] = bp_min.get("map")
lab = pd.concat([
    lab.loc[lab["var"].isin(["lactate", "alt", "ph"]), ["stay_id", "var", "valuenum"]],
    chart.loc[chart["var"].isin(["lactate_chart", "ph_chart"]), ["stay_id", "var", "valuenum"]]
    .assign(var=lambda d: d["var"].str.replace("_chart", "", regex=False)),
], ignore_index=True)
lab_agg = lab.groupby(["stay_id", "var"])["valuenum"]
lab_max, lab_min = lab_agg.max().unstack("var"), lab_agg.min().unstack("var")
result["lactate_24h_max"] = lab_max.get("lactate")
result["alt_24h_max"] = lab_max.get("alt")
result["ph_24h_min"] = lab_min.get("ph")

bl = pd.read_csv(os.path.join(OUTDIR, "02_baseline_vitals_labs.csv")).set_index("stay_id")
result["sbp_24h_min"] = pd.concat([result["sbp_24h_min"], bl["sbp"]], axis=1).min(axis=1)
result["map_24h_min"] = pd.concat([result["map_24h_min"], bl["map"]], axis=1).min(axis=1)
result["lactate_24h_max"] = pd.concat([result["lactate_24h_max"], bl["lactate"]], axis=1).max(axis=1)
result["alt_24h_max"] = pd.concat([result["alt_24h_max"], bl["alt"]], axis=1).max(axis=1)
result["ph_24h_min"] = pd.concat([result["ph_24h_min"], bl["ph"]], axis=1).min(axis=1)

print("\n[inputevents] vasoactive drugs started within 0-24 h...")
inp = pd.read_csv(os.path.join(DATADIR_ICU, "inputevents.csv"),
                  usecols=["stay_id", "starttime", "itemid"], parse_dates=["starttime"])
inp = inp[inp["stay_id"].isin(stay_ids) & inp["itemid"].isin(VASOACTIVE_ITEMIDS)]
inp = inp.merge(cohort[["stay_id", "window_24h"]], on="stay_id")
inp = inp[inp["starttime"] <= inp["window_24h"]]
result["vaso_count_24h"] = inp.assign(drug=inp["itemid"].map(VASOACTIVE_ITEMIDS)).groupby("stay_id")["drug"].nunique()

print("[procedureevents] MCS started within 0-24 h...")
proc = pd.read_csv(os.path.join(DATADIR_ICU, "procedureevents.csv"),
                   usecols=["stay_id", "starttime", "itemid"], parse_dates=["starttime"])
proc = proc[proc["stay_id"].isin(stay_ids) & proc["itemid"].isin(MCS_ITEMIDS)]
proc = proc.merge(cohort[["stay_id", "window_24h"]], on="stay_id")
proc = proc[proc["starttime"] <= proc["window_24h"]]
result["mcd_count_24h"] = proc.assign(device=proc["itemid"].map(MCS_ITEMIDS)).groupby("stay_id")["device"].nunique()
result[["vaso_count_24h", "mcd_count_24h"]] = result[["vaso_count_24h", "mcd_count_24h"]].fillna(0).astype(int)

print("\n[SCAI 24h] Classifying with the worst values within 0-24 h...")
result["scai_24h"] = [classify_scai_cswg(r.sbp_24h_min, r.map_24h_min, r.lactate_24h_max, r.alt_24h_max,
                                         r.ph_24h_min, r.mcd_count_24h, r.vaso_count_24h)
                      for r in result.itertuples()]
result = result.reset_index()

scai_bl = pd.read_csv(os.path.join(OUTDIR, "05_scai_baseline.csv"))
result = result.merge(scai_bl[["stay_id", "scai_baseline"]], on="stay_id", how="left")
result["progressed"] = (result["scai_baseline"].isin(["A", "B"]) &
                        result["scai_24h"].isin(["C", "D", "E"])).astype(int)

print("\n24h SCAI distribution:")
print(result["scai_24h"].value_counts().sort_index())
ab = result[result["scai_baseline"].isin(["A", "B"])]
print(f"\nA/B at baseline: {len(ab):,}; progressed to C/D/E within 24 h: {ab['progressed'].sum():,} "
      f"({ab['progressed'].mean() * 100:.1f}%)")

out_path = os.path.join(OUTDIR, "06_scai_24h.csv")
result.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")
