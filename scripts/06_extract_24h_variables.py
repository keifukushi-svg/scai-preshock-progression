import pandas as pd
import os

DATADIR_HOSP = "/Users/fukushikei/Documents/mimic-iv-3.1/hosp"
DATADIR_ICU  = "/Users/fukushikei/Documents/mimic-iv-3.1/icu"
OUTDIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CHUNKSIZE    = 2_000_000

cohort = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"),
                     parse_dates=["icu_intime","icu_outtime",
                                  "baseline_end","window_24h"])
stay_ids  = set(cohort["stay_id"])
hadm_ids  = set(cohort["hadm_id"])
print(f"Cohort: {len(cohort):,} admissions")

result = cohort[["hadm_id","stay_id"]].copy()

CHART_24H_MIN = {
    "sbp_24h_min": [220050, 220179],
    "map_24h_min": [220052, 220181],
}
all_chart_24h = set()
chart_24h_var = {}
for var, ids in CHART_24H_MIN.items():
    for iid in ids:
        all_chart_24h.add(iid)
        chart_24h_var[iid] = var

print("\n[chartevents] SBP/MAP in 0-24h window...")
chart_records = []
for chunk in pd.read_csv(
        os.path.join(DATADIR_ICU, "chartevents.csv"),
        usecols=["stay_id","charttime","itemid","valuenum"],
        parse_dates=["charttime"],
        chunksize=CHUNKSIZE):
    chunk = chunk[chunk["stay_id"].isin(stay_ids) &
                  chunk["itemid"].isin(all_chart_24h)]
    if len(chunk) > 0:
        chart_records.append(chunk)

chart_df = pd.concat(chart_records, ignore_index=True) if chart_records else pd.DataFrame()
print(f"  chart rows: {len(chart_df):,}")

if len(chart_df) > 0:
    chart_df["var"] = chart_df["itemid"].map(chart_24h_var)
    chart_df = chart_df.merge(cohort[["stay_id","icu_intime","window_24h"]],
                              on="stay_id", how="left")
    chart_df_24 = chart_df[
        (chart_df["charttime"] >= chart_df["icu_intime"]) &
        (chart_df["charttime"] <= chart_df["window_24h"]) &
        chart_df["valuenum"].notna() &
        (chart_df["valuenum"] > 0)
    ]
    chart_24_min = (chart_df_24.groupby(["stay_id","var"])["valuenum"]
                    .min().unstack("var").reset_index())
    result = result.merge(chart_24_min, on="stay_id", how="left")

LAB_24H = {
    "lactate_24h_max": ([50813], "max"),
    "alt_24h_max"    : ([50861], "max"),
    "ph_24h_min"     : ([50820], "min"),
}
all_lab_24h = set()
lab_24h_meta = {}
for var, (ids, agg) in LAB_24H.items():
    for iid in ids:
        all_lab_24h.add(iid)
        lab_24h_meta[iid] = (var, agg)

print("\n[labevents] Worst values in 0-24h window...")
lab_records = []
for chunk in pd.read_csv(
        os.path.join(DATADIR_HOSP, "labevents.csv"),
        usecols=["hadm_id","charttime","itemid","valuenum"],
        parse_dates=["charttime"],
        chunksize=CHUNKSIZE):
    chunk = chunk[chunk["hadm_id"].isin(hadm_ids) &
                  chunk["itemid"].isin(all_lab_24h)]
    if len(chunk) > 0:
        lab_records.append(chunk)

lab_df = pd.concat(lab_records, ignore_index=True) if lab_records else pd.DataFrame()
print(f"  lab rows: {len(lab_df):,}")

if len(lab_df) > 0:
    lab_df = lab_df.merge(cohort[["hadm_id","icu_intime","window_24h"]],
                           on="hadm_id", how="left")
    lab_df_24 = lab_df[
        (lab_df["charttime"] >= lab_df["icu_intime"]) &
        (lab_df["charttime"] <= lab_df["window_24h"]) &
        lab_df["valuenum"].notna()
    ]
    lab_df_24 = lab_df_24.copy()
    lab_df_24["var"] = lab_df_24["itemid"].map(lambda x: lab_24h_meta[x][0])
    lab_df_24["agg"] = lab_df_24["itemid"].map(lambda x: lab_24h_meta[x][1])

    for var_name, (ids, agg_method) in LAB_24H.items():
        sub = lab_df_24[lab_df_24["var"]==var_name]
        if len(sub) > 0:
            if agg_method == "max":
                agg = sub.groupby("hadm_id")["valuenum"].max().reset_index()
            else:
                agg = sub.groupby("hadm_id")["valuenum"].min().reset_index()
            agg.columns = ["hadm_id", var_name]
            result = result.merge(agg, on="hadm_id", how="left")

bl = pd.read_csv(os.path.join(OUTDIR, "02_baseline_vitals_labs.csv"))
result = result.merge(bl[["stay_id","sbp","map","lactate","alt","ph"]],
                      on="stay_id", how="left")

result["sbp_24h_min"] = result[["sbp_24h_min","sbp"]].min(axis=1)
result["map_24h_min"] = result[["map_24h_min","map"]].min(axis=1)

result["lactate_24h_max"] = result[["lactate_24h_max","lactate"]].max(axis=1)
result["alt_24h_max"]     = result[["alt_24h_max","alt"]].max(axis=1)

result["ph_24h_min"] = result[["ph_24h_min","ph"]].min(axis=1)

VASO_ITEMIDS = {221906:"norepi", 221289:"epi", 229617:"epi",
                222315:"vaso_drug", 221653:"dobuta", 221986:"milrino"}
MCS_ITEMIDS  = {224272:"iabp", 228169:"impella",
                229529:"ecmo", 229530:"ecmo"}

print("\n[inputevents] Vasopressors in 0-24h...")
inp = pd.read_csv(os.path.join(DATADIR_ICU, "inputevents.csv"),
                  usecols=["stay_id","starttime","itemid"],
                  parse_dates=["starttime"])
inp = inp[inp["stay_id"].isin(stay_ids) &
          inp["itemid"].isin(VASO_ITEMIDS.keys())].copy()
inp = inp.merge(cohort[["stay_id","icu_intime","window_24h"]],
                on="stay_id", how="left")
inp_24 = inp[inp["starttime"] <= inp["window_24h"]].copy()
inp_24.loc[:, "var"] = inp_24["itemid"].map(VASO_ITEMIDS)

for var in set(VASO_ITEMIDS.values()):
    stays = set(inp_24[inp_24["var"]==var]["stay_id"])
    result[f"{var}_24h"] = result["stay_id"].isin(stays).astype(int)
vaso_24_cols = [f"{v}_24h" for v in set(VASO_ITEMIDS.values())]
result["vaso_count_24h"] = result[vaso_24_cols].sum(axis=1)

print("\n[procedureevents] MCS in 0-24h...")
proc = pd.read_csv(os.path.join(DATADIR_ICU, "procedureevents.csv"),
                   usecols=["stay_id","starttime","itemid"],
                   parse_dates=["starttime"])
proc = proc[proc["stay_id"].isin(stay_ids) &
            proc["itemid"].isin(MCS_ITEMIDS.keys())].copy()
proc = proc.merge(cohort[["stay_id","window_24h"]], on="stay_id", how="left")
proc_24 = proc[proc["starttime"] <= proc["window_24h"]].copy()
proc_24.loc[:, "var"] = proc_24["itemid"].map(MCS_ITEMIDS)

for var in set(MCS_ITEMIDS.values()):
    stays = set(proc_24[proc_24["var"]==var]["stay_id"])
    result[f"{var}_24h"] = result["stay_id"].isin(stays).astype(int)
mcs_24_cols = [f"{v}_24h" for v in set(MCS_ITEMIDS.values())]
result["mcd_count_24h"] = result[mcs_24_cols].sum(axis=1)

def classify_scai(sbp, map_, lactate, alt, ph, mcd, vaso):
    def v(x): return x if pd.notna(x) else None
    sbp_v=v(sbp); map_v=v(map_); lact_v=v(lactate); alt_v=v(alt)
    ph_v=v(ph); mcd_v=int(mcd) if pd.notna(mcd) else 0
    vaso_v=int(vaso) if pd.notna(vaso) else 0

    if ((sbp_v is not None and sbp_v < 60) or (map_v is not None and map_v < 50) or
        (ph_v  is not None and ph_v  < 7.2) or (lact_v is not None and lact_v > 10) or
        mcd_v >= 3 or vaso_v >= 3):
        return "E"

    hypo = ((sbp_v is not None and 60<=sbp_v<=90) or
            (map_v is not None and 50<=map_v<=65))
    high_lact = (lact_v is not None and lact_v >= 5)
    mod_lact  = (lact_v is not None and 2<=lact_v<5)
    high_alt  = (alt_v is not None and alt_v > 500)
    mod_alt   = (alt_v is not None and 200<=alt_v<=500)

    if (mcd_v>=2 or vaso_v>=2 or (hypo and (high_lact or high_alt)) or
        (hypo and mcd_v in [1,2]) or (hypo and vaso_v in [1,2])):
        return "D"
    if (mcd_v==1 or vaso_v==1 or (hypo and (mod_lact or mod_alt))):
        return "C"
    if hypo:
        return "B"
    if sbp_v is not None or map_v is not None:
        return "A"
    return "A (needs evaluation)"

print("\n[SCAI 24h] Classifying with worst values in 0-24h...")
result["scai_24h"] = result.apply(
    lambda r: classify_scai(
        r.get("sbp_24h_min"), r.get("map_24h_min"),
        r.get("lactate_24h_max"), r.get("alt_24h_max"),
        r.get("ph_24h_min"),
        r.get("mcd_count_24h", 0), r.get("vaso_count_24h", 0)
    ), axis=1
)

scai_bl = pd.read_csv(os.path.join(OUTDIR, "05_scai_baseline.csv"))
result = result.merge(scai_bl[["stay_id","scai_baseline"]], on="stay_id", how="left")

result["progressed"] = (
    result["scai_baseline"].isin(["A","B"]) &
    result["scai_24h"].isin(["C","D","E"])
).astype(int)

print("\n24h SCAI distribution:")
print(result["scai_24h"].value_counts().sort_index())

ab_mask = result["scai_baseline"].isin(["A","B"])
print(f"\nA/B at baseline: {ab_mask.sum():,}")
if ab_mask.sum() > 0:
    prog_n   = result[ab_mask]["progressed"].sum()
    prog_pct = result[ab_mask]["progressed"].mean() * 100
    print(f"Progressed to C/D/E (0-24h): {prog_n:,} / {ab_mask.sum():,} = {prog_pct:.1f}%")

print("\nMissing rates in 0-24h window (A/B subset):")
df_ab = result[ab_mask]
for col in ["sbp_24h_min","map_24h_min","lactate_24h_max","alt_24h_max","ph_24h_min"]:
    if col in df_ab.columns:
        miss = df_ab[col].isna().mean() * 100
        print(f"  {col}: {miss:.1f}%")

out_path = os.path.join(OUTDIR, "06_scai_24h.csv")
result.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")
print(f"   Shape : {result.shape}")
