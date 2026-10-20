import pandas as pd
import numpy as np
import os

DATADIR_ICU  = "/Users/fukushikei/Documents/mimic-iv-3.1/icu"
DATADIR_HOSP = "/Users/fukushikei/Documents/mimic-iv-3.1/hosp"
OUTDIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CHUNKSIZE    = 2_000_000

feat   = pd.read_csv(os.path.join(OUTDIR, "07_features_AB.csv"))
cohort = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"),
                     parse_dates=["icu_intime", "window_24h"])
cohort = cohort[cohort["stay_id"].isin(feat["stay_id"])].copy()
stay_ids = set(cohort["stay_id"])
hadm_ids = set(cohort["hadm_id"])

print(f"A/B cohort: {len(feat):,} (progressed {feat['progressed'].sum():,})")

SBP_ITEMS = [220050, 220179]
MAP_ITEMS = [220052, 220181]
BP_ITEMS  = set(SBP_ITEMS + MAP_ITEMS)

print("\n[chartevents] reading SBP/MAP...")
bp_records = []
for chunk in pd.read_csv(os.path.join(DATADIR_ICU, "chartevents.csv"),
                          usecols=["stay_id","charttime","itemid","valuenum"],
                          parse_dates=["charttime"], chunksize=CHUNKSIZE):
    chunk = chunk[chunk["stay_id"].isin(stay_ids) & chunk["itemid"].isin(BP_ITEMS)]
    if len(chunk): bp_records.append(chunk)

bp_df = pd.concat(bp_records, ignore_index=True)
bp_df["var"] = bp_df["itemid"].apply(lambda x: "sbp" if x in SBP_ITEMS else "map")
bp_df = bp_df.merge(cohort[["stay_id","icu_intime","window_24h"]], on="stay_id")
bp_df = bp_df[(bp_df["charttime"] >= bp_df["icu_intime"]) &
              (bp_df["charttime"] <= bp_df["window_24h"]) &
              bp_df["valuenum"].notna() & (bp_df["valuenum"] > 0)]
bp_df["h"] = (bp_df["charttime"] - bp_df["icu_intime"]).dt.total_seconds() / 3600
bp_wide = bp_df.pivot_table(index=["stay_id","h"], columns="var",
                             values="valuenum", aggfunc="mean").reset_index()
bp_wide.columns.name = None
print(f"  BP records: {len(bp_wide):,} rows")

LAB_ITEMS = {50813: "lactate", 50861: "alt", 50820: "ph"}

print("[labevents] reading lactate/ALT/pH...")
lab_records = []
for chunk in pd.read_csv(os.path.join(DATADIR_HOSP, "labevents.csv"),
                          usecols=["hadm_id","charttime","itemid","valuenum"],
                          parse_dates=["charttime"], chunksize=CHUNKSIZE):
    chunk = chunk[chunk["hadm_id"].isin(hadm_ids) & chunk["itemid"].isin(LAB_ITEMS)]
    if len(chunk): lab_records.append(chunk)

lab_df = pd.concat(lab_records, ignore_index=True)
lab_df["var"] = lab_df["itemid"].map(LAB_ITEMS)
cohort_hadm = cohort[["stay_id","hadm_id","icu_intime","window_24h"]]
lab_df = lab_df.merge(cohort_hadm, on="hadm_id")
lab_df = lab_df[(lab_df["charttime"] >= lab_df["icu_intime"]) &
                (lab_df["charttime"] <= lab_df["window_24h"]) &
                lab_df["valuenum"].notna()]
lab_df["h"] = (lab_df["charttime"] - lab_df["icu_intime"]).dt.total_seconds() / 3600
lab_wide = lab_df.pivot_table(index=["stay_id","h"], columns="var",
                               values="valuenum", aggfunc="mean").reset_index()
lab_wide.columns.name = None
print(f"  Lab records: {len(lab_wide):,} rows")

VASO_ITEMS = {221906:"norepi", 221289:"epi", 229617:"epi",
              222315:"vaso",  221653:"dobu", 221986:"milrinone"}

print("[inputevents] reading vasoactive drugs...")
inp_records = []
for chunk in pd.read_csv(os.path.join(DATADIR_ICU, "inputevents.csv"),
                          usecols=["stay_id","starttime","itemid","amount"],
                          parse_dates=["starttime"], chunksize=CHUNKSIZE):
    chunk = chunk[chunk["stay_id"].isin(stay_ids) & chunk["itemid"].isin(VASO_ITEMS)]
    if len(chunk): inp_records.append(chunk)

inp_df = pd.concat(inp_records, ignore_index=True)
inp_df = inp_df.merge(cohort[["stay_id","icu_intime","window_24h"]], on="stay_id")
inp_df = inp_df[(inp_df["starttime"] >= inp_df["icu_intime"]) &
                (inp_df["starttime"] <= inp_df["window_24h"])]
inp_df["h"] = (inp_df["starttime"] - inp_df["icu_intime"]).dt.total_seconds() / 3600
inp_df["drug"] = inp_df["itemid"].map(VASO_ITEMS)
vaso_ts = (inp_df.groupby(["stay_id","h"])["drug"]
           .nunique().reset_index(name="vaso_count"))
print(f"  Vasoactive drug records: {len(vaso_ts):,} rows")

MCS_ITEMS = {224272:"iabp", 228169:"impella", 229529:"ecmo", 229530:"ecmo"}

print("[procedureevents] reading MCS...")
proc_df = pd.read_csv(os.path.join(DATADIR_ICU, "procedureevents.csv"),
                       usecols=["stay_id","starttime","itemid"],
                       parse_dates=["starttime"])
proc_df = proc_df[proc_df["stay_id"].isin(stay_ids) & proc_df["itemid"].isin(MCS_ITEMS)]
proc_df = proc_df.merge(cohort[["stay_id","icu_intime","window_24h"]], on="stay_id")
proc_df = proc_df[(proc_df["starttime"] >= proc_df["icu_intime"]) &
                  (proc_df["starttime"] <= proc_df["window_24h"])]
proc_df["h"] = (proc_df["starttime"] - proc_df["icu_intime"]).dt.total_seconds() / 3600
proc_df["device"] = proc_df["itemid"].map(MCS_ITEMS)
mcs_ts = proc_df.groupby(["stay_id","h"])["device"].nunique().reset_index(name="mcs_count")
print(f"  MCS records: {len(mcs_ts):,} rows")

def first_cde_time(stay_id):
    times = set()

    bp  = bp_wide[bp_wide["stay_id"] == stay_id]
    lab = lab_wide[lab_wide["stay_id"] == stay_id]
    vaso = vaso_ts[vaso_ts["stay_id"] == stay_id]
    mcs  = mcs_ts[mcs_ts["stay_id"] == stay_id]

    for h in sorted(set(bp["h"]) | set(lab["h"]) | set(vaso["h"]) | set(mcs["h"])):
        sbp_v = bp[bp["h"] <= h]["sbp"].dropna().iloc[-1] if len(bp[bp["h"] <= h]["sbp"].dropna()) else np.nan
        map_v = bp[bp["h"] <= h]["map"].dropna().iloc[-1] if len(bp[bp["h"] <= h]["map"].dropna()) else np.nan
        lac_v = lab[(lab["h"] <= h) & (lab["var"] == "lactate")]["valuenum"].dropna().iloc[-1] if "var" in lab.columns else np.nan
        alt_v = lab[(lab["h"] <= h) & (lab["var"] == "alt")]["valuenum"].dropna().iloc[-1] if "var" in lab.columns else np.nan
        ph_v  = lab[(lab["h"] <= h) & (lab["var"] == "ph")]["valuenum"].dropna().iloc[-1] if "var" in lab.columns else np.nan
        vc    = int(vaso[vaso["h"] <= h]["vaso_count"].max()) if len(vaso[vaso["h"] <= h]) else 0
        mc    = int(mcs[mcs["h"] <= h]["mcs_count"].max()) if len(mcs[mcs["h"] <= h]) else 0

        hypo = ((not np.isnan(sbp_v) and 60 <= sbp_v <= 90) or
                (not np.isnan(map_v) and 50 <= map_v <= 65))

        if ((not np.isnan(sbp_v) and sbp_v < 60) or
            (not np.isnan(map_v) and map_v < 50) or
            (not np.isnan(ph_v)  and ph_v  < 7.2) or
            (not np.isnan(lac_v) and lac_v > 10) or
            vc >= 3 or mc >= 3):
            return h, "E"

        if (mc == 2 or vc == 2 or
            (hypo and not np.isnan(lac_v) and lac_v >= 5) or
            (hypo and not np.isnan(alt_v) and alt_v > 500) or
            (hypo and vc in [1,2]) or (hypo and mc in [1,2])):
            return h, "D"

        if (mc == 1 or vc == 1 or
            (hypo and not np.isnan(lac_v) and 2 <= lac_v < 5) or
            (hypo and not np.isnan(alt_v) and 200 <= alt_v <= 500)):
            return h, "C"

    return None, None

prog_stays = feat[feat["progressed"] == 1]["stay_id"].tolist()
print(f"\nFirst C/D/E time for {len(prog_stays):,} progressed admissions...")

lab_long = lab_df[["stay_id","h","var","valuenum"]].copy()

results = []
for i, sid in enumerate(prog_stays):
    if i % 200 == 0: print(f"  {i}/{len(prog_stays)}...", flush=True)

    bp_s   = bp_wide[bp_wide["stay_id"] == sid].sort_values("h")
    lab_s  = lab_long[lab_long["stay_id"] == sid].sort_values("h")
    vaso_s = vaso_ts[vaso_ts["stay_id"] == sid].sort_values("h")
    mcs_s  = mcs_ts[mcs_ts["stay_id"] == sid].sort_values("h")

    all_times = sorted(set(bp_s["h"]) | set(lab_s["h"]) | set(vaso_s["h"]) | set(mcs_s["h"]))

    h_cde = None
    trigger = None

    for h in all_times:
        sbp_col = bp_s[bp_s["h"] <= h]["sbp"].dropna()
        map_col = bp_s[bp_s["h"] <= h]["map"].dropna()
        lac_col = lab_s[(lab_s["h"] <= h) & (lab_s["var"] == "lactate")]["valuenum"].dropna()
        alt_col = lab_s[(lab_s["h"] <= h) & (lab_s["var"] == "alt")]["valuenum"].dropna()
        ph_col  = lab_s[(lab_s["h"] <= h) & (lab_s["var"] == "ph")]["valuenum"].dropna()
        vc_rows = vaso_s[vaso_s["h"] <= h]
        mc_rows = mcs_s[mcs_s["h"] <= h]

        sbp_v = float(sbp_col.iloc[-1]) if len(sbp_col) else np.nan
        map_v = float(map_col.iloc[-1]) if len(map_col) else np.nan
        lac_v = float(lac_col.iloc[-1]) if len(lac_col) else np.nan
        alt_v = float(alt_col.iloc[-1]) if len(alt_col) else np.nan
        ph_v  = float(ph_col.iloc[-1])  if len(ph_col)  else np.nan
        vc    = int(vc_rows["vaso_count"].max()) if len(vc_rows) else 0
        mc    = int(mc_rows["mcs_count"].max())  if len(mc_rows) else 0

        hypo = ((not np.isnan(sbp_v) and 60 <= sbp_v <= 90) or
                (not np.isnan(map_v) and 50 <= map_v <= 65))

        if ((not np.isnan(sbp_v) and sbp_v < 60) or
            (not np.isnan(map_v) and map_v < 50) or
            (not np.isnan(ph_v)  and ph_v  < 7.2) or
            (not np.isnan(lac_v) and lac_v > 10) or
            vc >= 3 or mc >= 3):
            h_cde, trigger = h, "E"; break

        if (mc == 2 or vc == 2 or
            (hypo and not np.isnan(lac_v) and lac_v >= 5) or
            (hypo and not np.isnan(alt_v) and alt_v > 500) or
            (hypo and vc in [1,2]) or (hypo and mc in [1,2])):
            h_cde, trigger = h, "D"; break

        if (mc == 1 or vc == 1 or
            (hypo and not np.isnan(lac_v) and 2 <= lac_v < 5) or
            (hypo and not np.isnan(alt_v) and 200 <= alt_v <= 500)):
            h_cde, trigger = h, "C"; break

    results.append({
        "stay_id": sid,
        "h_to_first_CDE": h_cde,
        "trigger_stage": trigger
    })

res_df = pd.DataFrame(results)

nonprog = feat[feat["progressed"] == 0][["stay_id"]].copy()
nonprog["h_to_first_CDE"] = np.nan
nonprog["trigger_stage"]  = None
res_df = pd.concat([res_df, nonprog], ignore_index=True)
res_df = res_df.merge(feat[["stay_id","progressed","scai_baseline"]], on="stay_id")

prog_res = res_df[res_df["progressed"] == 1]
detected = prog_res["h_to_first_CDE"].notna().sum()
print(f"\n=== Results ===")
print(f"First C/D/E time identified: {detected} of {len(prog_res)} progressed admissions ({detected/len(prog_res)*100:.1f}%)")
print(f"\nh_to_first_CDE distribution:")
print(prog_res["h_to_first_CDE"].describe())
print(f"\nStage reached:")
print(prog_res["trigger_stage"].value_counts())
print(f"\nMedian by baseline stage:")
for stage in ["A","B"]:
    sub = prog_res[prog_res["scai_baseline"] == stage]
    med = sub["h_to_first_CDE"].median()
    print(f"  SCAI {stage}: median={med:.1f} h (n={len(sub)})")

out = os.path.join(OUTDIR, "08_progression_timing.csv")
res_df.to_csv(out, index=False)
print(f"\n✅ Saved: {out}")
