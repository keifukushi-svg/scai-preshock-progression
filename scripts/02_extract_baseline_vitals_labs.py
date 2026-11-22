import pandas as pd
import os
from scai_criteria import plausible

DATADIR_HOSP = "/Users/fukushikei/Documents/mimic-iv-3.1/hosp"
DATADIR_ICU  = "/Users/fukushikei/Documents/mimic-iv-3.1/icu"
OUTDIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

CHUNKSIZE = 2_000_000

cohort = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"),
                     parse_dates=["icu_intime", "icu_outtime",
                                  "baseline_end", "window_24h"])
stay_ids = set(cohort["stay_id"])
hadm_ids = set(cohort["hadm_id"])
print(f"Cohort: {len(cohort):,} admissions, {len(stay_ids):,} stay_ids")

CHART_ITEMS = {
    "hr"       : [220045],
    "sbp"      : [220050, 220179],
    "map"      : [220052, 220181],
    "rr"       : [220210],
    "temp_f"   : [223761],
    "spo2"     : [220277],
    "gcs_eye"  : [220739],
    "gcs_verbal": [223900],
    "gcs_motor": [223901],
    "ph_chart"       : [223830],
    "pao2_chart"     : [220224],
    "paco2_chart"    : [220235],
    "lactate_chart"  : [225668],
    "base_excess_chart": [224828],
    "ambulatory_aid" : [227343],
}
chart_item_to_var = {iid: var for var, ids in CHART_ITEMS.items() for iid in ids}

LAB_ITEMS = {
    "wbc"        : [51301],
    "plt"        : [51265],
    "hgb"        : [51222],
    "tbil"       : [50885],
    "albumin"    : [50862],
    "sodium"     : [50983],
    "potassium"  : [50971],
    "creatinine" : [50912],
    "calcium"    : [50893],
    "ast"        : [50878],
    "alt"        : [50861],
    "bun"        : [51006],
    "pt"         : [51274],
    "aptt"       : [51275],
    "inr"        : [51237],
    "lactate"    : [50813],
    "ph"         : [50820],
    "pao2"       : [50821],
    "paco2"      : [50818],
    "base_excess": [50802],
}
lab_item_to_var = {iid: var for var, ids in LAB_ITEMS.items() for iid in ids}
ARTERIAL_ONLY = {"pao2", "paco2"}
SPECIMEN_TYPE_ITEMID = 52033

print("\n[chartevents] Reading in chunks...")
chart_records = []
for i, chunk in enumerate(pd.read_csv(
        os.path.join(DATADIR_ICU, "chartevents.csv"),
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
        chunksize=CHUNKSIZE)):
    chunk = chunk[chunk["stay_id"].isin(stay_ids) &
                  chunk["itemid"].isin(chart_item_to_var)]
    if len(chunk) > 0:
        chart_records.append(chunk)
    if (i + 1) % 100 == 0:
        print(f"  chunk {i + 1}...", flush=True)

chart_df = pd.concat(chart_records, ignore_index=True)
chart_df["var"] = chart_df["itemid"].map(chart_item_to_var)
chart_df = chart_df.merge(cohort[["stay_id", "icu_intime", "window_24h"]], on="stay_id", how="left")
chart_df = chart_df[(chart_df["charttime"] >= chart_df["icu_intime"]) &
                    (chart_df["charttime"] <= chart_df["window_24h"]) &
                    chart_df["valuenum"].notna()].copy()
keep = pd.Series(True, index=chart_df.index)
for var, idx in chart_df.groupby("var").groups.items():
    keep.loc[idx] = plausible(var, chart_df.loc[idx, "valuenum"])
print(f"  chartevents 0-24 h: {len(chart_df):,} rows, implausible removed: {(~keep).sum():,}")
chart_df = chart_df[keep].copy()
chart_df["h"] = (chart_df["charttime"] - chart_df["icu_intime"]).dt.total_seconds() / 3600
chart_df[["stay_id", "charttime", "h", "var", "valuenum"]].to_csv(
    os.path.join(OUTDIR, "02_chartevents_0_24h.csv.gz"), index=False)

print("\n[labevents] Reading in chunks...")
lab_records = []
spec_records = []
for i, chunk in enumerate(pd.read_csv(
        os.path.join(DATADIR_HOSP, "labevents.csv"),
        usecols=["hadm_id", "specimen_id", "charttime", "itemid", "value", "valuenum"],
        parse_dates=["charttime"],
        chunksize=CHUNKSIZE)):
    chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
    spec = chunk[chunk["itemid"] == SPECIMEN_TYPE_ITEMID]
    if len(spec) > 0:
        spec_records.append(spec[["specimen_id", "value"]])
    chunk = chunk[chunk["itemid"].isin(lab_item_to_var)]
    if len(chunk) > 0:
        lab_records.append(chunk.drop(columns=["value"]))
    if (i + 1) % 100 == 0:
        print(f"  chunk {i + 1}...", flush=True)

spec_df = pd.concat(spec_records, ignore_index=True)
arterial_specimens = set(spec_df.loc[spec_df["value"].astype(str).str.strip() == "ART.", "specimen_id"])

lab_df = pd.concat(lab_records, ignore_index=True)
lab_df["var"] = lab_df["itemid"].map(lab_item_to_var)
gas = lab_df["var"].isin(ARTERIAL_ONLY)
not_arterial = gas & ~lab_df["specimen_id"].isin(arterial_specimens)
print(f"  pO2/pCO2 rows: {gas.sum():,}; not arterial, removed: {not_arterial.sum():,}")
lab_df = lab_df[~not_arterial]
lab_df = lab_df.merge(cohort[["hadm_id", "stay_id", "icu_intime", "window_24h"]], on="hadm_id", how="left")
lab_df = lab_df[(lab_df["charttime"] >= lab_df["icu_intime"]) &
                (lab_df["charttime"] <= lab_df["window_24h"]) &
                lab_df["valuenum"].notna()].copy()
keep = pd.Series(True, index=lab_df.index)
for var, idx in lab_df.groupby("var").groups.items():
    keep.loc[idx] = plausible(var, lab_df.loc[idx, "valuenum"])
print(f"  labevents 0-24 h: {len(lab_df):,} rows, implausible removed: {(~keep).sum():,}")
lab_df = lab_df[keep].copy()
lab_df["h"] = (lab_df["charttime"] - lab_df["icu_intime"]).dt.total_seconds() / 3600
lab_df[["stay_id", "hadm_id", "charttime", "h", "var", "valuenum"]].to_csv(
    os.path.join(OUTDIR, "02_labevents_0_24h.csv.gz"), index=False)


def first_in_baseline(df):
    base = df[df["h"] <= 6].sort_values("h")
    first = base.groupby(["stay_id", "var"])[["valuenum", "h"]].first()
    return first["valuenum"].unstack("var"), first["h"].unstack("var")


result = cohort[["hadm_id", "stay_id"]].set_index("stay_id")

chart_val, chart_h = first_in_baseline(chart_df)
lab_val, lab_h = first_in_baseline(lab_df)
chart_val = chart_val.reindex(index=result.index, columns=list(CHART_ITEMS))
chart_h = chart_h.reindex(index=result.index, columns=list(CHART_ITEMS))
lab_val = lab_val.reindex(index=result.index, columns=list(LAB_ITEMS))
lab_h = lab_h.reindex(index=result.index, columns=list(LAB_ITEMS))

for var in ["hr", "sbp", "map", "rr", "temp_f", "spo2", "ambulatory_aid"]:
    result[var] = chart_val[var]
    result[f"{var}_h"] = chart_h[var]

gcs_cols = ["gcs_eye", "gcs_verbal", "gcs_motor"]
gcs = chart_val[gcs_cols]
result["gcs_total"] = gcs.fillna(1).sum(axis=1).where(gcs.notna().any(axis=1))
result["gcs_total_h"] = chart_h[gcs_cols].max(axis=1)

for var in LAB_ITEMS:
    chart_var = f"{var}_chart"
    if chart_var in CHART_ITEMS:
        result[var] = lab_val[var].combine_first(chart_val[chart_var])
        result[f"{var}_h"] = lab_h[var].where(lab_val[var].notna(), chart_h[chart_var])
    else:
        result[var] = lab_val[var]
        result[f"{var}_h"] = lab_h[var]

result["temp_c"] = (result["temp_f"] - 32) * 5 / 9
result = result.reset_index()

print("\n  Missing rate (%):")
for col in ["sbp", "map", "lactate", "hr", "rr", "spo2", "temp_f", "gcs_total",
            "wbc", "plt", "hgb", "tbil", "albumin", "sodium", "potassium", "creatinine",
            "calcium", "ast", "alt", "bun", "pt", "aptt", "inr",
            "ph", "pao2", "paco2", "base_excess"]:
    print(f"    {col}: {result[col].isna().mean() * 100:.1f}%")

out_path = os.path.join(OUTDIR, "02_baseline_vitals_labs.csv")
result.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")
print(f"   Shape : {result.shape}")
