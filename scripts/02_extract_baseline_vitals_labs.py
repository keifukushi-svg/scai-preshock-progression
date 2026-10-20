import pandas as pd
import numpy as np
import os

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

all_chart_itemids = set()
item_to_var = {}
for var, ids in CHART_ITEMS.items():
    for iid in ids:
        all_chart_itemids.add(iid)
        item_to_var[iid] = var

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
    "aptt"       : [51274],
    "pt"         : [51275],
    "inr"        : [51237],
    "lactate"    : [50813],
    "ph"         : [50820],
    "pao2"       : [50821],
    "paco2"      : [50818],
    "base_excess": [50802],
}
all_lab_itemids = set()
lab_item_to_var = {}
for var, ids in LAB_ITEMS.items():
    for iid in ids:
        all_lab_itemids.add(iid)
        lab_item_to_var[iid] = var

def get_first_in_window(df, id_col, time_col, val_col, cohort_times,
                        window_start_col="icu_intime",
                        window_end_col="baseline_end"):
    result = {}
    for id_val, group in df.groupby(id_col):
        if id_val not in cohort_times:
            continue
        t_start, t_end = cohort_times[id_val]
        mask = (group[time_col] >= t_start) & (group[time_col] <= t_end)
        sub = group[mask].sort_values(time_col)
        valid = sub[sub[val_col].notna()]
        if len(valid) > 0:
            result[id_val] = valid.iloc[0][val_col]
    return result


stay_windows  = {row["stay_id"]: (row["icu_intime"], row["baseline_end"])
                 for _, row in cohort.iterrows()}
hadm_windows  = {row["hadm_id"]: (row["icu_intime"], row["baseline_end"])
                 for _, row in cohort.iterrows()}

print("\n[chartevents] Reading in chunks...")
chart_records = []
chunk_count = 0

for chunk in pd.read_csv(
        os.path.join(DATADIR_ICU, "chartevents.csv"),
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
        chunksize=CHUNKSIZE):

    chunk = chunk[
        chunk["stay_id"].isin(stay_ids) &
        chunk["itemid"].isin(all_chart_itemids)
    ]
    if len(chunk) > 0:
        chart_records.append(chunk)
    chunk_count += 1
    if chunk_count % 100 == 0:
        print(f"  chunk {chunk_count}...", flush=True)

print(f"  Done. Chunks: {chunk_count}")

if chart_records:
    chart_df = pd.concat(chart_records, ignore_index=True)
    print(f"  Filtered chartevents: {len(chart_df):,} rows")
else:
    chart_df = pd.DataFrame(columns=["stay_id","charttime","itemid","valuenum"])
    print("  WARNING: No chartevents found")

chart_df["var"] = chart_df["itemid"].map(item_to_var)

chart_df = chart_df.merge(
    cohort[["stay_id", "icu_intime", "baseline_end"]],
    on="stay_id", how="left"
)
chart_df = chart_df[
    (chart_df["charttime"] >= chart_df["icu_intime"]) &
    (chart_df["charttime"] <= chart_df["baseline_end"]) &
    chart_df["valuenum"].notna()
]

chart_first = (
    chart_df.sort_values("charttime")
    .groupby(["stay_id", "var"])["valuenum"]
    .first()
    .unstack("var")
    .reset_index()
)
for col in ["gcs_eye", "gcs_verbal", "gcs_motor"]:
    if col not in chart_first.columns:
        chart_first[col] = np.nan
chart_first["gcs_total"] = (
    chart_first["gcs_eye"].fillna(1) +
    chart_first["gcs_verbal"].fillna(1) +
    chart_first["gcs_motor"].fillna(1)
)
all_nan = chart_first[["gcs_eye","gcs_verbal","gcs_motor"]].isna().all(axis=1)
chart_first.loc[all_nan, "gcs_total"] = np.nan

for col in ["ph_chart","pao2_chart","paco2_chart","lactate_chart","base_excess_chart"]:
    if col not in chart_first.columns:
        chart_first[col] = np.nan

print(f"  chart_first shape: {chart_first.shape}")
print(f"  Columns: {chart_first.columns.tolist()}")


print("\n[labevents] Reading in chunks...")
lab_records = []
chunk_count = 0

for chunk in pd.read_csv(
        os.path.join(DATADIR_HOSP, "labevents.csv"),
        usecols=["hadm_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
        chunksize=CHUNKSIZE):

    chunk = chunk[
        chunk["hadm_id"].isin(hadm_ids) &
        chunk["itemid"].isin(all_lab_itemids)
    ]
    if len(chunk) > 0:
        lab_records.append(chunk)
    chunk_count += 1
    if chunk_count % 100 == 0:
        print(f"  chunk {chunk_count}...", flush=True)

print(f"  Done. Chunks: {chunk_count}")

if lab_records:
    lab_df = pd.concat(lab_records, ignore_index=True)
    print(f"  Filtered labevents: {len(lab_df):,} rows")
else:
    lab_df = pd.DataFrame(columns=["hadm_id","charttime","itemid","valuenum"])
    print("  WARNING: No labevents found")

lab_df["var"] = lab_df["itemid"].map(lab_item_to_var)
lab_df = lab_df.merge(
    cohort[["hadm_id", "icu_intime", "baseline_end"]],
    on="hadm_id", how="left"
)
lab_df = lab_df[
    (lab_df["charttime"] >= lab_df["icu_intime"]) &
    (lab_df["charttime"] <= lab_df["baseline_end"]) &
    lab_df["valuenum"].notna()
]

lab_first = (
    lab_df.sort_values("charttime")
    .groupby(["hadm_id", "var"])["valuenum"]
    .first()
    .unstack("var")
    .reset_index()
)
print(f"  lab_first shape: {lab_first.shape}")


print("\n[Merge] Combining all measurements...")

result = cohort[["hadm_id", "stay_id"]].copy()

result = result.merge(chart_first, on="stay_id", how="left")

result = result.merge(lab_first, on="hadm_id", how="left")

for lab_var, chart_var in [("ph","ph_chart"),("pao2","pao2_chart"),
                             ("paco2","paco2_chart"),("lactate","lactate_chart"),
                             ("base_excess","base_excess_chart")]:
    if chart_var in result.columns and lab_var in result.columns:
        result[lab_var] = result[lab_var].combine_first(result[chart_var])
        result = result.drop(columns=[chart_var])
    elif chart_var in result.columns:
        result = result.rename(columns={chart_var: lab_var})

if "temp_f" in result.columns:
    result["temp_c"] = (result["temp_f"] - 32) * 5 / 9

print("\n  Missing rate (%):")
essential = ["sbp", "map", "lactate"]
for col in essential:
    if col in result.columns:
        miss = result[col].isna().mean() * 100
        print(f"    {col}: {miss:.1f}%")

other_vars = ["hr", "rr", "spo2", "temp_f", "gcs_total",
              "wbc", "plt", "hgb", "tbil", "albumin",
              "sodium", "potassium", "creatinine", "calcium",
              "ast", "alt", "bun", "aptt", "pt", "inr",
              "ph", "pao2", "paco2", "base_excess"]
for col in other_vars:
    if col in result.columns:
        miss = result[col].isna().mean() * 100
        print(f"    {col}: {miss:.1f}%")

out_path = os.path.join(OUTDIR, "02_baseline_vitals_labs.csv")
result.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")
print(f"   Shape : {result.shape}")
