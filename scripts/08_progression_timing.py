import pandas as pd
import numpy as np
import os
from scai_criteria import VASOACTIVE_ITEMIDS, MCS_ITEMIDS, classify_scai_cswg

DATADIR_ICU = "/Users/fukushikei/Documents/mimic-iv-3.1/icu"
OUTDIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

feat = pd.read_csv(os.path.join(OUTDIR, "07_features_AB.csv"))
cohort = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"), parse_dates=["icu_intime"])
cohort = cohort[cohort["stay_id"].isin(feat["stay_id"])].merge(feat[["stay_id", "progressed"]], on="stay_id")
prog = cohort[cohort["progressed"] == 1].copy()
stay_ids = set(prog["stay_id"])
print(f"A/B cohort: {len(feat):,} (progressed within 24 h: {len(prog):,})")

chart = pd.read_csv(os.path.join(OUTDIR, "02_chartevents_0_24h.csv.gz"))
lab = pd.read_csv(os.path.join(OUTDIR, "02_labevents_0_24h.csv.gz"))
values = pd.concat([
    chart.loc[chart["var"].isin(["sbp", "map", "lactate_chart", "ph_chart"]), ["stay_id", "h", "var", "valuenum"]]
    .assign(var=lambda d: d["var"].str.replace("_chart", "", regex=False)),
    lab.loc[lab["var"].isin(["lactate", "alt", "ph"]), ["stay_id", "h", "var", "valuenum"]],
], ignore_index=True)
values = values[values["stay_id"].isin(stay_ids)].sort_values(["stay_id", "h"])
print(f"  values: {len(values):,} rows")

intime = prog.set_index("stay_id")["icu_intime"]


def intervals(path, item_map):
    d = pd.read_csv(path, usecols=["stay_id", "starttime", "itemid"], parse_dates=["starttime"])
    d = d[d["stay_id"].isin(stay_ids) & d["itemid"].isin(item_map)].copy()
    d["what"] = d["itemid"].map(item_map)
    t0 = d["stay_id"].map(intime)
    d["start_h"] = (d["starttime"] - t0).dt.total_seconds() / 3600
    return d[["stay_id", "what", "start_h"]]


print("[inputevents] vasoactive drug start times...")
drugs = intervals(os.path.join(DATADIR_ICU, "inputevents.csv"), VASOACTIVE_ITEMIDS)
print("[procedureevents] MCS start times...")
devices = intervals(os.path.join(DATADIR_ICU, "procedureevents.csv"), MCS_ITEMIDS)

val_g = dict(tuple(values.groupby("stay_id")))
drug_g = dict(tuple(drugs.groupby("stay_id")))
dev_g = dict(tuple(devices.groupby("stay_id")))


def n_started(iv, h):
    if iv is None:
        return 0
    return iv.loc[iv["start_h"] <= h, "what"].nunique()


rows = []
for r in prog.itertuples():
    sid = r.stay_id
    v = val_g.get(sid)
    dv, mv = drug_g.get(sid), dev_g.get(sid)
    times = set()
    if v is not None:
        times |= set(v["h"])
    if dv is not None:
        times |= set(dv["start_h"])
    if mv is not None:
        times |= set(mv["start_h"])
    latest = {}
    first_h, stage_at = np.nan, None
    v_iter = v.itertuples() if v is not None else iter(())
    pending = next(v_iter, None)
    for h in sorted(t for t in times if 0 <= t <= 24):
        while pending is not None and pending.h <= h:
            latest[pending.var] = pending.valuenum
            pending = next(v_iter, None)
        stage = classify_scai_cswg(latest.get("sbp"), latest.get("map"), latest.get("lactate"),
                                   latest.get("alt"), latest.get("ph"),
                                   n_started(mv, h), n_started(dv, h))
        if stage in ("C", "D", "E"):
            first_h, stage_at = h, stage
            break
    rows.append({"stay_id": sid, "h_to_first_CDE": first_h, "first_stage": stage_at})

res = pd.DataFrame(rows)
res = feat[["stay_id", "progressed", "scai_baseline"]].merge(res, on="stay_id", how="left")
p = res[res["progressed"] == 1]
print(f"\nFirst C/D/E time identified: {p['h_to_first_CDE'].notna().sum()} of {len(p)} "
      f"({p['h_to_first_CDE'].notna().mean() * 100:.1f}%)")
print(p["h_to_first_CDE"].describe())
print(p["first_stage"].value_counts())

out = os.path.join(OUTDIR, "08_progression_timing.csv")
res.to_csv(out, index=False)
print(f"\n✅ Saved: {out}")
