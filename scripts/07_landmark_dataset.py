import os
import warnings
import pandas as pd
from scai_criteria import VASOACTIVE_ITEMIDS, MCS_ITEMIDS, classify_scai_cswg

DATADIR_ICU  = "/Users/fukushikei/Documents/mimic-iv-3.1/icu"
OUTDIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

L = 6
CHART_VARS = ["hr", "sbp", "map", "rr", "temp_f", "spo2", "ambulatory_aid"]
LAB_VARS = ["wbc", "plt", "hgb", "tbil", "albumin", "sodium", "potassium", "creatinine", "calcium",
            "ast", "alt", "bun", "pt", "aptt", "inr"]
GAS_VARS = ["lactate", "ph", "pao2", "paco2", "base_excess"]
TV_VARS = CHART_VARS + ["gcs_total"] + GAS_VARS + LAB_VARS
STATIC = ["age", "gender_male", "weight_kg", "height_cm", "bmi", "ami_flag", "ahf_flag",
          "htn", "dm", "hyperlipidemia", "copd", "pneumonia", "ckd", "af"]
MEDS = ["acei_arb_baseline", "betablocker_baseline", "furosemide_baseline", "spironolactone_baseline"]

coh = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"), parse_dates=["icu_intime", "icu_outtime", "deathtime"])
coh = coh.set_index("stay_id")
ids = coh.index
full = pd.read_csv(os.path.join(OUTDIR, "06_features_full.csv"))
ab = pd.Series(ids.isin(full.loc[full["scai_baseline"].isin(["A", "B"]), "stay_id"]), index=ids)
demo = pd.read_csv(os.path.join(OUTDIR, "04_comorbidities_demographics.csv")).set_index("stay_id")
bl = pd.read_csv(os.path.join(OUTDIR, "02_baseline_vitals_labs.csv")).set_index("stay_id").reindex(ids)
trt = pd.read_csv(os.path.join(OUTDIR, "03_baseline_treatments.csv")).set_index("stay_id").reindex(ids)
chart = pd.read_csv(os.path.join(OUTDIR, "02_chartevents_0_24h.csv.gz"))
lab = pd.read_csv(os.path.join(OUTDIR, "02_labevents_0_24h.csv.gz"))

gcs_parts = chart[chart["var"].isin(["gcs_eye", "gcs_verbal", "gcs_motor"])]
gcs = (gcs_parts.pivot_table(index=["stay_id", "charttime", "h"], columns="var", values="valuenum", aggfunc="first")
       .fillna(1).sum(axis=1).rename("valuenum").reset_index().assign(var="gcs_total"))
pooled = pd.concat([
    chart.loc[chart["var"].isin(CHART_VARS), ["stay_id", "charttime", "h", "var", "valuenum"]],
    chart.loc[chart["var"].str.endswith("_chart"), ["stay_id", "charttime", "h", "var", "valuenum"]]
    .assign(var=lambda d: d["var"].str.replace("_chart", "", regex=False)),
    lab[["stay_id", "charttime", "h", "var", "valuenum"]],
    gcs[["stay_id", "charttime", "h", "var", "valuenum"]],
], ignore_index=True)
bp = (chart[chart["var"].isin(["sbp", "map"])].groupby(["stay_id", "var", "charttime", "h"])["valuenum"].min()
      .reset_index())

drugs = pd.read_csv(os.path.join(DATADIR_ICU, "inputevents.csv"), usecols=["stay_id", "starttime", "itemid"],
                    parse_dates=["starttime"])
drugs = drugs[drugs["stay_id"].isin(ids) & drugs["itemid"].isin(VASOACTIVE_ITEMIDS)].copy()
drugs["what"] = drugs["itemid"].map(VASOACTIVE_ITEMIDS)
drugs["h"] = (drugs["starttime"] - drugs["stay_id"].map(coh["icu_intime"])).dt.total_seconds() / 3600
proc = pd.read_csv(os.path.join(DATADIR_ICU, "procedureevents.csv"), usecols=["stay_id", "starttime", "itemid"],
                   parse_dates=["starttime"])
proc = proc[proc["stay_id"].isin(ids) & proc["itemid"].isin(MCS_ITEMIDS)].copy()
proc["what"] = proc["itemid"].map(MCS_ITEMIDS)
proc["h"] = (proc["starttime"] - proc["stay_id"].map(coh["icu_intime"])).dt.total_seconds() / 3600


def n_started(d, lo, hi):
    return d[(d["h"] > lo) & (d["h"] <= hi)].groupby("stay_id")["what"].nunique().reindex(ids).fillna(0)


def in_window(d, lo, hi):
    return d[(d["h"] >= lo) & (d["h"] <= hi)] if lo == 0 else d[(d["h"] > lo) & (d["h"] <= hi)]


def staging_values(lo, hi):
    w = in_window(pooled[pooled["var"].isin(["lactate", "ph", "alt"])], lo, hi)
    d = in_window(bp, lo, hi).sort_values("valuenum")
    b = d[d.groupby(["stay_id", "var"]).cumcount() == 1].set_index(["stay_id", "var"])["valuenum"]
    b = b.unstack("var").reindex(ids)
    g = w.groupby(["stay_id", "var"])["valuenum"]
    mx, mn = g.max().unstack("var").reindex(ids), g.min().unstack("var").reindex(ids)
    return pd.DataFrame({"sbp": b.get("sbp"), "map": b.get("map"), "lactate": mx.get("lactate"),
                         "alt": mx.get("alt"), "ph": mn.get("ph")}, index=ids)


def cde(v, vaso, mcd):
    return pd.Series([classify_scai_cswg(r["sbp"], r["map"], r["lactate"], r["alt"], r["ph"], mcd[i], vaso[i])
                      for i, r in v.iterrows()], index=v.index)


death_h = (coh["deathtime"] - coh["icu_intime"]).dt.total_seconds() / 3600
icu_h = (coh["icu_outtime"] - coh["icu_intime"]).dt.total_seconds() / 3600
vaso_L, mcd_L = trt["vaso_count_baseline"], trt["mcd_count_baseline"]
left = (icu_h < L) | (death_h < L)
vaso_o, mcd_o = n_started(drugs, L, 24), n_started(proc, L, 24)

# First, minimum, and maximum values and fluid input are kept for the descriptive tables;
# the models use the most recent value of each time-varying variable (steps 9 and 10).
feats = pd.DataFrame(index=ids)
for v in STATIC:
    feats[v] = coh[v] if v in ("age", "ami_flag", "ahf_flag") else demo[v].reindex(ids)
w = in_window(pooled, 0, L).sort_values("h")
g = w.groupby(["stay_id", "var"])["valuenum"]
last, mn, mx = (g.last().unstack("var").reindex(ids), g.min().unstack("var").reindex(ids),
                g.max().unstack("var").reindex(ids))
for v in TV_VARS:
    feats[f"{v}_first"], feats[f"{v}_last"] = bl[v], last.get(v)
    feats[f"{v}_min"], feats[f"{v}_max"] = mn.get(v), mx.get(v)
for m in MEDS:
    feats[m] = trt[m]
feats["fluid_input_6h"] = trt["fluid_input_6h"]

by_L = cde(staging_values(0, L), vaso_L, mcd_L).isin(["C", "D", "E"])
cohort = ab & ~by_L & ~left
y = cde(staging_values(L, 24), vaso_o, mcd_o).isin(["C", "D", "E"])
feats["progression"] = y.astype(int)
feats["holdout"] = coh["anchor_year_group"].eq("2017 - 2019").astype(int)
feats["subject_id"] = coh["subject_id"]
feats["inhospital_death"] = coh["hospital_expire_flag"].astype(int)
feats[cohort].to_csv(os.path.join(OUTDIR, "07_landmark_dataset.csv"), index_label="stay_id")

print(f"Stage A/B at baseline: {int(ab.sum()):,}; stage C/D/E within 0-{L} h: {int((ab & by_L).sum()):,}; "
      f"ICU discharge or death before {L} h: {int((ab & ~by_L & left).sum()):,}")
print(f"Landmark cohort: {int(cohort.sum()):,}; progression {L}-24 h: {int((cohort & y).sum()):,} "
      f"({(cohort & y).sum() / cohort.sum() * 100:.1f}%)")
for per, m in [("2008-2016", ~coh["anchor_year_group"].eq("2017 - 2019")),
               ("2017-2019", coh["anchor_year_group"].eq("2017 - 2019"))]:
    c = cohort & m
    print(f"  {per}: {int(c.sum()):,}, progression {int((c & y).sum()):,} ({(c & y).sum() / c.sum() * 100:.1f}%)")
print(f"Candidate predictors (most recent value of each time-varying variable): {len(STATIC) + len(TV_VARS) + len(MEDS)}")
