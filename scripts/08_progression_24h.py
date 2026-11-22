import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from scai_criteria import VASOACTIVE_ITEMIDS, MCS_ITEMIDS, classify_scai_cswg

DATADIR_ICU = "/path/to/mimic-iv-3.1/icu"
OUTDIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

L = 6
CDE = ["C", "D", "E"]

coh = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"), parse_dates=["icu_intime"]).set_index("stay_id")
ids = coh.index
full = pd.read_csv(os.path.join(OUTDIR, "06_features_full.csv")).set_index("stay_id").reindex(ids)
base = full["scai_baseline"]
ab = base.isin(["A", "B"])
death = full["inhospital_death"]
lm = pd.read_csv(os.path.join(OUTDIR, "07_landmark_dataset.csv")).set_index("stay_id")
cohort = pd.Series(ids.isin(lm.index), index=ids)
chart = pd.read_csv(os.path.join(OUTDIR, "02_chartevents_0_24h.csv.gz"))
lab = pd.read_csv(os.path.join(OUTDIR, "02_labevents_0_24h.csv.gz"))
gas_chart = chart.loc[chart["var"].str.endswith("_chart"), ["stay_id", "charttime", "h", "var", "valuenum"]] \
    .assign(var=lambda d: d["var"].str.replace("_chart", "", regex=False))
stg = pd.concat([gas_chart, lab[["stay_id", "charttime", "h", "var", "valuenum"]]], ignore_index=True)
stg = stg[stg["var"].isin(["lactate", "ph", "alt"])]
bp = (chart[chart["var"].isin(["sbp", "map"])].groupby(["stay_id", "var", "charttime", "h"])["valuenum"].min()
      .reset_index())


def starts(path, items):
    d = pd.read_csv(path, usecols=["stay_id", "starttime", "itemid"])
    d = d[d["stay_id"].isin(ids) & d["itemid"].isin(items)].copy()
    d["what"] = d["itemid"].map(items)
    d["h"] = (pd.to_datetime(d["starttime"]) - d["stay_id"].map(coh["icu_intime"])).dt.total_seconds() / 3600
    return d


drugs = starts(os.path.join(DATADIR_ICU, "inputevents.csv"), VASOACTIVE_ITEMIDS)
devs = starts(os.path.join(DATADIR_ICU, "procedureevents.csv"), MCS_ITEMIDS)


def n_started(d, lo, hi):
    return d[(d["h"] > lo) & (d["h"] <= hi)].groupby("stay_id")["what"].nunique().reindex(ids).fillna(0)


def in_window(d, lo, hi):
    return d[(d["h"] >= lo) & (d["h"] <= hi)] if lo == 0 else d[(d["h"] > lo) & (d["h"] <= hi)]


def staging_values(lo, hi):
    w, d = in_window(stg, lo, hi), in_window(bp, lo, hi).sort_values("valuenum")
    b = d[d.groupby(["stay_id", "var"]).cumcount() == 1].set_index(["stay_id", "var"])["valuenum"]
    b = b.unstack("var").reindex(ids)
    g = w.groupby(["stay_id", "var"])["valuenum"]
    mx, mn = g.max().unstack("var").reindex(ids), g.min().unstack("var").reindex(ids)
    return pd.DataFrame({"sbp": b.get("sbp"), "map": b.get("map"), "lactate": mx.get("lactate"),
                         "alt": mx.get("alt"), "ph": mn.get("ph")}, index=ids)


def stage(v, vaso, mcd):
    return pd.Series([classify_scai_cswg(r["sbp"], r["map"], r["lactate"], r["alt"], r["ph"], mcd[i], vaso[i])
                      for i, r in v.iterrows()], index=v.index)


def pct(m):
    return f"{int(m.sum()):,}/{len(m):,} ({m.mean() * 100:.1f}%)"


def wilson(e, n):
    lo, hi = proportion_confint(e, n, method="wilson")
    return f"{e:,}/{n:,} = {e / n * 100:.1f}% (95% CI {lo * 100:.1f}-{hi * 100:.1f})"


def criteria(sel, v, va, mc, phys, st):
    p, pv, pm = v[sel], va[sel], mc[sel]
    hyp = (p["sbp"] <= 90) | (p["map"] <= 65)
    print("  stage reached: " + ", ".join(f"{k} {n:,}" for k, n in st[sel].value_counts().sort_index().items()))
    print(f"  hypotension (SBP<=90 or MAP<=65) {pct(hyp)}; lactate>=2 {pct(p['lactate'] >= 2)}; "
          f"lactate>=5 {pct(p['lactate'] >= 5)}; ALT>=200 {pct(p['alt'] >= 200)}; pH<7.2 {pct(p['ph'] < 7.2)}")
    print(f"  vasoactive drug started {pct(pv >= 1)}; MCS started {pct(pm >= 1)}")
    print(f"  criteria met by physiological values alone {pct(phys[sel])}")
    print(f"  criteria met only because a drug or device was started {pct(~phys[sel])}: with hypotension "
          f"{int((~phys[sel] & hyp).sum()):,}, without hypotension {int((~phys[sel] & ~hyp).sum()):,}")


zero = pd.Series(0, index=ids)

print("=== Progression within 24 h (stage A/B at baseline) ===")
v24 = staging_values(0, 24)
vaso24, mcd24 = n_started(drugs, -np.inf, 24), n_started(devs, -np.inf, 24)
scai_24h = stage(v24, vaso24, mcd24)
progressed = scai_24h.isin(CDE) & ab
print(f"Progression {pct(progressed[ab])}")
for s in ["A", "B"]:
    m = ab & base.eq(s)
    print(f"  baseline stage {s}: {wilson(int((m & progressed).sum()), int(m.sum()))}")
print(pd.crosstab(base[ab], scai_24h[ab].replace({"A (needs evaluation)": "A*"}).rename("scai_24h")).to_string())
mort = death[ab].groupby(scai_24h[ab].replace({"A (needs evaluation)": "A"})).agg(["size", "sum", "mean"])
print("In-hospital death by stage at 24 h: " + "; ".join(
    f"{s} {int(r['sum'])}/{int(r['size'])} ({r['mean'] * 100:.1f}%)" for s, r in mort.iterrows()))
for g, name in [(False, "no progression"), (True, "progression")]:
    x = death[ab & (progressed == g)]
    print(f"In-hospital death, {name}: {int(x.sum())}/{len(x)} ({x.mean() * 100:.1f}%)")
print(f"  Fisher P {stats.fisher_exact(pd.crosstab(progressed[ab], death[ab]).values)[1]:.2g}")
print("Criteria among progressions within 24 h")
criteria(progressed, v24, vaso24, mcd24, stage(v24, zero, zero).isin(CDE) & ab, scai_24h)

print("\n=== Progression between 6 and 24 h (landmark cohort) ===")
vo = staging_values(L, 24)
vaso_o, mcd_o = n_started(drugs, L, 24), n_started(devs, L, 24)
st_o = stage(vo, vaso_o, mcd_o)
y = st_o.isin(CDE) & cohort
print(f"Progression {pct(y[cohort])}")
for s in ["A", "B"]:
    m = cohort & base.eq(s)
    print(f"  baseline stage {s}: {wilson(int((m & y).sum()), int(m.sum()))}")
for g, name in [(False, "no progression"), (True, "progression")]:
    x = death[cohort & (y == g)]
    print(f"In-hospital death, {name}: {int(x.sum())}/{len(x)} ({x.mean() * 100:.1f}%)")
print(f"  Fisher P {stats.fisher_exact(pd.crosstab(y[cohort], death[cohort]).values)[1]:.2g}")
print("Criteria among progressions between 6 and 24 h")
y_phys = stage(vo, zero, zero).isin(CDE) & cohort
criteria(y, vo, vaso_o, mcd_o, y_phys, st_o)
tonly = y & ~y_phys
dstart = drugs[(drugs["h"] > L) & (drugs["h"] <= 24) & drugs["stay_id"].isin(tonly[tonly].index)]
print("  drugs started in treatment-only progressions (admissions): " + ", ".join(
    f"{k} {v:,}" for k, v in dstart.groupby("what")["stay_id"].nunique().sort_values(ascending=False).items()))
pd.DataFrame({"physiology_only": y_phys[cohort].astype(int),
              "lactate_measured": vo.loc[cohort, "lactate"].notna().astype(int),
              "alt_measured": vo.loc[cohort, "alt"].notna().astype(int),
              "ph_measured": vo.loc[cohort, "ph"].notna().astype(int),
              "hypotension": ((vo["sbp"] <= 90) | (vo["map"] <= 65))[cohort].astype(int)}
             ).to_csv(os.path.join(OUTDIR, "08_followup_6_24h.csv"), index_label="stay_id")
