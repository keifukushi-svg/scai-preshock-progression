import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
B = 2000


def boot_strat(y, p, seed):
    rng = np.random.default_rng(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    a = [roc_auc_score(y[b], p[b]) for b in
         (np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]) for _ in range(B))]
    return np.percentile(a, [2.5, 97.5])


df = pd.read_csv(os.path.join(OUTDIR, "07_landmark_dataset.csv"))
fu = pd.read_csv(os.path.join(OUTDIR, "08_followup_6_24h.csv")).set_index("stay_id").loc[df["stay_id"]]
yp = fu["physiology_only"].values == 1
y = df["progression"].values == 1
ho = df["holdout"].values == 1
pt = np.load(os.path.join(OUTDIR, "10_temporal_catboost.npy"))
pn = np.load(os.path.join(OUTDIR, "10_nested_oof_catboost.npy"))

res = {"physiology_only_progression": int(yp.sum()), "treatment_only_progression": int((y & ~yp).sum())}
keep = ~(y & ~yp)
for label, k in [("treatment_only_as_non_events", np.ones(len(y), bool)), ("treatment_only_removed", keep)]:
    yv = yp.astype(int)
    lo_t, hi_t = boot_strat(yv[ho & k], pt[k[ho]], 43)
    lo_n, hi_n = boot_strat(yv[k], pn[k], 42)
    res[f"physiology_only_{label}"] = {
        "temporal": [round(roc_auc_score(yv[ho & k], pt[k[ho]]), 3), round(lo_t, 3), round(hi_t, 3)],
        "nested": [round(roc_auc_score(yv[k], pn[k]), 3), round(lo_n, 3), round(hi_n, 3)]}

grp = np.where(yp, "P", np.where(y, "T", "N"))
for name, p, m in [("nested", pn, np.ones(len(df), bool)), ("temporal", pt, ho)]:
    gg = grp[m]
    s = (gg == "T") | (gg == "N")
    res[f"{name}_T_vs_N"] = round(roc_auc_score((gg[s] == "T").astype(int), p[s]), 3)

res.update({f"followup_{v}_measured_pct": round(fu[f"{v}_measured"].mean() * 100, 1) for v in ["lactate", "alt", "ph"]})
add = ~y & (fu["lactate_measured"].values == 0) & (fu["hypotension"].values == 1)
res.update({"no_lactate_hypotension_not_progressed": int(add.sum()),
            "upper_bound_progression": int((y | add).sum()), "upper_bound_pct": round((y | add).mean() * 100, 1)})
json.dump(res, open(os.path.join(OUTDIR, "12_sensitivity_analyses.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
