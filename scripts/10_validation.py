import os
import re
import sys
import json
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
import statsmodels.api as sm
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "output")
NON_FEATURES = {"stay_id", "progression", "holdout", "subject_id", "inhospital_death"}
MODELS = ["XGBoost", "LightGBM", "CatBoost", "LogisticRegression"]
RS = 42
B = 2000
GRID = np.round(np.arange(0.10, 0.90, 0.05), 2)


def load():
    df = pd.read_csv(os.path.join(OUTDIR, "07_landmark_dataset.csv"))
    # Candidates: the most recent value of each time-varying variable within 0-6 h, with the static variables and medications
    cand = [c for c in df.columns if c not in NON_FEATURES
            and c != "fluid_input_6h" and (c.endswith("_last") or not re.search(r"_(first|last|min|max)$", c))]
    return df, cand, df["progression"].values


def select(d, y):
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    cand = [c for c in d.columns if d[c].notna().any()]
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(d[cand]), columns=cand)
    np.random.seed(RS)
    rf = RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=1, max_depth=5)
    bor = BorutaPy(rf, n_estimators="auto", verbose=0, random_state=RS, max_iter=50)
    bor.fit(X.values, y)
    Xs = X[[c for c, s, w in zip(cand, bor.support_, bor.support_weak_) if s or w]].copy()
    while len(Xs.columns) > 1:
        Xc = add_constant(Xs, has_constant="add")
        vifs = {c: variance_inflation_factor(Xc.values, i) for i, c in enumerate(Xc.columns) if c != "const"}
        worst = max(vifs, key=vifs.get)
        if vifs[worst] <= 10:
            break
        Xs = Xs.drop(columns=[worst])
    return list(Xs.columns)


def f1_threshold(y, p):
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in GRID]
    return float(GRID[int(np.argmax(f1s))])


def develop(name, X, y, groups, tr, te):
    from model_utils import tune, fit_calibrated, preprocess
    X_tr, X_te = preprocess(name, X[tr], X[te])
    params = tune(name, X_tr, y[tr])
    p_te = fit_calibrated(name, X_tr, y[tr], params).predict_proba(X_te)[:, 1]
    inner = np.zeros(len(tr))
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=RS).split(X[tr], y[tr], groups[tr]):
        Xa, Xb = preprocess(name, X[tr][a], X[tr][b])
        inner[b] = fit_calibrated(name, Xa, y[tr][a], params).predict_proba(Xb)[:, 1]
    return p_te, f1_threshold(y[tr], inner), params


def task(kind, k, tr, te):
    warnings.filterwarnings("ignore")
    sys.path.insert(0, HERE)
    df, cand, y = load()
    groups = df["subject_id"].values
    if kind == "temporal":
        sel = open(os.path.join(OUTDIR, "09_selected_features.txt")).read().split()
    else:
        sel = select(df.iloc[tr][cand], y[tr])
    X = df[sel].values.astype(float)
    return kind, k, sel, {name: develop(name, X, y, groups, tr, te) for name in MODELS}


def boot_strat(y, p, seed):
    rng = np.random.default_rng(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    a = [roc_auc_score(y[b], p[b]) for b in
         (np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]) for _ in range(B))]
    return np.percentile(a, [2.5, 97.5])


def boot_cluster(y, p, g, seed):
    rng = np.random.default_rng(seed)
    _, inv = np.unique(g, return_inverse=True)
    idx = pd.Series(np.arange(len(g))).groupby(inv).apply(np.array).tolist()
    a = []
    for _ in range(B):
        b = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        if 0 < y[b].sum() < len(b):
            a.append(roc_auc_score(y[b], p[b]))
    return np.percentile(a, [2.5, 97.5])


def calibration(y, p):
    lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return citl, slope, y.mean() / p.mean()


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    tn, fn = int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    citl, slope, oe = calibration(y, p)
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "cal_intercept": citl, "cal_slope": slope, "obs_exp": oe,
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "ppv": tp / (tp + fp) if tp + fp else np.nan, "npv": tn / (tn + fn) if tn + fn else np.nan,
            "f1": 2 * tp / (2 * tp + fp + fn), "accuracy": (tp + tn) / len(y)}


if __name__ == "__main__":
    df, cand, y = load()
    groups, ho = df["subject_id"].values, df["holdout"].values == 1
    outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=RS).split(np.zeros(len(y)), y, groups))
    tasks = [("temporal", 0, np.where(~ho)[0], np.where(ho)[0])] + \
            [("nested", k, tr, te) for k, (tr, te) in enumerate(outer)]
    print(f"n {len(y):,}, events {int(y.sum()):,}, patients {len(set(groups)):,}; "
          f"temporal {int((~ho).sum()):,}/{int(ho.sum()):,}", flush=True)
    res = Parallel(n_jobs=len(tasks))(delayed(task)(*t) for t in tasks)
    tmp = next(r for r in res if r[0] == "temporal")
    nst = sorted([r for r in res if r[0] == "nested"], key=lambda r: r[1])

    lines, rows, thr_rows, dca_rows = [], [], [], []
    fold_of = np.zeros(len(y), int)
    for k, (_, te) in enumerate(outer):
        fold_of[te] = k
    span = pd.Series(fold_of).groupby(groups).nunique()
    lines.append(f"patients whose admissions fall in more than one outer fold: {int((span > 1).sum())}")
    for name in MODELS:
        pt, tt, params = tmp[3][name]
        oof, thr = np.zeros(len(y)), np.zeros(len(y))
        for r in nst:
            te = outer[r[1]][1]
            oof[te], thr[te] = r[3][name][0], r[3][name][1]
            thr_rows.append({"model": name, "set": f"fold {r[1] + 1}", "threshold": r[3][name][1]})
        thr_rows.append({"model": name, "set": "2008-2016", "threshold": tt})
        np.save(os.path.join(OUTDIR, f"10_temporal_{name.lower()}.npy"), pt)
        np.save(os.path.join(OUTDIR, f"10_nested_oof_{name.lower()}.npy"), oof)
        if name == "CatBoost":
            json.dump(params, open(os.path.join(OUTDIR, "10_catboost_params.json"), "w"), indent=1)
        for ev, yv, p, th, g, seed in [("temporal", y[ho], pt, tt, groups[ho], 43), ("nested", y, oof, thr, groups, 42)]:
            m = metrics(yv, p, th)
            lo, hi = boot_strat(yv, p, seed)
            clo, chi = boot_cluster(yv, p, g, seed)
            rows.append({"evaluation": ev, "model": name, "n": len(yv), "events": int(yv.sum()),
                         **{k: round(v, 3) for k, v in m.items()}, "auroc_lo": round(lo, 3), "auroc_hi": round(hi, 3),
                         "auroc_lo_patient": round(clo, 3), "auroc_hi_patient": round(chi, 3),
                         "threshold": th if np.isscalar(th) else "per fold"})
            prev = yv.mean()
            for t in np.round(np.arange(0.01, 0.81, 0.01), 2):
                tp_ = ((p >= t) & (yv == 1)).sum() / len(yv)
                fp_ = ((p >= t) & (yv == 0)).sum() / len(yv)
                dca_rows.append({"evaluation": ev, "model": name, "threshold": t, "net_benefit": tp_ - fp_ * t / (1 - t),
                                 "treat_all": prev - (1 - prev) * t / (1 - t), "treat_none": 0.0})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTDIR, "10_results.csv"), index=False)
    pd.DataFrame(thr_rows).to_csv(os.path.join(OUTDIR, "10_thresholds.csv"), index=False)
    pd.DataFrame(dca_rows).to_csv(os.path.join(OUTDIR, "10_decision_curve.csv"), index=False)

    feat = [{"set": "2008-2016", "n": len(tmp[2]), "features": " ".join(tmp[2])}] + \
           [{"set": f"fold {r[1] + 1}", "n": len(r[2]), "features": " ".join(r[2])} for r in nst]
    fr = pd.DataFrame(feat)
    fr.to_csv(os.path.join(OUTDIR, "10_features.csv"), index=False)
    dev = set(tmp[2])
    cnt = pd.Series([f for r in nst for f in r[2]]).value_counts()
    cols = ["evaluation", "model", "n", "events", "auroc", "auroc_lo", "auroc_hi", "auroc_lo_patient",
            "auroc_hi_patient", "auprc", "brier", "cal_intercept", "cal_slope", "obs_exp", "threshold",
            "sensitivity", "specificity", "ppv", "npv", "f1", "accuracy"]
    print("\n".join([out[cols].to_string(index=False), "", pd.DataFrame(thr_rows).pivot(index="model", columns="set",
                     values="threshold").to_string(), ""] + lines +
                    ["", fr[["set", "n"]].to_string(index=False),
                     "folds selecting each feature (* = in the 2008-2016 set): " +
                     ", ".join(f"{f}{'*' if f in dev else ''} {n}" for f, n in cnt.items())]))
