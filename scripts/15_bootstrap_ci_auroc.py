import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

O = "output"
B = 2000
MODELS = ["catboost", "xgboost", "lightgbm", "logisticregression"]
NAME = {"catboost": "CatBoost", "xgboost": "XGBoost",
        "lightgbm": "LightGBM", "logisticregression": "Logistic Regression"}

df = pd.read_csv(os.path.join(O, "07_features_AB.csv"))
y_cv = df["progressed"].values
cohort = pd.read_csv(os.path.join(O, "01_cohort.csv"),
                     usecols=["stay_id", "anchor_year_group"])
mg = df.merge(cohort, on="stay_id", how="left")
y_tv = mg[mg["anchor_year_group"] == "2017 - 2019"]["progressed"].values


def boot_indices(y, B, seed):
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    return [np.concatenate([rng.choice(idx0, len(idx0), replace=True),
                            rng.choice(idx1, len(idx1), replace=True)])
            for _ in range(B)]


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return lo, hi


def run(y, probs, seed, label):
    bidx = boot_indices(y, B, seed)
    rows = []
    for m in MODELS:
        p = probs[m]
        au = np.array([roc_auc_score(y[b], p[b]) for b in bidx])
        ap = np.array([average_precision_score(y[b], p[b]) for b in bidx])
        br = np.array([brier_score_loss(y[b], p[b]) for b in bidx])
        rows.append({"cohort": label, "model": NAME[m],
                     "AUROC": round(roc_auc_score(y, p), 3),
                     "AUROC_lo": round(ci(au)[0], 3), "AUROC_hi": round(ci(au)[1], 3),
                     "AUPRC": round(average_precision_score(y, p), 3),
                     "AUPRC_lo": round(ci(ap)[0], 3), "AUPRC_hi": round(ci(ap)[1], 3),
                     "Brier": round(brier_score_loss(y, p), 3),
                     "Brier_lo": round(ci(br)[0], 3), "Brier_hi": round(ci(br)[1], 3)})
    pc = probs["catboost"]
    for m in MODELS:
        if m == "catboost":
            continue
        d = roc_auc_score(y, pc) - roc_auc_score(y, probs[m])
        ds = np.array([roc_auc_score(y[b], pc[b]) - roc_auc_score(y[b], probs[m][b])
                       for b in bidx])
        rows.append({"cohort": label, "model": f"CatBoost-vs-{NAME[m]}",
                     "AUROC": round(d, 3),
                     "AUROC_lo": round(ci(ds)[0], 3), "AUROC_hi": round(ci(ds)[1], 3)})
    return rows


cv_probs = {m: np.load(os.path.join(O, f"10_oof_{m}_8var.npy")) for m in MODELS}
tv_probs = {m: np.load(os.path.join(O, f"13_temporal_prob_{m}_8var.npy")) for m in MODELS}

out = run(y_cv, cv_probs, 42, "Primary CV (n=3806)")
out += run(y_tv, tv_probs, 43, "Temporal 2017-2019 (n=640)")
res = pd.DataFrame(out)
res.to_csv(os.path.join(O, "15_bootstrap_ci.csv"), index=False)
print(res.to_string(index=False))
print("\nsaved -> output/15_bootstrap_ci.csv")
