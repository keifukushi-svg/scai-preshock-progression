import pandas as pd
import numpy as np
import os
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from model_utils import RANDOM_STATE, MODELS, load_features, tune, fit_calibrated, preprocess
warnings.filterwarnings('ignore')

SCAI_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR   = os.path.join(SCAI_DIR, "output")
np.random.seed(RANDOM_STATE)

FEATURES = load_features(OUTDIR)
TARGET = "progressed"

df = pd.read_csv(os.path.join(OUTDIR, "07_features_AB.csv"))
cohort = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"), usecols=["stay_id", "anchor_year_group"])
df = df.merge(cohort, on="stay_id", how="left")
train = df[df["anchor_year_group"].isin(["2008 - 2010", "2011 - 2013", "2014 - 2016"])]
hold = df[df["anchor_year_group"] == "2017 - 2019"]
print(f"Derivation 2008-2016: n={len(train):,} (progressed within 24 h {train[TARGET].mean() * 100:.1f}%)")
print(f"Holdout 2017-2019   : n={len(hold):,} (progressed within 24 h {hold[TARGET].mean() * 100:.1f}%)")

X_tr_raw = train[FEATURES].values.astype(float)
X_ho_raw = hold[FEATURES].values.astype(float)
y_tr, y_ho = train[TARGET].values, hold[TARGET].values

rows = {}
for name in MODELS:
    X_tr, X_ho = preprocess(name, X_tr_raw, X_ho_raw)
    prob = fit_calibrated(name, X_tr, y_tr, tune(name, X_tr, y_tr)).predict_proba(X_ho)[:, 1]
    np.save(os.path.join(OUTDIR, f"13_temporal_prob_{name.lower()}.npy"), prob)
    rows[name] = {"auroc": roc_auc_score(y_ho, prob),
                  "auprc": average_precision_score(y_ho, prob),
                  "brier": brier_score_loss(y_ho, prob),
                  "mean_predicted": prob.mean(), "observed": y_ho.mean()}
    print(f"  {name}: AUROC={rows[name]['auroc']:.3f}", flush=True)

res = pd.DataFrame(rows).T.astype(float).round(3)
print(res.to_string())
res.to_csv(os.path.join(OUTDIR, "13_temporal_validation.csv"))
print("\n✅ Saved: 13_temporal_validation.csv, 13_temporal_prob_*.npy")
