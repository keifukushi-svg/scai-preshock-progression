import os
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

SCAI_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(SCAI_DIR, "output")

MODEL_ORDER = ["XGBoost", "LightGBM", "CatBoost", "LogisticRegression"]
MODEL_FILE = {"XGBoost": "xgboost", "LightGBM": "lightgbm",
              "CatBoost": "catboost", "LogisticRegression": "logisticregression"}

df = pd.read_csv(os.path.join(OUTDIR, "07_features_AB.csv"))
y_arr = df["progressed"].values
oof_probs = {m: np.load(os.path.join(OUTDIR, f"10_oof_{MODEL_FILE[m]}.npy")) for m in MODEL_ORDER}

bins = np.linspace(0.0, 1.0, 11)
cal_records = []
for m in MODEL_ORDER:
    prob_true, prob_pred = calibration_curve(y_arr, oof_probs[m], n_bins=10)
    counts = np.bincount(np.searchsorted(bins[1:-1], oof_probs[m]), minlength=10)
    for k, obs, pred in zip(np.flatnonzero(counts), prob_true, prob_pred):
        cal_records.append({"model": m, "bin_lower": bins[k], "bin_upper": bins[k + 1],
                            "n": int(counts[k]), "mean_predicted": pred, "observed": obs})
pd.DataFrame(cal_records).to_csv(os.path.join(OUTDIR, "12_calibration.csv"), index=False)
print("Saved: output/12_calibration.csv")

thresholds = np.linspace(0.01, 0.80, 300)
n_pts = len(y_arr)
prevalence = y_arr.mean()

dca_records = []
for m in MODEL_ORDER:
    for pt in thresholds:
        pred = (oof_probs[m] >= pt).astype(int)
        tp = int(((pred == 1) & (y_arr == 1)).sum())
        fp = int(((pred == 1) & (y_arr == 0)).sum())
        nb = tp / n_pts - fp / n_pts * pt / (1 - pt)
        dca_records.append({"model": m, "threshold": pt, "net_benefit": nb})
for pt in thresholds:
    dca_records.append({"model": "TreatAll", "threshold": pt,
                        "net_benefit": prevalence - (1 - prevalence) * pt / (1 - pt)})

dca_df = pd.DataFrame(dca_records)
dca_df.to_csv(os.path.join(OUTDIR, "12_dca_net_benefit.csv"), index=False)
print("Saved: output/12_dca_net_benefit.csv")
