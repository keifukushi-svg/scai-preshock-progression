import pandas as pd
import numpy as np
import os
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              f1_score, accuracy_score, precision_score, recall_score)

SCAI_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(SCAI_DIR, "output")

MODEL_FILE = {"XGBoost": "xgboost", "LightGBM": "lightgbm",
              "CatBoost": "catboost", "LogisticRegression": "logisticregression"}

df = pd.read_csv(os.path.join(OUTDIR, "07_features_AB.csv"))
cohort = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"), usecols=["stay_id", "anchor_year_group"])
df_merged = df.merge(cohort, on="stay_id", how="left")
df_val = df_merged[df_merged["anchor_year_group"].isin(["2017 - 2019"])].copy()
y_val = df_val["progressed"].values

derivation = pd.read_csv(os.path.join(OUTDIR, "10_model_results.csv"), index_col=0)

rows = {}
for name, fname in MODEL_FILE.items():
    prob = np.load(os.path.join(OUTDIR, f"13_temporal_prob_{fname}.npy"))
    threshold = float(derivation.loc[name, "threshold"])
    y_pred = (prob >= threshold).astype(int)
    rows[name] = {
        "auroc": roc_auc_score(y_val, prob),
        "auprc": average_precision_score(y_val, prob),
        "accuracy": accuracy_score(y_val, y_pred),
        "sensitivity": recall_score(y_val, y_pred, zero_division=0),
        "specificity": recall_score(1 - y_val, 1 - y_pred, zero_division=0),
        "ppv": precision_score(y_val, y_pred, zero_division=0),
        "npv": precision_score(1 - y_val, 1 - y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "brier": brier_score_loss(y_val, prob),
        "threshold": threshold,
    }

results_df = pd.DataFrame(rows).T
results_df = results_df[["auroc", "auprc", "accuracy", "sensitivity", "specificity",
                          "ppv", "npv", "f1", "brier", "threshold"]].round(3)
print("Temporal validation metrics — primary-analysis threshold applied (no re-optimization on holdout):")
print(results_df.to_string())

results_df.to_csv(os.path.join(OUTDIR, "14_temporal_validation_primary_threshold.csv"))
print("\n✅ Saved: output/14_temporal_validation_primary_threshold.csv")
