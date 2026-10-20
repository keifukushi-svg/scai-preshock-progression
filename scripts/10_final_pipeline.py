import pandas as pd
import numpy as np
import os
import warnings
import shap
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from model_utils import (RANDOM_STATE, MODELS, load_features, make_model, tune, tune_on_fold,
                         fit_calibrated, preprocess, evaluate)
warnings.filterwarnings('ignore')

SCAI_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR   = os.path.join(SCAI_DIR, "output")
np.random.seed(RANDOM_STATE)

FEATURES = load_features(OUTDIR)
TARGET = "progressed"
print(f"Predictors ({len(FEATURES)}): {FEATURES}")

df = pd.read_csv(os.path.join(OUTDIR, "07_features_AB.csv"))
X = df[FEATURES].values.astype(float)
y = df[TARGET].values
print(f"A/B cohort: n={len(y):,}, progressed within 24 h: {y.sum():,} ({y.mean() * 100:.1f}%)")

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results, oof = {}, {}
for name in MODELS:
    print(f"\n[{name}]")
    p = np.zeros(len(y))
    for fold, (tr, va) in enumerate(outer.split(X, y)):
        X_tr, X_va = preprocess(name, X[tr], X[va])
        params = tune_on_fold(name, X_tr, y[tr], X_va, y[va])
        p[va] = fit_calibrated(name, X_tr, y[tr], params).predict_proba(X_va)[:, 1]
        print(f"  Fold {fold + 1}: AUROC={roc_auc_score(y[va], p[va]):.3f}", flush=True)
    oof[name] = p
    results[name] = evaluate(y, p)
    np.save(os.path.join(OUTDIR, f"10_oof_{name.lower()}.npy"), p)

results_df = pd.DataFrame(results).T[["auroc", "auprc", "accuracy", "sensitivity", "specificity",
                                       "ppv", "npv", "f1", "brier", "threshold"]].astype(float).round(3)
print("\nPrimary 5-fold cross-validation (out-of-fold predictions):")
print(results_df.to_string())
results_df.to_csv(os.path.join(OUTDIR, "10_model_results.csv"))

print("\n[SHAP] CatBoost fitted on the full A/B cohort")
X_full = SimpleImputer(strategy="median").fit_transform(X)
best = tune("CatBoost", X_full, y)
final_cat = make_model("CatBoost", best, y).fit(X_full, y)
shap_values = shap.TreeExplainer(final_cat).shap_values(X_full)
shap_df = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}
                       ).sort_values("mean_abs_shap", ascending=False)
print(shap_df.to_string(index=False))
shap_df.to_csv(os.path.join(OUTDIR, "10_shap_values.csv"), index=False)
np.save(os.path.join(OUTDIR, "10_shap_matrix.npy"), shap_values)
pd.DataFrame(X_full, columns=FEATURES).to_csv(os.path.join(OUTDIR, "10_shap_X.csv"), index=False)
pd.Series(best).to_json(os.path.join(OUTDIR, "10_catboost_full_params.json"))
print("\n✅ Saved: 10_model_results.csv, 10_oof_*.npy, 10_shap_values.csv, 10_shap_matrix.npy, 10_shap_X.csv")
