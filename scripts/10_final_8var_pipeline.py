import catboost_sklearn_patch

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

SCAI_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR   = os.path.join(SCAI_DIR, "output")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FEATURES_8 = ["weight_kg", "gcs_total", "pao2", "lactate", "base_excess", "wbc", "ast", "inr"]
TARGET = "progressed"

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, accuracy_score, precision_score,
                              recall_score, brier_score_loss)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def evaluate_model(y_true, y_prob):
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.10, 0.90, 0.05):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    y_pred = (y_prob >= best_thresh).astype(int)
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "f1": best_f1, "threshold": best_thresh,
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(1-y_true, 1-y_pred, zero_division=0),
        "ppv": precision_score(y_true, y_pred, zero_division=0),
        "npv": precision_score(1-y_true, 1-y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
    }

print("=" * 60)
print("PART A: Primary 5-fold CV, 8-variable set, Optuna-tuned")
print("=" * 60)

df = pd.read_csv(os.path.join(OUTDIR, "07_features_AB.csv"))
X = df[FEATURES_8].copy()
y = df[TARGET].values

imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES_8)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
all_results = {}
oof_probs = {}

print("\n[XGBoost]")
oof_xgb = np.zeros(len(y))
for fold, (tr_idx, val_idx) in enumerate(cv.split(X_imp, y)):
    X_tr, X_val = X_imp.iloc[tr_idx], X_imp.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_est", 100, 500),
            "subsample": trial.suggest_float("ss", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("cbt", 0.5, 1.0),
            "scale_pos_weight": (y_tr==0).sum()/(y_tr==1).sum(),
            "random_state": RANDOM_STATE, "eval_metric": "auc", "verbosity": 0, "n_jobs": 1,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr)
        return roc_auc_score(y_val, m.predict_proba(X_val)[:,1])
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=25, show_progress_bar=False)
    best = study.best_params
    best["scale_pos_weight"] = (y_tr==0).sum()/(y_tr==1).sum()
    best["random_state"] = RANDOM_STATE; best["verbosity"] = 0; best["n_jobs"] = 1
    m = xgb.XGBClassifier(**best)
    cal = CalibratedClassifierCV(m, method="isotonic", cv=3)
    cal.fit(X_tr, y_tr)
    oof_xgb[val_idx] = cal.predict_proba(X_val)[:,1]
    print(f"  Fold {fold+1}: AUROC={roc_auc_score(y_val, oof_xgb[val_idx]):.3f}")
oof_probs["XGBoost"] = oof_xgb
all_results["XGBoost"] = evaluate_model(y, oof_xgb)

print("\n[LightGBM]")
oof_lgb = np.zeros(len(y))
for fold, (tr_idx, val_idx) in enumerate(cv.split(X_imp, y)):
    X_tr, X_val = X_imp.iloc[tr_idx], X_imp.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    def obj_lgb(trial):
        params = {
            "num_leaves": trial.suggest_int("nl", 20, 100),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n", 100, 500),
            "subsample": trial.suggest_float("ss", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("cbt", 0.5, 1.0),
            "is_unbalance": True, "random_state": RANDOM_STATE, "verbose": -1, "n_jobs": 1,
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr)
        return roc_auc_score(y_val, m.predict_proba(X_val)[:,1])
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(obj_lgb, n_trials=25, show_progress_bar=False)
    best = study.best_params
    best.update({"is_unbalance": True, "random_state": RANDOM_STATE, "verbose": -1, "n_jobs": 1})
    m = lgb.LGBMClassifier(**best)
    cal = CalibratedClassifierCV(m, method="isotonic", cv=3)
    cal.fit(X_tr, y_tr)
    oof_lgb[val_idx] = cal.predict_proba(X_val)[:,1]
    print(f"  Fold {fold+1}: AUROC={roc_auc_score(y_val, oof_lgb[val_idx]):.3f}")
oof_probs["LightGBM"] = oof_lgb
all_results["LightGBM"] = evaluate_model(y, oof_lgb)

print("\n[CatBoost]")
oof_cat = np.zeros(len(y))
for fold, (tr_idx, val_idx) in enumerate(cv.split(X_imp, y)):
    X_tr, X_val = X_imp.iloc[tr_idx], X_imp.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    def obj_cat(trial):
        params = {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 100, 500),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "scale_pos_weight": (y_tr==0).sum()/(y_tr==1).sum(),
            "random_seed": RANDOM_STATE, "verbose": 0, "thread_count": 1,
        }
        m = CatBoostClassifier(**params)
        m.fit(X_tr, y_tr)
        return roc_auc_score(y_val, m.predict_proba(X_val)[:,1])
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(obj_cat, n_trials=25, show_progress_bar=False)
    best = study.best_params
    best.update({"scale_pos_weight": (y_tr==0).sum()/(y_tr==1).sum(),
                 "random_seed": RANDOM_STATE, "verbose": 0, "thread_count": 1})
    m = CatBoostClassifier(**best)
    cal = CalibratedClassifierCV(m, method="isotonic", cv=3)
    cal.fit(X_tr, y_tr)
    oof_cat[val_idx] = cal.predict_proba(X_val)[:,1]
    print(f"  Fold {fold+1}: AUROC={roc_auc_score(y_val, oof_cat[val_idx]):.3f}")
oof_probs["CatBoost"] = oof_cat
all_results["CatBoost"] = evaluate_model(y, oof_cat)

print("\n[Logistic Regression]")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=FEATURES_8)
oof_lr = np.zeros(len(y))
for fold, (tr_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
    X_tr, X_val = X_scaled.iloc[tr_idx], X_scaled.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    m = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                           class_weight="balanced", solver="lbfgs")
    cal = CalibratedClassifierCV(m, method="isotonic", cv=3)
    cal.fit(X_tr, y_tr)
    oof_lr[val_idx] = cal.predict_proba(X_val)[:,1]
    print(f"  Fold {fold+1}: AUROC={roc_auc_score(y_val, oof_lr[val_idx]):.3f}")
oof_probs["LogisticRegression"] = oof_lr
all_results["LogisticRegression"] = evaluate_model(y, oof_lr)

results_df = pd.DataFrame(all_results).T
results_df = results_df[["auroc","auprc","accuracy","sensitivity","specificity",
                          "ppv","npv","f1","brier","threshold"]].round(3)
print("\nPrimary 5-fold CV results (8-variable set):")
print(results_df.to_string())
results_df.to_csv(os.path.join(OUTDIR, "10_model_results_8var.csv"))
for name, prob in oof_probs.items():
    np.save(os.path.join(OUTDIR, f"10_oof_{name.lower()}_8var.npy"), prob)
print(f"\n✅ Saved: 10_model_results_8var.csv")

print("\n" + "=" * 60)
print("PART B: SHAP analysis (CatBoost, full cohort fit)")
print("=" * 60)
import shap

def obj_cat_full(trial):
    params = {"depth": trial.suggest_int("depth",4,8),
              "learning_rate": trial.suggest_float("learning_rate",0.01,0.3,log=True),
              "iterations": trial.suggest_int("iterations",100,500),
              "l2_leaf_reg": trial.suggest_float("l2_leaf_reg",1,10),
              "scale_pos_weight": (y==0).sum()/(y==1).sum(),
              "random_seed": RANDOM_STATE, "verbose":0, "thread_count": 1}
    aucs=[]
    for tr_i, va_i in cv.split(X_imp, y):
        m = CatBoostClassifier(**params)
        m.fit(X_imp.iloc[tr_i], y[tr_i])
        aucs.append(roc_auc_score(y[va_i], m.predict_proba(X_imp.iloc[va_i])[:,1]))
    return np.mean(aucs)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(obj_cat_full, n_trials=25, show_progress_bar=False)
best = study.best_params
best.update({"scale_pos_weight": (y==0).sum()/(y==1).sum(), "random_seed": RANDOM_STATE, "verbose": 0, "thread_count": 1})
final_cat = CatBoostClassifier(**best)
final_cat.fit(X_imp, y)

explainer = shap.TreeExplainer(final_cat)
shap_values = explainer.shap_values(X_imp)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({"feature": FEATURES_8, "mean_abs_shap": mean_abs_shap}).sort_values(
    "mean_abs_shap", ascending=False)
print(shap_df.to_string(index=False))
shap_df.to_csv(os.path.join(OUTDIR, "10_shap_values_8var.csv"), index=False)
print(f"\n✅ Saved: 10_shap_values_8var.csv")

print("\n" + "=" * 60)
print("ALL DONE")
print("=" * 60)
