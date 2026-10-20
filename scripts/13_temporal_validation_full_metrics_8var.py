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

FEATURES_8 = ["lactate", "gcs_total", "pao2", "wbc", "ast", "weight_kg", "inr", "base_excess"]
TARGET = "progressed"

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              f1_score, accuracy_score, precision_score, recall_score)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv(os.path.join(OUTDIR, "07_features_AB.csv"))
cohort = pd.read_csv(os.path.join(OUTDIR, "01_cohort.csv"), usecols=["stay_id", "anchor_year_group"])
df_merged = df.merge(cohort, on="stay_id", how="left")

train_groups = ["2008 - 2010", "2011 - 2013", "2014 - 2016"]
val_groups = ["2017 - 2019"]
df_train = df_merged[df_merged["anchor_year_group"].isin(train_groups)].copy()
df_val   = df_merged[df_merged["anchor_year_group"].isin(val_groups)].copy()
print(f"Derivation : n={len(df_train):,} (progression {df_train[TARGET].mean()*100:.1f}%)")
print(f"Temporal val: n={len(df_val):,} (progression {df_val[TARGET].mean()*100:.1f}%)")

X_train = df_train[FEATURES_8].copy(); y_train = df_train[TARGET].values
X_val_t = df_val[FEATURES_8].copy();   y_val_t = df_val[TARGET].values

imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURES_8)
X_val_imp   = pd.DataFrame(imputer.transform(X_val_t), columns=FEATURES_8)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_val_sc   = scaler.transform(X_val_imp)

spw = (y_train==0).sum()/(y_train==1).sum()
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def evaluate_full(y_true, y_prob):
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.10, 0.90, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    y_pred = (y_prob >= best_thresh).astype(int)
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(1-y_true, 1-y_pred, zero_division=0),
        "ppv": precision_score(y_true, y_pred, zero_division=0),
        "npv": precision_score(1-y_true, 1-y_pred, zero_division=0),
        "f1": best_f1, "threshold": best_thresh,
        "brier": brier_score_loss(y_true, y_prob),
    }

results = {}
probs = {}

print("\n[XGBoost]")
def obj_xgb(trial):
    params = {"max_depth": trial.suggest_int("max_depth",3,8),
              "learning_rate": trial.suggest_float("lr",0.01,0.3,log=True),
              "n_estimators": trial.suggest_int("n_est",100,500),
              "subsample": trial.suggest_float("ss",0.5,1.0),
              "colsample_bytree": trial.suggest_float("cbt",0.5,1.0),
              "scale_pos_weight": spw, "random_state": RANDOM_STATE,
              "eval_metric":"auc","verbosity":0, "n_jobs":1}
    aucs=[]
    for tr_i, va_i in inner_cv.split(X_train_imp, y_train):
        m = xgb.XGBClassifier(**params)
        m.fit(X_train_imp.iloc[tr_i], y_train[tr_i])
        aucs.append(roc_auc_score(y_train[va_i], m.predict_proba(X_train_imp.iloc[va_i])[:,1]))
    return np.mean(aucs)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(obj_xgb, n_trials=25, show_progress_bar=False)
best = study.best_params; best["scale_pos_weight"]=spw
best["random_state"]=RANDOM_STATE; best["verbosity"]=0; best["n_jobs"]=1
m = xgb.XGBClassifier(**best)
cal = CalibratedClassifierCV(m, method="isotonic", cv=5)
cal.fit(X_train_imp.values, y_train)
prob = cal.predict_proba(X_val_imp.values)[:,1]
probs["XGBoost"] = prob
results["XGBoost"] = evaluate_full(y_val_t, prob)
print(f"  AUROC={results['XGBoost']['auroc']:.3f}")

print("\n[LightGBM]")
def obj_lgb(trial):
    params = {"num_leaves": trial.suggest_int("nl",20,100),
              "learning_rate": trial.suggest_float("lr",0.01,0.3,log=True),
              "n_estimators": trial.suggest_int("n",100,500),
              "subsample": trial.suggest_float("ss",0.5,1.0),
              "colsample_bytree": trial.suggest_float("cbt",0.5,1.0),
              "is_unbalance": True, "random_state": RANDOM_STATE, "verbose": -1, "n_jobs": 1}
    aucs=[]
    for tr_i, va_i in inner_cv.split(X_train_imp, y_train):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_train_imp.iloc[tr_i], y_train[tr_i])
        aucs.append(roc_auc_score(y_train[va_i], m.predict_proba(X_train_imp.iloc[va_i])[:,1]))
    return np.mean(aucs)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(obj_lgb, n_trials=25, show_progress_bar=False)
best = study.best_params
best.update({"is_unbalance": True, "random_state": RANDOM_STATE, "verbose": -1, "n_jobs": 1})
m = lgb.LGBMClassifier(**best)
cal = CalibratedClassifierCV(m, method="isotonic", cv=5)
cal.fit(X_train_imp.values, y_train)
prob = cal.predict_proba(X_val_imp.values)[:,1]
probs["LightGBM"] = prob
results["LightGBM"] = evaluate_full(y_val_t, prob)
print(f"  AUROC={results['LightGBM']['auroc']:.3f}")

print("\n[CatBoost]")
def obj_cat(trial):
    params = {"depth": trial.suggest_int("depth",4,8),
              "learning_rate": trial.suggest_float("learning_rate",0.01,0.3,log=True),
              "iterations": trial.suggest_int("iterations",100,500),
              "l2_leaf_reg": trial.suggest_float("l2_leaf_reg",1,10),
              "scale_pos_weight": spw, "random_seed": RANDOM_STATE, "verbose":0, "thread_count": 1}
    aucs=[]
    for tr_i, va_i in inner_cv.split(X_train_imp, y_train):
        m = CatBoostClassifier(**params)
        m.fit(X_train_imp.iloc[tr_i], y_train[tr_i])
        aucs.append(roc_auc_score(y_train[va_i], m.predict_proba(X_train_imp.iloc[va_i])[:,1]))
    return np.mean(aucs)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(obj_cat, n_trials=25, show_progress_bar=False)
best = study.best_params
best.update({"scale_pos_weight": spw, "random_seed": RANDOM_STATE, "verbose": 0, "thread_count": 1})
m = CatBoostClassifier(**best)
cal = CalibratedClassifierCV(m, method="isotonic", cv=5)
cal.fit(X_train_imp.values, y_train)
prob = cal.predict_proba(X_val_imp.values)[:,1]
probs["CatBoost"] = prob
results["CatBoost"] = evaluate_full(y_val_t, prob)
print(f"  AUROC={results['CatBoost']['auroc']:.3f}")

print("\n[Logistic Regression]")
m = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                       class_weight="balanced", solver="lbfgs")
cal = CalibratedClassifierCV(m, method="isotonic", cv=5)
cal.fit(X_train_sc, y_train)
prob = cal.predict_proba(X_val_sc)[:,1]
probs["LogisticRegression"] = prob
results["LogisticRegression"] = evaluate_full(y_val_t, prob)
print(f"  AUROC={results['LogisticRegression']['auroc']:.3f}")

results_df = pd.DataFrame(results).T
results_df = results_df[["auroc","auprc","accuracy","sensitivity","specificity",
                          "ppv","npv","f1","brier","threshold"]].round(3)
print("\nTemporal validation full metrics (8-variable set):")
print(results_df.to_string())
results_df.to_csv(os.path.join(OUTDIR, "13_temporal_validation_full_metrics_8var.csv"))

for name, prob in probs.items():
    np.save(os.path.join(OUTDIR, f"13_temporal_prob_{name.lower()}_8var.npy"), prob)

print(f"\n✅ Saved: 13_temporal_validation_full_metrics_8var.csv, 13_temporal_prob_*_8var.npy")
