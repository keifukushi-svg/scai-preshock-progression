import catboost_sklearn_patch

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_TRIALS = 25


def make_model(name, params, y):
    spw = (y == 0).sum() / (y == 1).sum()
    if name == "XGBoost":
        return xgb.XGBClassifier(**params, scale_pos_weight=spw, random_state=RANDOM_STATE,
                                 eval_metric="auc", verbosity=0, n_jobs=1)
    if name == "LightGBM":
        return lgb.LGBMClassifier(**params, is_unbalance=True, random_state=RANDOM_STATE,
                                  verbose=-1, n_jobs=1)
    if name == "CatBoost":
        return CatBoostClassifier(**params, scale_pos_weight=spw, random_seed=RANDOM_STATE,
                                  verbose=0, thread_count=1, allow_writing_files=False)
    return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                              class_weight="balanced", solver="lbfgs")


def suggest(name, trial):
    if name == "XGBoost":
        return {"max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)}
    if name == "LightGBM":
        return {"num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)}
    return {"depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 100, 500),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10)}


def tune(name, X, y):
    if name == "LogisticRegression":
        return {}
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = suggest(name, trial)
        aucs = []
        for tr, va in inner.split(X, y):
            m = make_model(name, params, y[tr])
            m.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def fit_calibrated(name, X_tr, y_tr, params):
    cal = CalibratedClassifierCV(make_model(name, params, y_tr), method="isotonic", cv=5)
    cal.fit(X_tr, y_tr)
    return cal


def preprocess(name, X_tr, X_te):
    imp = SimpleImputer(strategy="median").fit(X_tr)
    X_tr, X_te = imp.transform(X_tr), imp.transform(X_te)
    if name == "LogisticRegression":
        sc = StandardScaler().fit(X_tr)
        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    return X_tr, X_te
