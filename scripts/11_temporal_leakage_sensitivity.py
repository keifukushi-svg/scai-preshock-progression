import pandas as pd
import numpy as np
import os
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

SCAI_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR   = os.path.join(SCAI_DIR, "output")
RANDOM_STATE = 42

FEATURES = [l.strip() for l in open(os.path.join(OUTDIR, "09_selected_features.txt")).read().split()[1:]]
TARGET = "progressed"

df = pd.read_csv(os.path.join(OUTDIR, "07_features_AB.csv"))
timing = pd.read_csv(os.path.join(OUTDIR, "08_progression_timing.csv"))
df = df.merge(timing[["stay_id", "h_to_first_CDE"]], on="stay_id", how="left")
y = df[TARGET].values

X_orig = df[FEATURES].copy()
X_mask = df[FEATURES].copy()
records = []
for var in FEATURES:
    hcol = f"{var}_h"
    if hcol not in df.columns:
        print(f"  {var}: measured before ICU admission or time-invariant, not masked")
        continue
    mask = df["h_to_first_CDE"].notna() & (df[hcol] > df["h_to_first_CDE"])
    X_mask.loc[mask, var] = np.nan
    records += [{"stay_id": s, "variable": var, "h_to_first_CDE": t, "measurement_h": m}
                for s, t, m in zip(df.loc[mask, "stay_id"], df.loc[mask, "h_to_first_CDE"], df.loc[mask, hcol])]
    print(f"  {var}: masked {int(mask.sum())} values measured after first attainment of C/D/E")
pd.DataFrame(records).to_csv(os.path.join(OUTDIR, "11_masked_values.csv"), index=False)


def oof_lr(X):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    p = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
        sc = StandardScaler().fit(imp.transform(X.iloc[tr]))
        X_tr = sc.transform(imp.transform(X.iloc[tr]))
        X_va = sc.transform(imp.transform(X.iloc[va]))
        lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                class_weight="balanced", solver="lbfgs")
        cal = CalibratedClassifierCV(lr, method="isotonic", cv=5).fit(X_tr, y[tr])
        p[va] = cal.predict_proba(X_va)[:, 1]
    return p


auc_orig = roc_auc_score(y, oof_lr(X_orig))
auc_mask = roc_auc_score(y, oof_lr(X_mask))
print(f"\nLogistic regression AUROC, original values           : {auc_orig:.4f}")
print(f"Logistic regression AUROC, post-progression masked   : {auc_mask:.4f}")
print(f"Difference                                           : {auc_mask - auc_orig:+.4f}")

pd.DataFrame({"metric": ["auroc_original", "auroc_masked", "auroc_difference", "n_values_masked"],
              "value": [auc_orig, auc_mask, auc_mask - auc_orig, len(records)]}
             ).to_csv(os.path.join(OUTDIR, "11_temporal_leakage_sensitivity.csv"), index=False)
print("\n✅ Saved: 11_temporal_leakage_sensitivity.csv, 11_masked_values.csv")
