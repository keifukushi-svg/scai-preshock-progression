import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
warnings.filterwarnings("ignore")

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
RANDOM_STATE = 42
NON_FEATURES = {"progression", "holdout", "subject_id", "inhospital_death"}
np.random.seed(RANDOM_STATE)

df = pd.read_csv(os.path.join(OUTDIR, "07_landmark_dataset.csv"), index_col="stay_id")
df = df[df["holdout"] == 0]
# Candidates: the most recent value of each time-varying variable within 0-6 h, with the static variables and medications
cand = [c for c in df.columns if c not in NON_FEATURES and df[c].notna().any()
        and c != "fluid_input_6h" and (c.endswith("_last") or not re.search(r"_(first|last|min|max)$", c))]
X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[cand]), columns=cand)
y = df["progression"].values
print(f"Development cohort 2008-2016: n={len(y):,}, progression {y.sum():,} ({y.mean() * 100:.1f}%), "
      f"candidates {len(cand)}")

rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, max_depth=5)
bor = BorutaPy(rf, n_estimators="auto", verbose=0, random_state=RANDOM_STATE, max_iter=50)
bor.fit(X.values, y)
kept = [c for c, s, w in zip(cand, bor.support_, bor.support_weak_) if s or w]
pd.DataFrame({"feature": cand, "rank": bor.ranking_,
              "status": ["Confirmed" if s else "Tentative" if w else "Rejected"
                         for s, w in zip(bor.support_, bor.support_weak_)]}
             ).sort_values("rank").to_csv(os.path.join(OUTDIR, "09_boruta.csv"), index=False)
print(f"Boruta kept {len(kept)}: {kept}")

Xs, vif_rows = X[kept].copy(), []
while len(Xs.columns) > 1:
    Xc = add_constant(Xs, has_constant="add")
    vifs = {c: variance_inflation_factor(Xc.values, i) for i, c in enumerate(Xc.columns) if c != "const"}
    vif_rows += [{"step": len(kept) - len(Xs.columns) + 1, "feature": c, "vif": v} for c, v in vifs.items()]
    worst = max(vifs, key=vifs.get)
    if vifs[worst] <= 10:
        break
    print(f"  VIF removes {worst} ({vifs[worst]:.1f})")
    Xs = Xs.drop(columns=[worst])
pd.DataFrame(vif_rows).to_csv(os.path.join(OUTDIR, "09_vif.csv"), index=False)
final = list(Xs.columns)
print(f"Final {len(final)}: {final}")
open(os.path.join(OUTDIR, "09_selected_features.txt"), "w").write("\n".join(final) + "\n")
