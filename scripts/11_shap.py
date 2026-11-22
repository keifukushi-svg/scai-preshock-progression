import os
import json
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from model_utils import make_model

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

df = pd.read_csv(os.path.join(OUTDIR, "07_landmark_dataset.csv"))
y, ho = df["progression"].values, df["holdout"].values == 1
feats = open(os.path.join(OUTDIR, "09_selected_features.txt")).read().split()
params = json.load(open(os.path.join(OUTDIR, "10_catboost_params.json")))
Xd = SimpleImputer(strategy="median").fit_transform(df.loc[~ho, feats].values.astype(float))
yd = y[~ho]
cat = make_model("CatBoost", params, yd).fit(Xd, yd)
sv = shap.TreeExplainer(cat).shap_values(Xd)
np.save(os.path.join(OUTDIR, "11_shap_values.npy"), sv)
imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
imp.to_csv(os.path.join(OUTDIR, "11_shap_importance.csv"), index=False)
print(imp.to_string(index=False))
