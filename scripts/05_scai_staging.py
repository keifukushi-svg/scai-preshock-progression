import pandas as pd
import os
from scai_criteria import classify_scai_cswg

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

vl  = pd.read_csv(os.path.join(OUTDIR, "02_baseline_vitals_labs.csv"))
trt = pd.read_csv(os.path.join(OUTDIR, "03_baseline_treatments.csv"))

df = vl.merge(trt[["stay_id","vaso_count_baseline","mcd_count_baseline"]],
              on="stay_id", how="left")
print(f"Merged: {len(df):,} rows")


print("Classifying SCAI-CSWG stages...")
df["scai_baseline"] = df.apply(
    lambda r: classify_scai_cswg(
        r.get("sbp"), r.get("map"),
        r.get("lactate"), r.get("alt"), r.get("ph"),
        r.get("mcd_count_baseline", 0), r.get("vaso_count_baseline", 0)
    ),
    axis=1
)

print("\n=== SCAI-CSWG baseline distribution ===")
dist = df["scai_baseline"].value_counts().sort_index()
print(dist)
print(f"\nTotal: {len(df):,}")

ORDER = ["A", "A (needs evaluation)", "B", "C", "D", "E"]
for s in ORDER:
    n = (df["scai_baseline"] == s).sum()
    pct = n / len(df) * 100
    print(f"  {s:25s}: {n:5,} ({pct:.1f}%)")

ab_mask = df["scai_baseline"].isin(["A","B"])
print(f"\nA/B (before exclusion)           : {ab_mask.sum():,}")
print(f"C/D/E (before exclusion)         : {df['scai_baseline'].isin(['C','D','E']).sum():,}")
print(f"A (needs eval) (before exclusion): {(df['scai_baseline']=='A (needs evaluation)').sum():,}")

out_path = os.path.join(OUTDIR, "05_scai_baseline.csv")
df[["hadm_id","stay_id","sbp","map","lactate","alt","ph",
    "vaso_count_baseline","mcd_count_baseline","scai_baseline"]].to_csv(
    out_path, index=False)
print(f"\n✅ Saved: {out_path}")
