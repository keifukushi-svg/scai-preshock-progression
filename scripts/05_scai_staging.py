import pandas as pd
import os

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

vl  = pd.read_csv(os.path.join(OUTDIR, "02_baseline_vitals_labs.csv"))
trt = pd.read_csv(os.path.join(OUTDIR, "03_baseline_treatments.csv"))

df = vl.merge(trt[["stay_id","vaso_count_baseline","mcd_count_baseline"]],
              on="stay_id", how="left")
print(f"Merged: {len(df):,} rows")


def classify_scai_cswg(sbp, map_, lactate, alt, ph, mcd, vaso):
    def v(x):
        return x if pd.notna(x) else None

    sbp_v   = v(sbp)
    map_v   = v(map_)
    lact_v  = v(lactate)
    alt_v   = v(alt)
    ph_v    = v(ph)
    mcd_v   = int(mcd)  if pd.notna(mcd)  else 0
    vaso_v  = int(vaso) if pd.notna(vaso) else 0

    if ((sbp_v  is not None and sbp_v  <  60) or
        (map_v  is not None and map_v  <  50) or
        (ph_v   is not None and ph_v   <  7.2) or
        (lact_v is not None and lact_v > 10) or
        mcd_v >= 3 or vaso_v >= 3):
        return "E"

    hypo_sbp = (sbp_v is not None) and (60 <= sbp_v <= 90)
    hypo_map = (map_v is not None) and (50 <= map_v <= 65)
    hypotension = hypo_sbp or hypo_map

    high_lact = (lact_v is not None) and (5 <= lact_v <= 10)
    high_alt  = (alt_v  is not None) and (alt_v > 500)

    if (mcd_v == 2 or vaso_v == 2 or
        (hypotension and (high_lact or high_alt)) or
        (hypotension and mcd_v in [1, 2]) or
        (hypotension and vaso_v in [1, 2])):
        return "D"

    mod_lact = (lact_v is not None) and (2 <= lact_v < 5)
    mod_alt  = (alt_v  is not None) and (200 <= alt_v <= 500)

    if (mcd_v == 1 or vaso_v == 1 or
        (hypotension and (mod_lact or mod_alt))):
        return "C"

    b_hypo_clean = (hypotension and
                    (lact_v is None or lact_v < 2) and
                    (alt_v  is None or alt_v  < 200) and
                    (ph_v   is None or ph_v   >= 7.2))
    b_lact = mod_lact
    b_alt  = mod_alt

    if b_hypo_clean or b_lact or b_alt:
        return "B"

    needs_eval = ((lact_v is not None and lact_v >= 5) or
                  (alt_v  is not None and alt_v  > 500))
    if needs_eval:
        return "A (needs evaluation)"

    return "A"


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
