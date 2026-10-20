import pandas as pd

VASOACTIVE_ITEMIDS = {
    221906: "norepinephrine",
    221289: "epinephrine",
    229617: "epinephrine",
    222315: "vasopressin",
    221749: "phenylephrine",
    221662: "dopamine",
    221653: "dobutamine",
    221986: "milrinone",
}
VASOACTIVE_CLASSES = sorted(set(VASOACTIVE_ITEMIDS.values()))

MCS_ITEMIDS = {224272: "iabp", 228169: "impella", 229529: "ecmo", 229530: "ecmo"}
MCS_CLASSES = sorted(set(MCS_ITEMIDS.values()))

PLAUSIBLE_RANGE = {
    "hr": (20, 250),
    "sbp": (30, 300),
    "map": (20, 250),
    "rr": (2, 80),
    "temp_f": (77, 113),
    "spo2": (50, 100),
    "ph": (6.5, 8.0),
    "pao2": (10, 800),
    "paco2": (5, 200),
    "lactate": (0.1, 30),
    "base_excess": (-40, 40),
}

CANDIDATE_PREDICTORS = [
    "age", "gender_male", "weight_kg", "height_cm", "bmi",
    "hr", "sbp", "map", "rr", "temp_f", "spo2",
    "gcs_total", "ambulatory_aid",
    "ph", "pao2", "paco2", "lactate", "base_excess",
    "wbc", "plt", "hgb", "tbil", "albumin", "sodium", "potassium",
    "creatinine", "calcium", "ast", "alt", "bun", "pt", "aptt", "inr",
    "htn", "dm", "hyperlipidemia", "copd", "pneumonia", "ckd", "af",
    "ami_flag", "ahf_flag",
    "acei_arb_baseline", "betablocker_baseline",
    "furosemide_baseline", "spironolactone_baseline",
]


def plausible(var, values):
    base = var[:-6] if var.endswith("_chart") else var
    if base not in PLAUSIBLE_RANGE:
        return pd.Series(True, index=values.index)
    lo, hi = PLAUSIBLE_RANGE[base]
    return values.between(lo, hi)


def classify_scai_cswg(sbp, map_, lactate, alt, ph, mcd, vaso):
    def v(x):
        return x if pd.notna(x) else None

    sbp_v = v(sbp)
    map_v = v(map_)
    lact_v = v(lactate)
    alt_v = v(alt)
    ph_v = v(ph)
    mcd_v = int(mcd) if pd.notna(mcd) else 0
    vaso_v = int(vaso) if pd.notna(vaso) else 0

    if ((sbp_v is not None and sbp_v < 60) or
            (map_v is not None and map_v < 50) or
            (ph_v is not None and ph_v < 7.2) or
            (lact_v is not None and lact_v > 10) or
            mcd_v >= 3 or vaso_v >= 3):
        return "E"

    hypo_sbp = (sbp_v is not None) and (60 <= sbp_v <= 90)
    hypo_map = (map_v is not None) and (50 <= map_v <= 65)
    hypotension = hypo_sbp or hypo_map

    high_lact = (lact_v is not None) and (5 <= lact_v <= 10)
    high_alt = (alt_v is not None) and (alt_v > 500)

    if (mcd_v == 2 or vaso_v == 2 or
            (hypotension and (high_lact or high_alt)) or
            (hypotension and mcd_v in [1, 2]) or
            (hypotension and vaso_v in [1, 2])):
        return "D"

    mod_lact = (lact_v is not None) and (2 <= lact_v < 5)
    mod_alt = (alt_v is not None) and (200 <= alt_v <= 500)

    if (mcd_v == 1 or vaso_v == 1 or
            (hypotension and (mod_lact or mod_alt))):
        return "C"

    b_hypo_clean = (hypotension and
                    (lact_v is None or lact_v < 2) and
                    (alt_v is None or alt_v < 200) and
                    (ph_v is None or ph_v >= 7.2))
    if b_hypo_clean or mod_lact or mod_alt:
        return "B"

    if ((lact_v is not None and lact_v >= 5) or
            (alt_v is not None and alt_v > 500)):
        return "A (needs evaluation)"

    return "A"
