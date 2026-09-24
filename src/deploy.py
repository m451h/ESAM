"""The input contract for triage inference.

`src/features.py` describes the model's 50 features in terms of the raw export's
column names -- Persian headers, a misspelled `HDL cholestrol`, `WBC.1`. That is
fine for analysis and useless as an interface.

This module is the interface. It declares the **44 values a caller supplies**
(the other 6 features are ratios computed here), each with a stable snake_case
alias, a unit, and the plausible range enforced during training. `vectorize()`
turns caller input into the exact feature matrix the fitted model expects.

Units are not documented in the source export; they were inferred from the
observed distributions of the 36,885 visits and are stated here so a caller can
check them. A silent unit mismatch is the most likely way to get confidently
wrong answers out of this model, so `vectorize()` range-checks every input and
reports what it rejected rather than failing quietly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .features import LADDER, feature_columns


# --------------------------------------------------------------- exclusions
# Features present in the analysis panel but withheld from the deployed model.
# Each entry is a safety decision, not a performance one; the AUROC cost of each
# is recorded so the trade is auditable.
EXCLUDED_FEATURES: dict[str, str] = {
    "VT D3":
        "Runs backwards and implausibly steeply: among DEXA recipients at their "
        "FIRST visit, low bone mass is 0.5% where VT D3 <= 20 and 18.6% where it "
        "is > 40 -- a 37-fold gradient in the direction opposite to physiology. "
        "It is not a site effect (it holds within a single employer: 3.3% vs "
        "~16%) and not treatment feedback (first visits alone show it). The "
        "column's distribution is also not that of a 25-OH vitamin D assay: 638 "
        "distinct values bunched at 34-43 with a thin left tail, in a population "
        "where deficiency should dominate. Whatever it encodes, a deployed model "
        "using it would tell genuinely deficient patients to skip densitometry. "
        "Cost of removal, leave-one-employer-out fold-wise AUROC: low bone mass "
        "0.723 -> 0.715, prostate 0.775 -> 0.766 -- both far inside the "
        "fold-to-fold SD (~0.08). Suspect the column, not the finding.",
}


@dataclass(frozen=True)
class Input:
    alias: str          # what the caller supplies
    column: str         # internal column name in visits.parquet
    unit: str
    block: str
    description: str

    @property
    def range(self) -> tuple[float, float] | None:
        return C.PLAUSIBLE_RANGE.get(self.column)

    @property
    def used_by_model(self) -> bool:
        return self.column not in EXCLUDED_FEATURES

    @property
    def exclusion_reason(self) -> str | None:
        return EXCLUDED_FEATURES.get(self.column)


def _i(alias, column, unit, block, description) -> Input:
    return Input(alias, column, unit, block, description)


# --------------------------------------------------------------- the contract
INPUT_SPEC: list[Input] = [
    # -- demographics ---------------------------------------------------------
    _i("age", "age", "years", "demographics", "Age at visit (16-95)"),
    _i("sex", "male", "'male' | 'female'", "demographics",
       "Biological sex; also accepts 1/0 for male"),

    # -- vitals / anthropometry ----------------------------------------------
    _i("bmi", C.COL_BMI, "kg/m^2", "vitals", "Body mass index"),
    _i("systolic_bp", C.COL_SBP, "mmHg", "vitals", "Systolic blood pressure"),
    _i("diastolic_bp", C.COL_DBP, "mmHg", "vitals", "Diastolic blood pressure"),

    # -- chemistry ------------------------------------------------------------
    _i("fasting_blood_sugar", "Fasting Blood Sugar", "mg/dL", "chemistry", "Fasting glucose"),
    _i("urea", "Blood Urea", "mg/dL", "chemistry", "Blood urea"),
    _i("creatinine", "Creatinine", "mg/dL", "chemistry", "Serum creatinine"),
    _i("uric_acid", "Uric Acid", "mg/dL", "chemistry", "Serum uric acid"),
    _i("cholesterol", "Cholesterol", "mg/dL", "chemistry", "Total cholesterol"),
    _i("triglycerides", "Triglycerides", "mg/dL", "chemistry", "Triglycerides"),
    _i("hdl", "HDL cholestrol", "mg/dL", "chemistry", "HDL cholesterol"),
    _i("ldl", "LDL cholestrol", "mg/dL", "chemistry", "LDL cholesterol"),
    _i("calcium", "Calcium", "mg/dL", "chemistry", "Serum calcium"),
    _i("phosphorus", "Phosphorus", "mg/dL", "chemistry", "Serum phosphorus"),
    _i("total_bilirubin", "Total Bilirubin", "mg/dL", "chemistry", "Total bilirubin"),
    _i("direct_bilirubin", "Direct bilirubin", "mg/dL", "chemistry", "Direct bilirubin"),
    _i("ast", "SGOT (AST)", "U/L", "chemistry", "Aspartate aminotransferase"),
    _i("alt", "SGPT (ALT)", "U/L", "chemistry", "Alanine aminotransferase"),
    _i("alkaline_phosphatase", "Alkaline phosphatase", "U/L", "chemistry", "ALP"),
    _i("esr", "ESR", "mm/hr", "chemistry", "Erythrocyte sedimentation rate"),
    _i("tsh", "TSH", "mIU/L", "chemistry", "Thyroid stimulating hormone"),
    _i("vitamin_d", "VT D3", "ng/mL", "chemistry", "25-OH vitamin D"),
    _i("ferritin", "FER", "ng/mL", "chemistry", "Serum ferritin"),
    _i("iron", "IRON", "ug/dL", "chemistry", "Serum iron"),
    _i("tibc", "TIBC", "ug/dL", "chemistry", "Total iron binding capacity"),
    _i("hba1c", "A1C", "%", "chemistry", "Glycated haemoglobin"),

    # -- haematology ----------------------------------------------------------
    _i("hb", "HB", "g/dL", "haematology", "Haemoglobin"),
    _i("hct", "HCT", "%", "haematology", "Haematocrit"),
    _i("mcv", "MCV", "fL", "haematology", "Mean corpuscular volume"),
    _i("mch", "MCH", "pg", "haematology", "Mean corpuscular haemoglobin"),
    _i("mchc", "MCHC", "g/dL", "haematology", "MCH concentration"),
    _i("plt", "PLT", "x10^9/L", "haematology", "Platelet count"),
    _i("rdw_cv", "RDW.CV", "%", "haematology", "Red cell distribution width (CV)"),
    _i("rdw_sd", "RDW.SD", "fL", "haematology", "Red cell distribution width (SD)"),
    _i("wbc", "WBC.1", "x10^9/L", "haematology", "White cell count"),
    _i("rbc", "RBC.1", "x10^12/L", "haematology", "Red cell count"),
    _i("neutrophils_pct", "N", "% of WBC", "haematology", "Neutrophils"),
    _i("lymphocytes_pct", "L", "% of WBC", "haematology", "Lymphocytes"),
    _i("monocytes_pct", "M", "% of WBC", "haematology", "Monocytes"),
    _i("eosinophils_pct", "EO", "% of WBC", "haematology", "Eosinophils"),
    _i("basophils_pct", "BAS", "% of WBC", "haematology", "Basophils"),
    _i("mpv", "MPV", "fL", "haematology", "Mean platelet volume"),
    _i("pdw", "PDW", "fL", "haematology", "Platelet distribution width"),
]

BY_ALIAS = {s.alias: s for s in INPUT_SPEC}
BY_COLUMN = {s.column: s for s in INPUT_SPEC}

# The model's feature order: the analysis panel minus the withheld columns.
ANALYSIS_FEATURES = feature_columns(LADDER["D_full_core"])
FEATURE_ORDER = [c for c in ANALYSIS_FEATURES if c not in EXCLUDED_FEATURES]

DERIVED_DOC = {
    "nlr": "neutrophils_pct / lymphocytes_pct",
    "ast_alt": "ast / alt",
    "tg_hdl": "triglycerides / hdl",
    "non_hdl": "cholesterol - hdl",
    "transferrin_sat": "100 * iron / tibc",
    "pulse_pressure": "systolic_bp - diastolic_bp",
}


# ------------------------------------------------------------------ vectorise
def _to_frame(records) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, dict):
        return pd.DataFrame([records])
    return pd.DataFrame(list(records))


def _normalise_sex(s: pd.Series) -> pd.Series:
    """Accept 'male'/'female'/'M'/'F'/1/0 -> 1/0, anything else -> NaN."""
    txt = s.astype(str).str.strip().str.lower()
    male = txt.isin(["male", "m", "1", "1.0", "true", C.SEX_MALE])
    female = txt.isin(["female", "f", "0", "0.0", "false", C.SEX_FEMALE])
    return pd.Series(np.where(male, 1.0, np.where(female, 0.0, np.nan)), index=s.index)


def vectorize(records) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Caller input -> (X, report).

    `records` may be a dict, a list of dicts, or a DataFrame. Keys may be the
    snake_case aliases above or the internal column names; unknown keys are
    ignored and listed in the report.

    Returns the 50-column feature matrix in training order, and a per-row report
    of what was missing, out of range, or unrecognised. Out-of-range values are
    set to NaN, exactly as `clean.py` does during training -- the model imputes
    them rather than trusting an impossible number.
    """
    df = _to_frame(records)
    n = len(df)
    out = pd.DataFrame(index=range(n))
    rejected = [[] for _ in range(n)]
    missing = [[] for _ in range(n)]

    supplied_keys = set(df.columns)
    known = {s.alias for s in INPUT_SPEC} | {s.column for s in INPUT_SPEC}
    unknown = sorted(supplied_keys - known)
    ignored = sorted(k for k in supplied_keys & known
                     if not (BY_ALIAS.get(k) or BY_COLUMN[k]).used_by_model)

    df = df.reset_index(drop=True)

    for spec in INPUT_SPEC:
        key = spec.alias if spec.alias in df.columns else (
            spec.column if spec.column in df.columns else None)
        if key is None:
            col = pd.Series(np.nan, index=range(n), dtype=float)
        elif spec.alias == "sex":
            col = _normalise_sex(df[key])
        else:
            col = pd.to_numeric(df[key], errors="coerce")

        rng = spec.range
        if rng is not None:
            bad = col.notna() & ~col.between(*rng)
            for i in np.where(bad.to_numpy())[0]:
                rejected[i].append(f"{spec.alias}={df[key].iloc[i]} outside "
                                   f"{rng[0]}-{rng[1]} {spec.unit}")
            col = col.mask(bad)

        if spec.used_by_model:
            for i in np.where(col.isna().to_numpy())[0]:
                missing[i].append(spec.alias)

        out[spec.column] = col.astype(float)

    # derived ratios -- identical formulas to clean.py
    out["nlr"] = out["N"] / out["L"].replace(0, np.nan)
    out["ast_alt"] = out["SGOT (AST)"] / out["SGPT (ALT)"].replace(0, np.nan)
    out["tg_hdl"] = out["Triglycerides"] / out["HDL cholestrol"].replace(0, np.nan)
    out["non_hdl"] = out["Cholesterol"] - out["HDL cholestrol"]
    out["transferrin_sat"] = 100 * out["IRON"] / out["TIBC"].replace(0, np.nan)
    out["pulse_pressure"] = out[C.COL_SBP] - out[C.COL_DBP]

    X = out.reindex(columns=FEATURE_ORDER)

    n_used = sum(1 for s in INPUT_SPEC if s.used_by_model)
    report = pd.DataFrame({
        "n_missing": [len(m) for m in missing],
        "completeness": [1 - len(m) / n_used for m in missing],
        "missing": [", ".join(m) for m in missing],
        "rejected": [" | ".join(r) for r in rejected],
        "unknown_keys": [", ".join(unknown)] * n,
        "ignored_keys": [", ".join(ignored)] * n,
    })
    return X, report


def input_table() -> pd.DataFrame:
    """The contract as a table -- what a caller must send."""
    rows = []
    for s in INPUT_SPEC:
        lo, hi = s.range if s.range else (np.nan, np.nan)
        rows.append({"alias": s.alias, "unit": s.unit, "block": s.block,
                     "accepted_min": lo, "accepted_max": hi,
                     "used_by_model": s.used_by_model,
                     "description": s.description, "source_column": s.column})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t = input_table()
    n_used = int(t.used_by_model.sum())
    print(f"{len(INPUT_SPEC)} accepted inputs ({n_used} used, "
          f"{len(INPUT_SPEC)-n_used} withheld) + {len(DERIVED_DOC)} derived "
          f"= {len(FEATURE_ORDER)} model features\n")
    print(t.to_string(index=False))
    for col, why in EXCLUDED_FEATURES.items():
        alias = BY_COLUMN[col].alias if col in BY_COLUMN else col
        print(f"\nWITHHELD -- {alias} ({col}):\n  {why}")
    out = C.TABLES / "model_inputs.csv"
    t.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nwrote {out}")
