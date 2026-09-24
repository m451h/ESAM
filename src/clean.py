"""Stage 1: raw Excel export -> analysis-ready parquet.

Run:  python -m src.clean

Steps
-----
1. Read the single-sheet export (36,885 x 592) as text.
2. Replace "test not performed" sentinel strings with true missing values.
3. Parse Jalali dates; derive visit year and age.
4. Coerce the core panel to numeric and null out unit-entry errors.
5. Coalesce the split DEXA columns back into one value + WHO category per site.
6. Construct a person identifier and a protocol (employer) identifier.
7. Emit data/processed/visits.parquet plus a data-quality report.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import config as C


# ------------------------------------------------------------------ utilities
def _jalali_year(value) -> float:
    """Leading year component of a Jalali date string like '1402/09/28'."""
    s = str(value)
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return np.nan


def _jalali_ordinal(value) -> float:
    """Approximate days-since-epoch for a Jalali date, for within-year ordering."""
    parts = str(value).split("/")
    if len(parts) != 3 or not all(p.strip().isdigit() for p in parts):
        return np.nan
    y, m, d = (int(p) for p in parts)
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return np.nan
    return y * 365.25 + (m - 1) * 30.44 + d


def strip_sentinels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Blank out 'not registered' style placeholders. Returns (df, per-column count)."""
    pattern = re.compile("|".join(C.SENTINEL_PATTERNS), re.IGNORECASE)
    counts = {}
    out = {}
    for col in df.columns:
        s = df[col]
        # pandas >=3 gives text columns dtype 'str' rather than 'object', so test
        # for "not numeric" instead of testing for object specifically.
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            counts[col] = 0
            out[col] = s
        else:
            hit = s.notna() & s.astype(str).str.contains(pattern, na=False)
            counts[col] = int(hit.sum())
            out[col] = s.mask(hit)
    return pd.DataFrame(out, index=df.index), pd.Series(counts)


def coerce_numeric(s: pd.Series) -> pd.Series:
    """Numeric coercion tolerant of stray unit text and Persian/Arabic digits."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype(str)
    t = t.str.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
    t = t.str.replace(r"[^0-9.\-+eE]", "", regex=True).replace({"": None, "-": None, ".": None})
    return pd.to_numeric(t, errors="coerce")


def coalesce_dexa(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rebuild one value + WHO category per (measure, site) from the split columns.

    The export writes each densitometry measurement into the column matching its
    clinical category, so `BMD L.Hip , osteopenia` is non-null only for people the
    reporting radiologist classed as osteopenic. Recovering both the number and
    the label means we can use the value as a continuous target and the label as
    the operator's own call.
    """
    out, report = {}, {}
    for measure, (lo, hi) in C.DEXA_RANGE.items():
        for site in C.DEXA_SITES:
            value = pd.Series(np.nan, index=df.index, dtype=float)
            category = pd.Series(pd.NA, index=df.index, dtype=object)
            found = 0
            for fa, en in C.DEXA_CATEGORIES.items():
                col = f"{measure} {site} , {fa}"
                if col not in df.columns:
                    continue
                found += 1
                v = coerce_numeric(df[col]).where(lambda x: x.between(lo, hi))
                take = v.notna() & value.isna()
                value = value.mask(take, v)
                category = category.mask(take, en)
            if not found:
                continue
            key = f"dexa_{measure.replace('.', '').lower()}_{site.replace(' ', '_').replace('.', '').lower()}"
            out[key] = value
            out[key + "_cat"] = category
            report[key] = {"n": int(value.notna().sum()),
                           "by_category": category.value_counts().to_dict()}
    return pd.DataFrame(out, index=df.index), report


# ----------------------------------------------------------------------- main
def build() -> pd.DataFrame:
    print(f"reading {C.RAW_XLSX.name} ...")
    raw = pd.read_excel(C.RAW_XLSX, sheet_name=C.RAW_SHEET, dtype=str)
    print(f"  raw shape {raw.shape}")

    qc: dict = {"raw_shape": list(raw.shape)}

    # --- 1. sentinels -------------------------------------------------------
    naive_missing = raw.isna().mean()
    df, sentinel_counts = strip_sentinels(raw)
    true_missing = df.isna().mean()
    qc["sentinel_total_cells"] = int(sentinel_counts.sum())
    qc["sentinel_top_columns"] = (
        sentinel_counts.sort_values(ascending=False).head(25)
        .astype(int).to_dict()
    )
    qc["missingness_understated_by_pp"] = (
        ((true_missing - naive_missing) * 100).sort_values(ascending=False)
        .head(25).round(1).to_dict()
    )
    print(f"  blanked {int(sentinel_counts.sum()):,} sentinel cells")

    # --- 2. dates, age ------------------------------------------------------
    df["visit_year"] = raw[C.COL_VISIT_DATE].map(_jalali_year)
    df["birth_year"] = raw[C.COL_BIRTH_DATE].map(_jalali_year)
    df["visit_ord"] = raw[C.COL_VISIT_DATE].map(_jalali_ordinal)
    df["age"] = df["visit_year"] - df["birth_year"]
    bad_age = ~df["age"].between(16, 95)
    qc["implausible_age_rows"] = int(bad_age.sum())
    df.loc[bad_age, "age"] = np.nan
    df["male"] = (raw[C.COL_SEX] == C.SEX_MALE).astype(int)
    df["female"] = (raw[C.COL_SEX] == C.SEX_FEMALE).astype(int)

    # --- 3. person + protocol identifiers -----------------------------------
    # NOTE: the export carries no patient key, so identity is reconstructed from
    # (birth date, sex). Same-day collisions are rare (see qc) but this remains a
    # quasi-identifier -- see paper/LIMITATIONS.md.
    df["person_id"] = (
        raw[C.COL_BIRTH_DATE].astype(str) + "|" + raw[C.COL_SEX].astype(str)
    )
    same_day = df.groupby(["person_id", C.COL_VISIT_DATE]).size()
    qc["person_id"] = {
        "n_person_ids": int(df["person_id"].nunique()),
        "n_visits": int(len(df)),
        "same_person_same_day_collisions": int((same_day > 1).sum()),
    }

    employer = raw[C.COL_EMPLOYER].fillna("unknown")
    big = employer.value_counts()
    big = big[big >= C.MIN_EMPLOYER_N].index
    df["employer"] = employer
    df["protocol"] = employer.where(employer.isin(big), "other_small")
    qc["n_protocol_folds"] = int(df["protocol"].nunique())

    # --- 4. core panel numerics + unit-error repair -------------------------
    unit_errors = {}
    for col in C.CORE_NUMERIC:
        if col not in df.columns:
            print(f"  ! missing core column {col!r}")
            continue
        v = coerce_numeric(df[col])
        lo, hi = C.PLAUSIBLE_RANGE[col]
        bad = v.notna() & ~v.between(lo, hi)
        unit_errors[col] = int(bad.sum())
        df[col] = v.mask(bad)
    qc["unit_errors_nulled"] = {k: v for k, v in sorted(
        unit_errors.items(), key=lambda kv: -kv[1]) if v}
    qc["unit_errors_total"] = int(sum(unit_errors.values()))
    print(f"  nulled {sum(unit_errors.values()):,} out-of-range core values")

    # derived ratios that are standard in screening practice
    df["nlr"] = df["N"] / df["L"].replace(0, np.nan)                # neutrophil:lymphocyte
    df["ast_alt"] = df["SGOT (AST)"] / df["SGPT (ALT)"].replace(0, np.nan)
    df["tg_hdl"] = df["Triglycerides"] / df["HDL cholestrol"].replace(0, np.nan)
    df["non_hdl"] = df["Cholesterol"] - df["HDL cholestrol"]
    df["transferrin_sat"] = 100 * df["IRON"] / df["TIBC"].replace(0, np.nan)
    df["pulse_pressure"] = df[C.COL_SBP] - df[C.COL_DBP]

    # --- 5. DEXA ------------------------------------------------------------
    dexa, dexa_report = coalesce_dexa(df)
    df = pd.concat([df, dexa], axis=1)
    qc["dexa"] = dexa_report

    # --- 6. corrupted columns ----------------------------------------------
    qc["corrupted_columns"] = C.CORRUPTED_COLUMNS
    df = df.drop(columns=[c for c in C.CORRUPTED_COLUMNS if c in df.columns])

    # --- 7. persist ---------------------------------------------------------
    df = df.sort_values(["person_id", "visit_ord"]).reset_index(drop=True)
    out = C.PROCESSED / "visits.parquet"
    df.to_parquet(out)
    (C.PROCESSED / "qc_report.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {out}  shape={df.shape}")
    print(f"  wrote {C.PROCESSED / 'qc_report.json'}")
    return df


if __name__ == "__main__":
    build()
