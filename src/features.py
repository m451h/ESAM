"""Core (cheap) feature panel — the only inputs the triage model may use.

The premise of the study is that these are obtained on essentially every
attendee, so a model built on them can be applied before deciding whether to
order an expensive module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

DERIVED = ["nlr", "ast_alt", "tg_hdl", "non_hdl", "transferrin_sat", "pulse_pressure"]

BLOCKS = {
    "demographics": ["age", "male"],
    "vitals": C.CORE_VITALS + ["pulse_pressure"],
    "chemistry": C.CORE_CHEMISTRY + ["ast_alt", "tg_hdl", "non_hdl", "transferrin_sat"],
    "haematology": C.CORE_HAEMATOLOGY + ["nlr"],
}

LADDER = {
    "A_demographics": ["demographics"],
    "B_plus_vitals": ["demographics", "vitals"],
    "C_plus_chemistry": ["demographics", "vitals", "chemistry"],
    "D_full_core": ["demographics", "vitals", "chemistry", "haematology"],
}


def feature_columns(blocks: list[str] | None = None) -> list[str]:
    blocks = blocks or list(BLOCKS)
    cols: list[str] = []
    for b in blocks:
        for c in BLOCKS[b]:
            if c not in cols:
                cols.append(c)
    return cols


def build_features(df: pd.DataFrame, blocks: list[str] | None = None) -> pd.DataFrame:
    """Numeric core panel. Missing values are left as NaN on purpose: the models
    either consume NaN natively (histogram GBDT) or impute inside a pipeline
    fitted on training folds only."""
    cols = [c for c in feature_columns(blocks) if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    return X


def core_panel_completeness(df: pd.DataFrame) -> pd.Series:
    """Fraction of the full core panel observed per visit — used for cohort flow."""
    X = build_features(df)
    return X.notna().mean(axis=1)


def missingness_indicators(X: pd.DataFrame) -> pd.DataFrame:
    """Binary 'was this measured' companions, for the missingness-aware arm."""
    return X.notna().astype(np.int8).add_suffix("__observed")


if __name__ == "__main__":
    d = pd.read_parquet(C.PROCESSED / "visits.parquet")
    X = build_features(d)
    comp = core_panel_completeness(d)
    print(f"core panel: {X.shape[1]} features over {len(X):,} visits")
    print(f"mean completeness {comp.mean()*100:.1f}%   "
          f">=80% complete: {(comp>=.8).mean()*100:.1f}% of visits")
    rep = pd.DataFrame({
        "observed_%": (X.notna().mean() * 100).round(1),
        "median": X.median().round(2),
        "iqr_lo": X.quantile(.25).round(2),
        "iqr_hi": X.quantile(.75).round(2),
    }).sort_values("observed_%", ascending=False)
    print(rep.to_string())
    rep.to_csv(C.TABLES / "core_panel_description.csv", encoding="utf-8-sig")
