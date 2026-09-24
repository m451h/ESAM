"""Subgroup performance and screening-equity analysis.

Two questions the discussion section has to answer:

1. Does the triage model work equally well across sex, age band and employer?
   A referral rule that discriminates well on average but poorly for one group
   would concentrate missed findings in that group.

2. Who is currently under-screened? Module coverage is set by employer contract,
   so some groups already receive fewer expensive studies. A model trained on
   that pattern inherits it. We report coverage by subgroup alongside model
   performance so the two can be read together.

Run:  python -m src.subgroups
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import evaluate as E
from .targets import TARGETS, TARGETS_BY_KEY, build_targets

PRED = C.RESULTS / "predictions"
PRIMARY = ["dexa_low_bone_mass", "us_nephrolithiasis", "prostate_heterogeneous",
           "us_gallstones", "k6_severe_distress"]
AGE_BINS = [15, 30, 40, 50, 100]
AGE_LABELS = ["<30", "30-39", "40-49", "50+"]


def _load(target, design="protocol", model="gbdt"):
    f = PRED / f"{target}__{design}__{model}__D_full_core.parquet"
    if not f.exists():
        f = PRED / f"{target}__person__{model}__D_full_core.parquet"
    return pd.read_parquet(f) if f.exists() else None


def performance_by_subgroup(df: pd.DataFrame, design="protocol") -> pd.DataFrame:
    meta = df[["male", "age", "protocol"]].copy()
    meta["sex"] = np.where(meta["male"] == 1, "Men", "Women")
    meta["age_band"] = pd.cut(meta["age"], AGE_BINS, labels=AGE_LABELS, right=False)

    rows = []
    for key in PRIMARY:
        d = _load(key, design)
        if d is None:
            continue
        m = meta.iloc[d["row"].values].reset_index(drop=True)
        d = d.reset_index(drop=True)
        for var in ("sex", "age_band"):
            for lvl, g in d.groupby(m[var].astype(str), observed=True):
                if len(g) < 150 or g.y.nunique() < 2 or g.y.sum() < 10:
                    continue
                rows.append({
                    "target": key, "target_name": TARGETS_BY_KEY[key].name,
                    "variable": var, "level": lvl, "n": len(g),
                    "prevalence_%": round(100 * g.y.mean(), 1),
                    "AUROC": round(E.auroc(g.y, g.p), 3),
                    "AUPRC": round(E.auprc(g.y, g.p), 3),
                    "scans_avoided_at_sens90_%": round(
                        E.tests_avoided_at_sensitivity(g.y.values, g.p.values, .90)["tests_avoided_%"], 1),
                })
    return pd.DataFrame(rows)


def coverage_by_subgroup(df: pd.DataFrame) -> pd.DataFrame:
    """Who currently receives each expensive module."""
    _, received = build_targets(df)
    meta = pd.DataFrame({
        "sex": np.where(df["male"] == 1, "Men", "Women"),
        "age_band": pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str),
    })
    rows = []
    for t in TARGETS:
        r = received[t.key]
        elig = pd.Series(True, index=df.index)
        if t.sex == "male":
            elig = df["male"] == 1
        elif t.sex == "female":
            elig = df["female"] == 1
        for var in ("sex", "age_band"):
            for lvl, idx in meta[elig].groupby(meta.loc[elig, var], observed=True).groups.items():
                if len(idx) < 200:
                    continue
                rows.append({"module": t.module, "target": t.key,
                             "variable": var, "level": str(lvl),
                             "n_eligible": len(idx),
                             "coverage_%": round(100 * r.loc[idx].mean(), 1)})
    return pd.DataFrame(rows)


def calibration_drift_across_protocols(target="dexa_low_bone_mass") -> pd.DataFrame:
    """Prevalence and calibration per held-out employer.

    Expected finding: discrimination transfers better than calibration, because
    baseline prevalence differs across employers. This is the honest caveat for
    any deployment -- thresholds need local recalibration.
    """
    d = _load(target, "protocol")
    if d is None:
        return pd.DataFrame()
    rows = []
    for fold, g in d.groupby("fold"):
        if len(g) < 150 or g.y.nunique() < 2:
            continue
        cal = E.calibration(g.y.values, g.p.values)
        rows.append({"held_out_protocol": str(fold)[:34], "n": len(g),
                     "observed_prev_%": round(100 * g.y.mean(), 1),
                     "mean_predicted_%": round(100 * g.p.mean(), 1),
                     "AUROC": round(E.auroc(g.y, g.p), 3),
                     "cal_intercept": round(cal["cal_intercept"], 2),
                     "cal_slope": round(cal["cal_slope"], 2)})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main():
    df = pd.read_parquet(C.PROCESSED / "visits.parquet")

    perf = performance_by_subgroup(df)
    if not perf.empty:
        perf.to_csv(C.TABLES / "subgroup_performance.csv", index=False, encoding="utf-8-sig")
        print("--- model performance by subgroup ---")
        print(perf.to_string(index=False))
        print()
        print("largest within-target AUROC gaps:")
        for (k, var), g in perf.groupby(["target", "variable"]):
            if len(g) > 1:
                gap = g.AUROC.max() - g.AUROC.min()
                lo = g.loc[g.AUROC.idxmin()]
                hi = g.loc[g.AUROC.idxmax()]
                print(f"  {k:24s} {var:9s} gap {gap:.3f}  "
                      f"({lo.level}={lo.AUROC:.3f} vs {hi.level}={hi.AUROC:.3f})")
        print()

    cov = coverage_by_subgroup(df)
    cov.to_csv(C.TABLES / "coverage_by_subgroup.csv", index=False, encoding="utf-8-sig")
    print("--- current module coverage by subgroup (%) ---")
    piv = cov.pivot_table(index=["module", "target"], columns="level", values="coverage_%")
    print(piv.to_string())
    print()

    for k in ("dexa_low_bone_mass", "us_nephrolithiasis"):
        drift = calibration_drift_across_protocols(k)
        if drift.empty:
            continue
        drift.to_csv(C.TABLES / f"calibration_drift_{k}.csv", index=False, encoding="utf-8-sig")
        print(f"--- {k}: calibration across held-out employers ---")
        print(drift.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
