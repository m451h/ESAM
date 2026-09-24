"""Cohort flow, Table 1, and the module-coverage matrix that motivates the design.

Run:  python -m src.describe_cohort
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .features import core_panel_completeness
from .targets import TARGETS, build_targets


def cohort_flow(df: pd.DataFrame) -> pd.DataFrame:
    comp = core_panel_completeness(df)
    steps = [
        ("Visit records in merged export", len(df)),
        ("Reconstructed individuals", df["person_id"].nunique()),
        ("Visits with usable age", int(df["age"].notna().sum())),
        ("Visits with >=50% of core panel measured", int((comp >= .50).sum())),
        ("Visits with >=80% of core panel measured", int((comp >= .80).sum())),
        ("Individuals with >=2 visits", int((df.groupby("person_id").size() >= 2).sum())),
        ("Individuals with >=3 visits", int((df.groupby("person_id").size() >= 3).sum())),
    ]
    return pd.DataFrame(steps, columns=["step", "n"])


def table_one(df: pd.DataFrame) -> pd.DataFrame:
    """Characteristics overall and by sex."""
    def block(d: pd.DataFrame, name: str) -> dict:
        num = lambda c: pd.to_numeric(d[c], errors="coerce")
        row = {"group": name, "visits": len(d), "individuals": d["person_id"].nunique()}
        row["age median (IQR)"] = _mq(num("age"))
        row["female %"] = round(100 * d["female"].mean(), 1)
        for label, col in [("BMI", C.COL_BMI), ("SBP mmHg", C.COL_SBP), ("DBP mmHg", C.COL_DBP),
                           ("Fasting glucose", "Fasting Blood Sugar"), ("HbA1c %", "A1C"),
                           ("Total cholesterol", "Cholesterol"), ("Triglycerides", "Triglycerides"),
                           ("HDL-C", "HDL cholestrol"), ("LDL-C", "LDL cholestrol"),
                           ("ALT", "SGPT (ALT)"), ("Creatinine", "Creatinine"),
                           ("TSH", "TSH"), ("25-OH vitamin D", "VT D3"),
                           ("Haemoglobin", "HB")]:
            row[label] = _mq(num(col))
        row["married %"] = round(100 * d[C.COL_MARITAL].eq("متاهل در کنار همسر").mean(), 1)
        row["tobacco use % (of respondents)"] = _pct_tobacco(d)
        row["physically active %"] = round(100 * d["فعالیت فیزیکی"].eq("دارد").mean(), 1)
        return row

    def _mq(s: pd.Series) -> str:
        s = s.dropna()
        if s.empty:
            return "-"
        return f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})"

    def _pct_tobacco(d: pd.DataFrame) -> float:
        """Direct tobacco use. Item is asked of ~47% of visits; 'declined to
        answer' and passive-exposure responses are excluded from the denominator."""
        col = "مصرف انواع تنباکو"
        if col not in d:
            return np.nan
        s = d[col].dropna()
        s = s[s.isin(["بله", "خیر"])]
        if s.empty:
            return np.nan
        return round(100 * s.eq("بله").mean(), 1)

    rows = [block(df, "All"),
            block(df[df["male"] == 1], "Men"),
            block(df[df["female"] == 1], "Women")]
    return pd.DataFrame(rows).set_index("group").T


def module_coverage(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coverage of each expensive module by employer and by programme year.

    This is the empirical backing for the leave-one-protocol-out design: if
    coverage were driven by patient health it would be similar across employers.
    """
    _, received = build_targets(df)
    keys = [t.key for t in TARGETS]
    by_emp = (received.assign(protocol=df["protocol"].values)
              .groupby("protocol")[keys].mean().mul(100).round(1))
    by_emp["n_visits"] = df.groupby("protocol").size()
    by_emp = by_emp[by_emp["n_visits"] >= C.MIN_EMPLOYER_N].sort_values("n_visits", ascending=False)
    by_year = (received.assign(year=df["visit_year"].values)
               .groupby("year")[keys].mean().mul(100).round(1))
    by_year["n_visits"] = df.groupby("visit_year").size()
    return by_emp, by_year


def longitudinal_summary(df: pd.DataFrame) -> pd.DataFrame:
    comp = core_panel_completeness(df)
    d = df.assign(ok=(comp >= .5))
    g = d[d.ok].groupby("person_id")["visit_year"].nunique()
    rows = [{"distinct_lab_years": int(k), "individuals": int(v),
             "person_visits": int(k * v)} for k, v in g.value_counts().sort_index().items()]
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(C.PROCESSED / "visits.parquet")

    flow = cohort_flow(df)
    flow.to_csv(C.TABLES / "cohort_flow.csv", index=False, encoding="utf-8-sig")
    print("--- cohort flow ---"); print(flow.to_string(index=False)); print()

    t1 = table_one(df)
    t1.to_csv(C.TABLES / "table1_characteristics.csv", encoding="utf-8-sig")
    print("--- Table 1 ---"); print(t1.to_string()); print()

    by_emp, by_year = module_coverage(df)
    by_emp.to_csv(C.TABLES / "module_coverage_by_employer.csv", encoding="utf-8-sig")
    by_year.to_csv(C.TABLES / "module_coverage_by_year.csv", encoding="utf-8-sig")
    print("--- module coverage by employer (%) ---")
    print(by_emp.to_string()); print()
    print("--- module coverage by programme year (%) ---")
    print(by_year.to_string()); print()

    lon = longitudinal_summary(df)
    lon.to_csv(C.TABLES / "longitudinal_summary.csv", index=False, encoding="utf-8-sig")
    print("--- repeat participation ---"); print(lon.to_string(index=False))

    spread = {k: {"min": float(by_emp[k].min()), "max": float(by_emp[k].max()),
                  "range_pp": float(by_emp[k].max() - by_emp[k].min())}
              for k in [t.key for t in TARGETS] if k in by_emp}
    (C.TABLES / "module_coverage_spread.json").write_text(
        json.dumps(spread, indent=2), encoding="utf-8")
    print("\n--- coverage spread across employers (percentage points) ---")
    for k, v in sorted(spread.items(), key=lambda kv: -kv[1]["range_pp"]):
        print(f"  {k:24s} {v['min']:5.1f}% - {v['max']:5.1f}%   range {v['range_pp']:5.1f}pp")


if __name__ == "__main__":
    main()
