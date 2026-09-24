"""Expensive-module targets: what the triage model is asked to anticipate.

Every target is a `Target` carrying
  * `label`     -- binary outcome, defined only where the module was performed
  * `eligible`  -- who could in principle receive the module (e.g. sex-specific)
  * `cost`      -- rough relative cost of the module, for net-benefit framing

The `label is not-null` mask is precisely "this person received this module",
which is the quantity driven by the employer's contract rather than by health.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Target:
    key: str
    name: str
    module: str
    build: Callable[[pd.DataFrame], pd.Series]
    sex: str | None = None          # 'male' | 'female' | None
    cost: str = "medium"            # low | medium | high
    negative_control: bool = False
    note: str = ""
    tags: list[str] = field(default_factory=list)


# ------------------------------------------------------------------- helpers
def _yes(df: pd.DataFrame, col: str) -> pd.Series:
    """yes/no exam field -> 1/0, missing where the exam was not performed."""
    s = df[col]
    return s.isin(["yes", "Yes", "YES"]).astype(float).where(s.notna())


def _isin(df: pd.DataFrame, col: str, positives: list[str]) -> pd.Series:
    s = df[col]
    return s.isin(positives).astype(float).where(s.notna())


def _dexa_low_bone_mass(df: pd.DataFrame) -> pd.Series:
    """Reporting radiologist classed either site as osteopenic or osteoporotic.

    Uses the BMD-filed category rather than the T-score-filed one: the two
    disagree sharply (11% vs 1.4% abnormal) and only the BMD figure is credible
    for a cohort of this age. See paper/LIMITATIONS.md.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for site in ("lhip", "lumbar_spine"):
        cat = df.get(f"dexa_bmd_{site}_cat")
        if cat is None:
            continue
        v = cat.isin(["osteopenia", "osteoporosis"]).astype(float).where(cat.notna())
        out = out.where(v.isna(), np.fmax(out.fillna(0), v))
    return out


def _birads(df: pd.DataFrame) -> pd.Series:
    """BI-RADS >= 3 (anything beyond negative/benign) on mammography."""
    s = df["birads_category"]
    pos = ["BIRAD 3 :Probably benign finding", "BIRAD 4 :suspicious abnormality",
           "BIRAD 5 :highly suggestive of malignancy", "BIRAD 6 :biopsy proven malignancy"]
    neg = ["BIRAD 1 :negative", "BIRAD 2 :Benign Finding"]
    return s.isin(pos).astype(float).where(s.isin(pos + neg))


def _tox_any(df: pd.DataFrame) -> pd.Series:
    drugs = ["Morphine", "Marijuana", "Cocaine", "Benzodiazepines",
             "Amphetamine", "Buprenorphin", "Methadone", "Tramadol"]
    have = pd.concat([df[d].notna() for d in drugs], axis=1).any(axis=1)
    pos = pd.concat([df[d].astype(str).str.contains("Positive", na=False) for d in drugs], axis=1).any(axis=1)
    return pos.astype(float).where(have)


def _k6_severe(df: pd.DataFrame) -> pd.Series:
    v = pd.to_numeric(df["امتیاز ارزیابی روانشناختی K6"], errors="coerce")
    return (v >= 13).astype(float).where(v.notna())


def _carotid_abnormal(df: pd.DataFrame) -> pd.Series:
    cols = ["rcca", "rica", "reca", "lcca", "lica", "leca"]
    normal = ["No plaque", "No stenosis"]
    have = pd.concat([df[c].notna() for c in cols], axis=1).any(axis=1)
    abn = pd.concat([df[c].notna() & ~df[c].isin(normal) for c in cols], axis=1).any(axis=1)
    return abn.astype(float).where(have)


# -------------------------------------------------------------------- registry
TARGETS: list[Target] = [
    Target("dexa_low_bone_mass", "Low bone mass (osteopenia/osteoporosis)",
           "DEXA densitometry", _dexa_low_bone_mass, cost="high",
           tags=["primary"],
           note="BMD-filed WHO category at hip or lumbar spine."),

    Target("us_nephrolithiasis", "Nephrolithiasis",
           "Abdominal ultrasound", lambda d: _yes(d, "nephrolithiasis"),
           cost="medium", tags=["primary"]),

    Target("us_kidney_stone", "Kidney stone",
           "Abdominal ultrasound", lambda d: _yes(d, "stone_in_the_kidney"),
           cost="medium"),

    Target("us_hydronephrosis", "Hydronephrosis",
           "Abdominal ultrasound", lambda d: _yes(d, "hydronephrosis"),
           cost="medium"),

    Target("us_gallstones", "Gallstones",
           "Abdominal ultrasound", lambda d: _yes(d, "presence_stones_in_the_gallbladder"),
           cost="medium", negative_control=True, tags=["primary"],
           note="Retained as a negative control: gallstone risk is dominated by "
                "factors not captured in blood chemistry, so a well-behaved model "
                "should show little lift here."),

    Target("prostate_heterogeneous", "Heterogeneous prostate echotexture",
           "Prostate ultrasound", lambda d: _isin(d, "kind_of_echo_in_prostate", ["Hetrogeneous"]),
           sex="male", cost="medium", tags=["primary"]),

    Target("echo_mitral_abnormal", "Mitral valve abnormality",
           "Echocardiography",
           lambda d: (d["mitral_valve"].notna() & d["mitral_valve"].ne("Mitral valve - normal")
                      ).astype(float).where(d["mitral_valve"].notna()),
           cost="high"),

    Target("echo_mr_mild_plus", "Mitral regurgitation >= mild",
           "Echocardiography",
           lambda d: _isin(d, "mr", ["MR - mild", "MR - mild to moderate",
                                     "MR - moderate", "MR - moderate to severe"]),
           cost="high"),

    Target("carotid_abnormal", "Any carotid plaque/stenosis/wall thickening",
           "Carotid Doppler", _carotid_abnormal, cost="high",
           note="Small denominator (~1.6k); reported with wide intervals."),

    Target("mammo_birads_3plus", "Mammography BI-RADS >= 3",
           "Mammography", _birads, sex="female", cost="high"),

    Target("k6_severe_distress", "Severe psychological distress (K6 >= 13)",
           "K6 questionnaire", _k6_severe, cost="low", tags=["primary"],
           note="Cheap module, but included to test whether blood chemistry "
                "carries any signal about mental-health screening outcomes."),

    Target("tox_any_positive", "Any positive urine drug screen",
           "Urine toxicology", _tox_any, cost="medium"),
]

TARGETS_BY_KEY = {t.key: t for t in TARGETS}


def build_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (labels, received) frames aligned to `df`.

    labels[key]   : 1/0/NaN outcome
    received[key] : 1 where the module was performed and the person was eligible
    """
    labels, received = {}, {}
    for t in TARGETS:
        y = t.build(df).astype(float)
        if t.sex == "male":
            y = y.where(df["male"] == 1)
        elif t.sex == "female":
            y = y.where(df["female"] == 1)
        labels[t.key] = y
        elig = pd.Series(True, index=df.index)
        if t.sex == "male":
            elig = df["male"] == 1
        elif t.sex == "female":
            elig = df["female"] == 1
        received[t.key] = (y.notna() & elig).astype(int)
    return pd.DataFrame(labels, index=df.index), pd.DataFrame(received, index=df.index)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    labels, received = build_targets(df)
    rows = []
    for t in TARGETS:
        y = labels[t.key]
        rows.append({
            "key": t.key, "target": t.name, "module": t.module,
            "sex": t.sex or "all", "cost": t.cost,
            "n_received": int(y.notna().sum()),
            "n_positive": int(np.nansum(y)),
            "prevalence_%": round(100 * y.mean(), 2) if y.notna().any() else np.nan,
            "coverage_%": round(100 * received[t.key].mean(), 1),
            "negative_control": t.negative_control,
        })
    return pd.DataFrame(rows).sort_values("n_received", ascending=False)


if __name__ == "__main__":
    d = pd.read_parquet(C.PROCESSED / "visits.parquet")
    s = summarise(d)
    print(s.to_string(index=False))
    s.to_csv(C.TABLES / "target_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nwrote {C.TABLES / 'target_summary.csv'}")
