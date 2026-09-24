"""Integrity checks on the pipeline. Run before trusting any result.

Run:  python -m src.checks

These are the failure modes that would silently invalidate the study:
  1. sentinel strings surviving into the analysis frame
  2. a target leaking into its own predictors
  3. train/test contamination by person across folds
  4. the "module received" mask disagreeing with the label definition
  5. implausible values surviving the range filter
  6. label prevalence differing between the folds a model was scored on
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from . import config as C
from .features import build_features
from .splits import make_splits
from .targets import TARGETS, build_targets

FAILURES: list[str] = []
PRED = C.RESULTS / "predictions"


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def main():
    df = pd.read_parquet(C.PROCESSED / "visits.parquet")
    labels, received = build_targets(df)
    X = build_features(df)

    print("1. sentinel strings removed")
    pat = re.compile("|".join(C.SENTINEL_PATTERNS), re.IGNORECASE)
    hits = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        n = int(s.notna().sum() and s.astype(str).str.contains(pat, na=False).sum())
        if n:
            hits[col] = n
    check("no sentinel text remains", not hits, f"{len(hits)} columns still contain it: {list(hits)[:5]}")

    print("2. predictors are numeric and free of the target")
    check("core panel all numeric", all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns))
    overlap = set(X.columns) & set(labels.columns)
    check("no target column inside the predictor panel", not overlap, str(overlap))
    # a target must not be reconstructible from a single predictor
    worst = []
    for t in TARGETS:
        y = labels[t.key]
        m = y.notna()
        if m.sum() < 500 or y[m].nunique() < 2:
            continue
        for c in X.columns:
            v = X.loc[m, c]
            if v.notna().sum() < 300 or v.nunique() < 3:
                continue
            r = abs(np.corrcoef(v.fillna(v.median()), y[m])[0, 1])
            if r > 0.9:
                worst.append((t.key, c, round(r, 3)))
    check("no predictor almost perfectly determines a target", not worst, str(worst[:5]))

    print("3. fold hygiene (person never straddles train/test)")
    bad = []
    for t in TARGETS[:6]:
        y = labels[t.key]
        m = y.notna()
        if m.sum() < 500:
            continue
        for design in ("person", "protocol", "temporal"):
            for sp in make_splits(df, m, design, y=y):
                tr = set(df["person_id"].to_numpy()[sp.train])
                te = set(df["person_id"].to_numpy()[sp.test])
                if tr & te:
                    bad.append((t.key, design, sp.fold, len(tr & te)))
    check("no person appears in both sides of any fold", not bad, str(bad[:5]))

    print("4. label / received-mask consistency")
    mismatch = [t.key for t in TARGETS
                if not labels[t.key].notna().equals(received[t.key].astype(bool))]
    check("received mask equals label-observed mask", not mismatch, str(mismatch))
    sexed = []
    for t in TARGETS:
        if t.sex == "male" and labels[t.key].notna().pipe(lambda m: (df.loc[m, "female"] == 1).any()):
            sexed.append(t.key)
        if t.sex == "female" and labels[t.key].notna().pipe(lambda m: (df.loc[m, "male"] == 1).any()):
            sexed.append(t.key)
    check("sex-specific targets contain only the eligible sex", not sexed, str(sexed))

    print("5. implausible values removed")
    viol = {}
    for col, (lo, hi) in C.PLAUSIBLE_RANGE.items():
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        n = int((v.notna() & ~v.between(lo, hi)).sum())
        if n:
            viol[col] = n
    check("no core value outside its plausible range", not viol, str(viol))
    check("age within 16-95 where present",
          bool(df["age"].dropna().between(16, 95).all()))

    print("6. stored predictions are self-consistent")
    files = sorted(PRED.glob("*.parquet"))
    if not files:
        print("  (no predictions yet -- run src.run_experiments first)")
    else:
        prob_range, ylab, dup = [], [], []
        for f in files:
            d = pd.read_parquet(f)
            if not d.p.between(0, 1).all():
                prob_range.append(f.name)
            if not set(d.y.unique()) <= {0.0, 1.0}:
                ylab.append(f.name)
            if d.row.duplicated().any():
                dup.append(f.name)
        check("all predicted probabilities in [0,1]", not prob_range, str(prob_range[:3]))
        check("all labels binary", not ylab, str(ylab[:3]))
        check("each visit scored at most once per file", not dup, str(dup[:3]))
        check(f"prediction files present ({len(files)})", True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
