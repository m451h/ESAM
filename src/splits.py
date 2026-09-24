"""Validation designs.

Three designs of increasing external-validity demand:

`person`   -- 5-fold grouped by reconstructed person id. Guards against the same
              individual's repeat visits straddling train and test. This is the
              weakest design and is reported only for comparability.

`protocol` -- leave-one-employer-out. Which expensive modules a person receives
              is set by their employer's screening contract and the programme
              year, not by their health: coverage of DEXA ranges from 1% to 83%
              across employers. Holding out a whole employer therefore tests
              transfer to a screening protocol the model never saw, and is the
              design this study leans on.

`temporal` -- train on the earlier programme years, test on the later ones.
              Tests drift in assays, staff and case-mix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config as C


@dataclass
class Split:
    design: str
    fold: str
    train: np.ndarray
    test: np.ndarray


def person_folds(df: pd.DataFrame, mask: pd.Series, n_splits: int = 5) -> Iterator[Split]:
    idx = np.where(mask.to_numpy())[0]
    groups = df["person_id"].to_numpy()[idx]
    n = min(n_splits, len(np.unique(groups)))
    if n < 2:
        return
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n).split(idx, groups=groups)):
        yield Split("person", f"fold{k+1}", idx[tr], idx[te])


def protocol_folds(df: pd.DataFrame, mask: pd.Series,
                   min_test: int = 150, min_test_pos: int = 10,
                   y: pd.Series | None = None) -> Iterator[Split]:
    """Leave-one-employer-out over employers with enough held-out signal.

    A person can in principle appear under two employers across years (job
    change); those visits are dropped from the training side of the fold so that
    no individual straddles the boundary.
    """
    idx = np.where(mask.to_numpy())[0]
    protocol = df["protocol"].to_numpy()
    person = df["person_id"].to_numpy()
    for name, cnt in pd.Series(protocol[idx]).value_counts().items():
        if name == "other_small" or cnt < min_test:
            continue
        te = idx[protocol[idx] == name]
        if y is not None:
            pos = np.nansum(y.to_numpy()[te])
            if pos < min_test_pos or pos > len(te) - min_test_pos:
                continue
        held_people = set(person[te])
        tr = np.array([i for i in idx if protocol[i] != name and person[i] not in held_people])
        if len(tr) < 500:
            continue
        yield Split("protocol", str(name), tr, te)


def temporal_folds(df: pd.DataFrame, mask: pd.Series, cut: float = 1403) -> Iterator[Split]:
    idx = np.where(mask.to_numpy())[0]
    year = pd.to_numeric(df["visit_year"], errors="coerce").to_numpy()
    person = df["person_id"].to_numpy()
    te = idx[year[idx] >= cut]
    held = set(person[te])
    tr = np.array([i for i in idx if year[i] < cut and person[i] not in held])
    if len(tr) >= 500 and len(te) >= 150:
        yield Split("temporal", f"train<{int(cut)}_test>={int(cut)}", tr, te)


DESIGNS = {"person": person_folds, "protocol": protocol_folds, "temporal": temporal_folds}


def make_splits(df: pd.DataFrame, mask: pd.Series, design: str,
                y: pd.Series | None = None) -> list[Split]:
    if design == "protocol":
        return list(protocol_folds(df, mask, y=y))
    return list(DESIGNS[design](df, mask))


if __name__ == "__main__":
    d = pd.read_parquet(C.PROCESSED / "visits.parquet")
    from .targets import build_targets, TARGETS
    labels, _ = build_targets(d)
    rows = []
    for t in TARGETS:
        m = labels[t.key].notna()
        for design in ("person", "protocol", "temporal"):
            sp = make_splits(d, m, design, y=labels[t.key])
            rows.append({"target": t.key, "design": design, "n_folds": len(sp),
                         "median_test_n": int(np.median([len(s.test) for s in sp])) if sp else 0})
    out = pd.DataFrame(rows).pivot(index="target", columns="design",
                                   values=["n_folds", "median_test_n"])
    print(out.to_string())
