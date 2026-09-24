"""Model arms.

`prevalence`  -- predicts the training prevalence for everyone. Floor.
`age_sex_lr`  -- the clinical status quo: who gets referred today is largely a
                 function of age and sex.
`logistic`    -- regularised linear model on median-imputed, standardised core
                 panel. Tests whether anything non-linear is needed.
`gbdt`        -- histogram gradient boosting; consumes NaN natively, so it sees
                 the real missingness pattern rather than an imputed one.
`gbdt_miss`   -- gbdt plus explicit "was this analyte measured" indicators.
`gbdt_values_only` -- gbdt on the core panel with the *ordering* signal removed
                 by imputing every feature, i.e. no NaN pattern left to exploit.

Reading the two control arms
----------------------------
Which analytes were ordered is itself protocol information, so a model that
consumes NaN natively could in principle score well without using any measured
value. These two arms bracket that.

`gbdt_miss` reproduces `gbdt` *bit-for-bit* on every target. This is expected,
not a bug: histogram gradient boosting already learns a default direction for
missing values at each split, so the added indicators are exactly redundant with
information the model has, and under deterministic split selection they are never
chosen. Verified directly -- the augmented matrix does carry 50 extra columns, 49
with non-zero variance, and the predictions still match to 0.0.

The arm that actually carries information is therefore `gbdt_values_only`:
imputing every feature destroys the NaN pattern, so the gap between it and
`gbdt` is the share of performance attributable to *which tests were ordered*
rather than to their values. That gap is small throughout (<= 0.02 AUROC), which
is the claim the paper makes. Do not cite `gbdt_miss` as evidence on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import missingness_indicators


class PrevalenceBaseline(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.p_)
        return np.column_stack([1 - p, p])


def _logistic() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", C=1.0, max_iter=3000,
                                   class_weight=None, random_state=C.RANDOM_STATE)),
    ])


def _gbdt() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
        random_state=C.RANDOM_STATE,
    )


def _gbdt_imputed() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", _gbdt())])


class WithMissingnessIndicators(BaseEstimator, ClassifierMixin):
    """GBDT on [values || observed-indicators]."""

    def __init__(self):
        self.inner = _gbdt()

    def _aug(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([X, missingness_indicators(X)], axis=1)

    def fit(self, X, y):
        self.columns_ = list(self._aug(X).columns)
        self.inner.fit(self._aug(X), y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        A = self._aug(X).reindex(columns=self.columns_, fill_value=0)
        return self.inner.predict_proba(A)


MODELS = {
    "prevalence": PrevalenceBaseline,
    "age_sex_lr": _logistic,          # restricted to demographic block at call site
    "logistic": _logistic,
    "gbdt": _gbdt,
    "gbdt_miss": WithMissingnessIndicators,
    "gbdt_values_only": _gbdt_imputed,
}


def build(name: str):
    return MODELS[name]()
