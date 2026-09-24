"""Stage 2 of deployment: fit deliverable models and serialise them.

Run:  python -m src.fit_final

`run_experiments.py` fits a model inside every CV fold, records its predictions
and discards the estimator. That is correct for measuring performance and leaves
nothing to deploy. This module fits one model per target on all eligible rows and
writes it to `models/`, together with an operating threshold and a model card.

Three decisions are made here, and each is arguable, so each is recorded in the
card rather than buried:

1. **Which targets ship.** Only those clearing the pre-specified negative control
   (gallstones, fold-wise AUROC 0.643) by a margin worth acting on. That is two
   of twelve. The rest are fitted only with `--all`, and are marked
   `status="not_validated"` so `predict.py` refuses them without an override.

2. **Which model arm.** `gbdt_values_only` -- median imputation, then gradient
   boosting -- not the marginally stronger `gbdt`. `gbdt` consumes the NaN
   pattern natively, and the NaN pattern in a new EHR encodes that site's
   ordering habits, not this programme's. `gbdt_values_only` is the only arm
   whose assumptions survive the move. The cost is <= 0.02 AUROC.

3. **How the threshold is set.** Per held-out employer, the threshold achieving
   90% sensitivity is computed; the shipped threshold is the median across
   employers. The card then reports what that single threshold actually achieved
   in each held-out employer -- which is the honest estimate of what a new site
   should expect, and it is not 90% on the nose.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import evaluate as E
from . import models as M
from .deploy import EXCLUDED_FEATURES, FEATURE_ORDER
from .features import LADDER, build_features
from .splits import make_splits
from .targets import TARGETS_BY_KEY, build_targets

MODEL_DIR = C.ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ARM = "gbdt_values_only"
DESIGN_FOR_THRESHOLD = "protocol"
FEATURE_SET = "D_full_core"
DEFAULT_SENSITIVITY = 0.90

# Fold-wise AUROC of the pre-specified negative control under the
# leave-one-employer-out design. Anything not clearly above this is noise.
NEGATIVE_CONTROL_AUROC = 0.643

VALIDATED: dict[str, str] = {
    "dexa_low_bone_mass":
        "fold-wise AUROC 0.733 +/- 0.08, +0.09 over the negative control; "
        "positive net benefit at a 10% threshold",
    "prostate_heterogeneous":
        "fold-wise AUROC 0.774 +/- 0.10, +0.13 over the negative control; "
        "positive net benefit at a 10% threshold",
}


# ------------------------------------------------------------------ threshold
def _oof_protocol(df: pd.DataFrame, y_full: pd.Series, X: pd.DataFrame,
                  idx: np.ndarray) -> pd.DataFrame:
    """Leave-one-employer-out predictions for the *deployed* configuration.

    The threshold cannot be read off `results/predictions/`: those were produced
    by `run_experiments` over the full analysis panel, which includes features
    this module withholds (see `deploy.EXCLUDED_FEATURES`). Deriving the
    operating point from a model that differs from the one being shipped is how
    a threshold silently stops meaning what the card says it means. So the folds
    are re-run here, on exactly the matrix the fitted pipeline consumes.
    """
    mask = y_full.notna()
    pos = {int(i): k for k, i in enumerate(idx)}
    y = y_full.to_numpy()[idx]
    rows = []
    for sp in make_splits(df, mask, DESIGN_FOR_THRESHOLD, y=y_full):
        tr = np.array([pos[int(i)] for i in sp.train])
        te = np.array([pos[int(i)] for i in sp.test])
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        mdl = M.build(MODEL_ARM)
        mdl.fit(X.iloc[tr], y[tr])
        rows.append(pd.DataFrame({
            "fold": str(sp.fold), "y": y[te],
            "p": mdl.predict_proba(X.iloc[te])[:, 1],
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["fold", "y", "p"])


def _fold_thresholds(d: pd.DataFrame, sens: float) -> dict:
    """Turn leave-one-employer-out predictions into a shipped operating point.

    Returns the threshold plus, critically, the spread of what it achieves across
    held-out employers. A threshold tuned on pooled predictions would look
    tighter and would be lying: baseline prevalence differs markedly between
    employers.
    """
    if d.empty:
        return {"error": "no employer folds could be fitted"}

    per_fold = []
    for fold, g in d.groupby("fold"):
        if len(g) < 100 or g["y"].sum() < 10 or g["y"].nunique() < 2:
            continue
        r = E.tests_avoided_at_sensitivity(g["y"].to_numpy(), g["p"].to_numpy(), sens)
        per_fold.append({"fold": str(fold), "n": int(len(g)),
                         "prevalence": float(g["y"].mean()),
                         "threshold_for_target_sens": float(r["threshold"])})
    if not per_fold:
        return {"error": "no held-out employer had enough signal to set a threshold"}

    chosen = float(np.median([f["threshold_for_target_sens"] for f in per_fold]))

    # What does that one threshold actually do at each site?
    achieved = []
    for fold, g in d.groupby("fold"):
        if len(g) < 100 or g["y"].sum() < 10 or g["y"].nunique() < 2:
            continue
        y, p = g["y"].to_numpy(), g["p"].to_numpy()
        refer = p >= chosen
        npos = y.sum()
        achieved.append({
            "fold": str(fold), "n": int(len(g)), "prevalence": float(y.mean()),
            "sensitivity": float(y[refer].sum() / npos) if npos else np.nan,
            "scans_avoided_%": float(100 * (~refer).mean()),
            "missed_n": int(y[~refer].sum()),
        })

    sens_vals = [a["sensitivity"] for a in achieved]
    saved_vals = [a["scans_avoided_%"] for a in achieved]
    return {
        "sensitivity_target": sens,
        "threshold": chosen,
        "threshold_across_folds": {
            "min": float(np.min([f["threshold_for_target_sens"] for f in per_fold])),
            "max": float(np.max([f["threshold_for_target_sens"] for f in per_fold])),
        },
        "at_this_threshold": {
            "sensitivity_median": float(np.median(sens_vals)),
            "sensitivity_min": float(np.min(sens_vals)),
            "sensitivity_max": float(np.max(sens_vals)),
            "scans_avoided_median_%": float(np.median(saved_vals)),
            "scans_avoided_min_%": float(np.min(saved_vals)),
            "scans_avoided_max_%": float(np.max(saved_vals)),
        },
        "per_employer": achieved,
        "n_employers": len(achieved),
    }


# ----------------------------------------------------------------------- fit
def fit_one(df: pd.DataFrame, labels: pd.DataFrame, target: str,
            sens: float = DEFAULT_SENSITIVITY) -> dict:
    t = TARGETS_BY_KEY[target]
    y_full = labels[target]
    mask = y_full.notna()          # == "this person received the module"
    idx = np.where(mask.to_numpy())[0]
    if len(idx) < 300:
        raise ValueError(f"{target}: only {len(idx)} labelled visits")

    X = build_features(df, LADDER[FEATURE_SET]).iloc[idx].reset_index(drop=True)
    X = X.reindex(columns=FEATURE_ORDER)
    y = y_full.to_numpy()[idx]

    pipe = M.build(MODEL_ARM)
    pipe.fit(X, y)

    oof = _oof_protocol(df, y_full, X, idx)
    thr = _fold_thresholds(oof, sens)
    if not oof.empty:
        thr["fold_wise_auroc"] = float(np.mean([
            E.auroc(g["y"], g["p"]) for _, g in oof.groupby("fold")
            if g["y"].nunique() > 1 and len(g) >= 100 and g["y"].sum() >= 10]))
    validated = target in VALIDATED

    card = {
        "target": target,
        "name": t.name,
        "module": t.module,
        "status": "validated" if validated else "not_validated",
        "evidence": VALIDATED.get(
            target,
            f"Does not clear the negative control (fold-wise AUROC {NEGATIVE_CONTROL_AUROC} "
            f"for gallstones). Fitted for completeness only; not for use."),
        "model_arm": MODEL_ARM,
        "why_this_arm": "Imputes every feature, so it cannot exploit which tests "
                        "were ordered. The ordering pattern is site-specific and "
                        "will not transfer.",
        "design_for_threshold": DESIGN_FOR_THRESHOLD,
        "feature_order": FEATURE_ORDER,
        "n_features": len(FEATURE_ORDER),
        "excluded_features": EXCLUDED_FEATURES,
        "sex_restriction": t.sex,
        "n_train_visits": int(len(idx)),
        "n_train_positive": int(np.nansum(y)),
        "train_prevalence": float(np.nanmean(y)),
        "operating_point": thr,
        "trained_on": {
            "source": C.RAW_XLSX.name,
            "cohort": "Iranian corporate occupational health screening, "
                      "median age 37, 62.8% male, programme years 1398-1405",
            "label_source": "reporting operator's own classification, not adjudicated",
        },
        "fitted": date.today().isoformat(),
        "random_state": C.RANDOM_STATE,
    }

    joblib.dump({"pipeline": pipe, "card": card}, MODEL_DIR / f"{target}.joblib")
    (MODEL_DIR / f"{target}.card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="also fit targets that fail the negative control "
                         "(written with status=not_validated)")
    ap.add_argument("--sensitivity", type=float, default=DEFAULT_SENSITIVITY)
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(C.PROCESSED / "visits.parquet")
    labels, _ = build_targets(df)

    keys = args.targets or (list(TARGETS_BY_KEY) if args.all else list(VALIDATED))

    print(f"fitting {len(keys)} target(s) -> {MODEL_DIR}\n")
    for k in keys:
        try:
            card = fit_one(df, labels, k, args.sensitivity)
        except Exception as exc:
            print(f"  ! {k}: {type(exc).__name__}: {exc}")
            continue
        op = card["operating_point"]
        print(f"  {k}")
        print(f"      status      {card['status']}")
        print(f"      trained on  {card['n_train_visits']:,} visits, "
              f"{card['n_train_positive']:,} positive "
              f"({card['train_prevalence']*100:.1f}%)")
        if "error" in op:
            print(f"      threshold   -- {op['error']}")
        else:
            a = op["at_this_threshold"]
            print(f"      threshold   {op['threshold']:.4f}  "
                  f"(targeting {op['sensitivity_target']*100:.0f}% sensitivity)")
            print(f"      across {op['n_employers']} held-out employers: "
                  f"sensitivity {a['sensitivity_min']*100:.0f}-{a['sensitivity_max']*100:.0f}%"
                  f" (median {a['sensitivity_median']*100:.0f}%), "
                  f"scans avoided {a['scans_avoided_min_%']:.0f}-{a['scans_avoided_max_%']:.0f}%"
                  f" (median {a['scans_avoided_median_%']:.0f}%)")
        print()


if __name__ == "__main__":
    main()
