"""Metrics: discrimination, calibration, and the decision-analytic quantities
that matter for a referral policy.

Discrimination alone does not answer "should we order this test". The two
quantities the paper leans on are

  net benefit          -- decision-curve analysis; is triaging on the model
                          better than testing everyone or testing nobody, at a
                          clinically defensible risk threshold?
  tests avoided @ sens -- if the programme insists on catching >= S of the
                          abnormal findings, what fraction of scans can it skip?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from . import config as C


# ------------------------------------------------------------- discrimination
def auroc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def auprc(y, p):
    return average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan


# --------------------------------------------------------------- calibration
def calibration(y, p, eps: float = 1e-6) -> dict:
    """Cox calibration intercept/slope on the logit scale, plus Brier and ECE."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    logit = np.log(p / (1 - p))
    out = {"brier": float(brier_score_loss(y, p))}
    try:
        import statsmodels.api as sm
        slope_fit = sm.GLM(y, sm.add_constant(logit), family=sm.families.Binomial()).fit()
        out["cal_intercept"] = float(slope_fit.params[0])
        out["cal_slope"] = float(slope_fit.params[1])
    except Exception:
        out["cal_intercept"] = np.nan
        out["cal_slope"] = np.nan
    # 10-bin expected calibration error
    bins = np.quantile(p, np.linspace(0, 1, 11))
    bins[0], bins[-1] = -np.inf, np.inf
    idx = np.digitize(p, bins[1:-1])
    ece = 0.0
    for b in np.unique(idx):
        m = idx == b
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    out["ece"] = float(ece)
    return out


# ----------------------------------------------------------- decision curves
def net_benefit(y, p, threshold: float) -> float:
    """Vickers & Elkin net benefit of referring everyone with p >= threshold."""
    y = np.asarray(y, float)
    refer = np.asarray(p, float) >= threshold
    n = len(y)
    tp = float(((refer == 1) & (y == 1)).sum())
    fp = float(((refer == 1) & (y == 0)).sum())
    w = threshold / (1 - threshold)
    return tp / n - fp / n * w


def net_benefit_all(y, threshold: float) -> float:
    y = np.asarray(y, float)
    prev = y.mean()
    return prev - (1 - prev) * threshold / (1 - threshold)


def decision_curve(y, p, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.51, 0.01), 3)
    rows = []
    for t in thresholds:
        rows.append({
            "threshold": t,
            "nb_model": net_benefit(y, p, t),
            "nb_test_all": net_benefit_all(y, t),
            "nb_test_none": 0.0,
        })
    d = pd.DataFrame(rows)
    # standardised: extra true findings per 100 people vs the better default
    d["nb_best_default"] = d[["nb_test_all", "nb_test_none"]].max(axis=1)
    d["delta_vs_default_per_100"] = 100 * (d["nb_model"] - d["nb_best_default"])
    return d


# ------------------------------------------------------------- operational
def tests_avoided_at_sensitivity(y, p, sens: float = 0.90) -> dict:
    """Rank by predicted risk; keep testing until `sens` of positives captured."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    order = np.argsort(-p)
    ys = y[order]
    npos = ys.sum()
    if npos == 0:
        return {"sens_target": sens, "tests_avoided_%": np.nan, "achieved_sens": np.nan,
                "threshold": np.nan, "ppv": np.nan, "npv": np.nan}
    cum = np.cumsum(ys)
    need = np.searchsorted(cum, sens * npos) + 1
    need = min(need, len(ys))
    tested = ys[:need]
    skipped = ys[need:]
    return {
        "sens_target": sens,
        "tests_avoided_%": 100 * (len(ys) - need) / len(ys),
        "achieved_sens": float(tested.sum() / npos),
        "threshold": float(p[order][need - 1]),
        "ppv": float(tested.mean()) if need else np.nan,
        "npv": float(1 - skipped.mean()) if len(skipped) else np.nan,
        "missed_n": int(skipped.sum()),
    }


# --------------------------------------------------------------- aggregation
def summarise_predictions(y, p, groups=None, n_boot: int = 0,
                          rng_seed: int = C.RANDOM_STATE) -> dict:
    """Point estimates plus optional person-clustered bootstrap intervals."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    out = {
        "n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
        "auroc": auroc(y, p), "auprc": auprc(y, p),
        "auprc_lift": (auprc(y, p) / y.mean()) if y.mean() > 0 else np.nan,
    }
    out.update(calibration(y, p))
    for s in (0.80, 0.90, 0.95):
        r = tests_avoided_at_sensitivity(y, p, s)
        out[f"tests_avoided_at_sens{int(s*100)}_%"] = r["tests_avoided_%"]
        out[f"missed_at_sens{int(s*100)}"] = r["missed_n"]
    dc = decision_curve(y, p)
    for t in (0.05, 0.10, 0.20):
        row = dc.loc[(dc.threshold - t).abs().idxmin()]
        out[f"net_benefit_gain_per100_at_pt{int(t*100)}"] = row["delta_vs_default_per_100"]

    if n_boot and groups is not None:
        rng = np.random.default_rng(rng_seed)
        uniq = np.unique(groups)
        stats = {"auroc": [], "auprc": [], "tests_avoided_at_sens90_%": []}
        gidx = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([gidx[g] for g in pick])
            yb, pb = y[sel], p[sel]
            if len(np.unique(yb)) < 2:
                continue
            stats["auroc"].append(auroc(yb, pb))
            stats["auprc"].append(auprc(yb, pb))
            stats["tests_avoided_at_sens90_%"].append(
                tests_avoided_at_sensitivity(yb, pb, 0.90)["tests_avoided_%"])
        for k, v in stats.items():
            if v:
                out[f"{k}_lo"] = float(np.nanpercentile(v, 2.5))
                out[f"{k}_hi"] = float(np.nanpercentile(v, 97.5))
    return out


def macro_by_fold(d: pd.DataFrame, min_n: int = 100, min_pos: int = 10) -> dict:
    """Fold-wise metrics averaged across folds.

    For the leave-one-employer-out design, pooling out-of-fold predictions and
    scoring them together is misleading: baseline prevalence differs markedly
    between employers, so a model that is perfectly discriminating *within* every
    employer can still score poorly on the pooled set (and a constant-prevalence
    baseline scores below 0.5). Averaging the metric computed separately in each
    held-out employer is the quantity that answers "does this transfer to a new
    site". Reported alongside, not instead of, the pooled figure.
    """
    aur, aup, lift, saved, ns = [], [], [], [], []
    for _, g in d.groupby("fold"):
        if len(g) < min_n or g["y"].sum() < min_pos or g["y"].nunique() < 2:
            continue
        aur.append(auroc(g["y"], g["p"]))
        ap = auprc(g["y"], g["p"])
        aup.append(ap)
        lift.append(ap / g["y"].mean())
        saved.append(tests_avoided_at_sensitivity(g["y"].values, g["p"].values, .90)["tests_avoided_%"])
        ns.append(len(g))
    if not aur:
        return {}
    w = np.array(ns, float)
    return {
        "n_folds_scored": len(aur),
        "macro_auroc": float(np.mean(aur)),
        "macro_auroc_sd": float(np.std(aur, ddof=1)) if len(aur) > 1 else 0.0,
        "macro_auroc_min": float(np.min(aur)),
        "macro_auroc_max": float(np.max(aur)),
        "weighted_auroc": float(np.average(aur, weights=w)),
        "macro_auprc_lift": float(np.mean(lift)),
        "macro_tests_avoided_at_sens90_%": float(np.mean(saved)),
    }


def delong_like_bootstrap_delta(y, p1, p2, groups, n_boot: int = 500,
                                rng_seed: int = C.RANDOM_STATE) -> dict:
    """Person-clustered bootstrap of AUROC difference between two models."""
    y, p1, p2 = map(lambda a: np.asarray(a, float), (y, p1, p2))
    rng = np.random.default_rng(rng_seed)
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([gidx[g] for g in pick])
        if len(np.unique(y[sel])) < 2:
            continue
        deltas.append(auroc(y[sel], p2[sel]) - auroc(y[sel], p1[sel]))
    if not deltas:
        return {}
    deltas = np.array(deltas)
    return {
        "delta_auroc": float(auroc(y, p2) - auroc(y, p1)),
        "delta_lo": float(np.percentile(deltas, 2.5)),
        "delta_hi": float(np.percentile(deltas, 97.5)),
        "p_two_sided": float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean())),
    }
