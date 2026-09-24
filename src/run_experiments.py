"""Main experiment driver.

Run:
    python -m src.run_experiments                 # everything
    python -m src.run_experiments --quick         # person-design only, no bootstrap
    python -m src.run_experiments --targets dexa_low_bone_mass us_nephrolithiasis

Writes
    results/predictions/<target>__<design>__<model>__<featureset>.parquet
    results/tables/main_results.csv
    results/tables/ablation.csv
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from . import config as C
from . import evaluate as E
from . import models as M
from .features import build_features, LADDER
from .splits import make_splits
from .targets import TARGETS, TARGETS_BY_KEY, build_targets

PRED_DIR = C.RESULTS / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

MAIN_MODELS = ["prevalence", "logistic", "gbdt", "gbdt_miss", "gbdt_values_only"]
MAIN_DESIGNS = ["person", "protocol", "temporal"]


def _fit_predict(model_name: str, X: pd.DataFrame, y: np.ndarray,
                 tr: np.ndarray, te: np.ndarray, positions: dict) -> np.ndarray:
    """Fit on train rows, predict test rows. tr/te are dataframe-level indices."""
    tr_pos = np.array([positions[int(i)] for i in tr])
    te_pos = np.array([positions[int(i)] for i in te])
    Xtr, ytr = X.iloc[tr_pos], y[tr_pos]
    if len(np.unique(ytr)) < 2:
        return np.full(len(te_pos), float(ytr.mean()))
    mdl = M.build(model_name)
    mdl.fit(Xtr, ytr)
    return mdl.predict_proba(X.iloc[te_pos])[:, 1]


def run_target(df: pd.DataFrame, labels: pd.DataFrame, target_key: str,
               designs: list[str], model_names: list[str],
               feature_sets: dict[str, list[str]], n_boot: int,
               boot_models: tuple[str, ...] = ("gbdt",)) -> list[dict]:
    t = TARGETS_BY_KEY[target_key]
    y_full = labels[target_key]
    mask = y_full.notna()
    if mask.sum() < 300 or np.nansum(y_full) < 25:
        print(f"  [skip] {target_key}: n={int(mask.sum())} pos={int(np.nansum(y_full))}")
        return []

    rows = []
    sub_idx = np.where(mask.to_numpy())[0]
    positions = {int(i): k for k, i in enumerate(sub_idx)}
    y = y_full.to_numpy()[sub_idx]
    groups = df["person_id"].to_numpy()[sub_idx]

    for fs_name, blocks in feature_sets.items():
        X = build_features(df, blocks).iloc[sub_idx].reset_index(drop=True)
        for design in designs:
            splits = make_splits(df, mask, design, y=y_full)
            if not splits:
                continue
            for model_name in model_names:
                oof = np.full(len(sub_idx), np.nan)
                fold_of = np.empty(len(sub_idx), dtype=object)
                for sp in splits:
                    try:
                        p = _fit_predict(model_name, X, y, sp.train, sp.test, positions)
                    except Exception as exc:
                        print(f"    ! {target_key}/{design}/{model_name}/{fs_name} "
                              f"fold {sp.fold}: {type(exc).__name__}: {exc}")
                        continue
                    te_pos = np.array([positions[int(i)] for i in sp.test])
                    oof[te_pos] = p
                    fold_of[te_pos] = sp.fold

                ok = ~np.isnan(oof)
                if ok.sum() < 100 or len(np.unique(y[ok])) < 2:
                    continue

                stem = f"{target_key}__{design}__{model_name}__{fs_name}"
                pd.DataFrame({
                    "row": sub_idx[ok], "person_id": groups[ok], "fold": fold_of[ok],
                    "y": y[ok], "p": oof[ok],
                    "protocol": df["protocol"].to_numpy()[sub_idx][ok],
                    "visit_year": df["visit_year"].to_numpy()[sub_idx][ok],
                }).to_parquet(PRED_DIR / f"{stem}.parquet")

                # Bootstrap intervals are expensive; compute them for the model
                # whose intervals the paper actually reports.
                nb = n_boot if model_name in boot_models else 0
                res = E.summarise_predictions(y[ok], oof[ok], groups=groups[ok], n_boot=nb)
                res.update({
                    "target": target_key, "target_name": t.name, "module": t.module,
                    "negative_control": t.negative_control, "cost": t.cost,
                    "design": design, "model": model_name, "features": fs_name,
                    "n_folds": len(splits), "coverage_evaluated": float(ok.mean()),
                })
                rows.append(res)
                print(f"    {target_key:24s} {design:9s} {model_name:17s} {fs_name:17s} "
                      f"AUROC {res['auroc']:.3f}  AUPRC {res['auprc']:.3f} "
                      f"(base {res['prevalence']:.3f})  saved@90 {res['tests_avoided_at_sens90_%']:.0f}%")
    return rows


def _worker(target_key: str, designs, model_names, feature_sets, n_boot):
    """Runs in a separate process; loads its own copy of the data."""
    df = pd.read_parquet(C.PROCESSED / "visits.parquet")
    labels, _ = build_targets(df)
    return run_target(df, labels, target_key, designs, model_names, feature_sets, n_boot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=3,
                    help="parallel target workers; each also uses OpenMP threads")
    args = ap.parse_args()

    df = pd.read_parquet(C.PROCESSED / "visits.parquet")
    labels, _ = build_targets(df)

    keys = args.targets or [t.key for t in TARGETS]
    designs = ["person"] if args.quick else MAIN_DESIGNS
    n_boot = 0 if args.quick else args.n_boot

    def dispatch(model_names, feature_sets, banner):
        print("=" * 104, flush=True)
        print(banner, flush=True)
        print("=" * 104, flush=True)
        if args.jobs > 1 and len(keys) > 1:
            from joblib import Parallel, delayed
            batches = Parallel(n_jobs=args.jobs, backend="loky", verbose=5)(
                delayed(_worker)(k, designs, model_names, feature_sets, n_boot) for k in keys)
            return [r for b in batches for r in b]
        rows = []
        for k in keys:
            rows += run_target(df, labels, k, designs, model_names, feature_sets, n_boot)
        return rows

    main_rows = dispatch(MAIN_MODELS, {"D_full_core": LADDER["D_full_core"]},
                         "MAIN: all models x designs, full core panel")
    if main_rows:
        out = pd.DataFrame(main_rows)
        out.to_csv(C.TABLES / "main_results.csv", index=False, encoding="utf-8-sig")
        print(f"\nwrote {C.TABLES/'main_results.csv'}  ({len(out)} rows)", flush=True)

    abl_rows = dispatch(["gbdt"], LADDER, "ABLATION: feature-block ladder (gbdt)")
    if abl_rows:
        out = pd.DataFrame(abl_rows)
        out.to_csv(C.TABLES / "ablation.csv", index=False, encoding="utf-8-sig")
        print(f"\nwrote {C.TABLES/'ablation.csv'}  ({len(out)} rows)", flush=True)


if __name__ == "__main__":
    main()
