"""Add fold-wise (macro) metrics to the results tables, from stored predictions.

Pooling out-of-fold predictions across employers with different baseline
prevalence distorts pooled AUROC -- visibly so for the constant-prediction
baseline, which scores well below 0.5 pooled. Fold-wise averaging is the correct
summary for the leave-one-employer-out design. No models are refitted.

Run:  python -m src.augment_results
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from . import evaluate as E

PRED = C.RESULTS / "predictions"


def augment(path):
    if not path.exists():
        return None
    d = pd.read_csv(path)
    extra = []
    for r in d.itertuples():
        f = PRED / f"{r.target}__{r.design}__{r.model}__{r.features}.parquet"
        extra.append(E.macro_by_fold(pd.read_parquet(f)) if f.exists() else {})
    out = pd.concat([d, pd.DataFrame(extra, index=d.index)], axis=1)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def main():
    for name in ("main_results.csv", "ablation.csv"):
        out = augment(C.TABLES / name)
        if out is None:
            continue
        print(f"augmented {name}  ({len(out)} rows)")

    d = pd.read_csv(C.TABLES / "main_results.csv")
    s = d[(d.design == "protocol") & (d.features == "D_full_core")
          & d.model.isin(["prevalence", "logistic", "gbdt"])]
    piv = s.pivot_table(index="target", columns="model",
                        values=["auroc", "macro_auroc"]).round(3)
    print()
    print("pooled vs fold-wise AUROC, leave-one-employer-out")
    print("(the prevalence baseline shows the artefact most clearly)")
    print(piv.to_string())


if __name__ == "__main__":
    main()
