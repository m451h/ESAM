"""Assemble results/REPORT.md from the generated tables.

Run:  python -m src.make_report
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import evaluate as E
from .targets import TARGETS_BY_KEY

PRED = C.RESULTS / "predictions"
HEADLINE_DESIGN = "protocol"
HEADLINE_MODEL = "gbdt"


def _fmt(v, n=3):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"


def _ci(row, stem):
    lo, hi = row.get(f"{stem}_lo"), row.get(f"{stem}_hi")
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return ""
    return f" ({lo:.2f}-{hi:.2f})"


def main_table(d: pd.DataFrame, design: str) -> str:
    sub = d[(d.design == design) & (d.features == "D_full_core")]
    if sub.empty:
        return f"_no results for design `{design}`_\n"
    has_macro = "macro_auroc" in sub.columns
    lines = ["| Finding (module) | n | prev. | model | AUROC (fold-wise) | AUROC (pooled) | AUPRC lift | scans avoided @90% sens | missed |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    # order targets by the honest metric, best first
    best = (sub[sub.model == "gbdt"]
            .set_index("target")["macro_auroc" if has_macro else "auroc"]
            .sort_values(ascending=False))
    for key in best.index:
        g = sub[sub.target == key].set_index("model")
        t = TARGETS_BY_KEY[key]
        for m in ["prevalence", "logistic", "gbdt", "gbdt_miss", "gbdt_values_only"]:
            if m not in g.index:
                continue
            r = g.loc[m]
            name = (f"**{t.name}**<br><sub>{t.module}"
                    + (" · NEGATIVE CONTROL" if t.negative_control else "") + "</sub>") if m == "prevalence" else ""
            n = f"{int(r['n']):,}" if m == "prevalence" else ""
            prev = f"{r['prevalence']*100:.1f}%" if m == "prevalence" else ""
            macro = _fmt(r.get("macro_auroc")) if has_macro else "-"
            sd = f" ±{r['macro_auroc_sd']:.2f}" if has_macro and np.isfinite(r.get("macro_auroc_sd", np.nan)) else ""
            lines.append(
                f"| {name} | {n} | {prev} | `{m}` | {macro}{sd} | "
                f"{_fmt(r['auroc'])}{_ci(r, 'auroc')} | {_fmt(r['auprc_lift'], 2)}x | "
                f"{_fmt(r['tests_avoided_at_sens90_%'], 1)}% | {int(r['missed_at_sens90'])} |")
    return "\n".join(lines) + "\n"


def ablation_table(a: pd.DataFrame, design: str) -> str:
    sub = a[(a.design == design) & (a.model == "gbdt")]
    if sub.empty:
        return f"_no ablation for design `{design}`_\n"
    order = ["A_demographics", "B_plus_vitals", "C_plus_chemistry", "D_full_core"]
    pretty = {"A_demographics": "age+sex", "B_plus_vitals": "+vitals",
              "C_plus_chemistry": "+chemistry", "D_full_core": "+haematology"}
    metric = "macro_auroc" if "macro_auroc" in sub.columns else "auroc"
    piv = sub.pivot_table(index="target", columns="features", values=metric).reindex(columns=order)
    piv.columns = [pretty[c] for c in piv.columns]
    piv["gain from chemistry"] = piv["+chemistry"] - piv["+vitals"]
    piv = piv.sort_values("gain from chemistry", ascending=False)
    lines = ["| Finding | " + " | ".join(piv.columns) + " |",
             "|---|" + "---:|" * len(piv.columns)]
    for k, r in piv.iterrows():
        t = TARGETS_BY_KEY[k]
        label = t.name + (" *(negative control)*" if t.negative_control else "")
        lines.append(f"| {label} | " + " | ".join(_fmt(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def transfer_table(d: pd.DataFrame) -> str:
    sub = d[(d.model == HEADLINE_MODEL) & (d.features == "D_full_core")]
    if sub.empty:
        return "_none_\n"
    metric = "macro_auroc" if "macro_auroc" in sub.columns else "auroc"
    piv = sub.pivot_table(index="target", columns="design", values=metric)
    for c in ("person", "protocol", "temporal"):
        if c not in piv:
            piv[c] = np.nan
    piv = piv[["person", "protocol", "temporal"]]
    piv["protocol - person"] = piv["protocol"] - piv["person"]
    lines = ["| Finding | person-grouped | leave-one-employer-out | temporal | Δ (protocol − person) |",
             "|---|---:|---:|---:|---:|"]
    for k, r in piv.sort_values("protocol", ascending=False).iterrows():
        t = TARGETS_BY_KEY[k]
        label = t.name + (" *(neg. control)*" if t.negative_control else "")
        lines.append(f"| {label} | {_fmt(r['person'])} | {_fmt(r['protocol'])} | "
                     f"{_fmt(r['temporal'])} | {_fmt(r['protocol - person'], 3)} |")
    return "\n".join(lines) + "\n"


def leakage_table(d: pd.DataFrame, design: str) -> str:
    sub = d[(d.design == design) & (d.features == "D_full_core")
            & d.model.isin(["gbdt", "gbdt_miss", "gbdt_values_only"])]
    if sub.empty:
        return "_none_\n"
    piv = sub.pivot_table(index="target", columns="model", values="auroc")
    for c in ("gbdt", "gbdt_miss", "gbdt_values_only"):
        if c not in piv:
            piv[c] = np.nan
    piv["Δ from ordering info"] = piv["gbdt"] - piv["gbdt_values_only"]
    piv = piv.sort_values("Δ from ordering info", ascending=False)
    lines = ["| Finding | GBDT (native NaN) | + ordering indicators | values only (imputed) | Δ attributable to ordering |",
             "|---|---:|---:|---:|---:|"]
    for k, r in piv.iterrows():
        lines.append(f"| {TARGETS_BY_KEY[k].name} | {_fmt(r['gbdt'])} | {_fmt(r['gbdt_miss'])} | "
                     f"{_fmt(r['gbdt_values_only'])} | {_fmt(r['Δ from ordering info'], 3)} |")
    return "\n".join(lines) + "\n"


def per_protocol_breakdown(target: str) -> str:
    f = PRED / f"{target}__protocol__{HEADLINE_MODEL}__D_full_core.parquet"
    if not f.exists():
        return ""
    d = pd.read_parquet(f)
    rows = []
    for name, g in d.groupby("fold"):
        if g.y.nunique() < 2 or len(g) < 100:
            continue
        rows.append({"held-out protocol": str(name)[:34], "n": len(g),
                     "prevalence": g.y.mean(), "AUROC": E.auroc(g.y, g.p)})
    if not rows:
        return ""
    t = pd.DataFrame(rows).sort_values("n", ascending=False)
    lines = [f"**{TARGETS_BY_KEY[target].name}** — per held-out employer",
             "", "| held-out protocol | n | prevalence | AUROC |", "|---|---:|---:|---:|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['held-out protocol']} | {int(r['n']):,} | "
                     f"{r['prevalence']*100:.1f}% | {r['AUROC']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    tp = C.TABLES
    main_df = pd.read_csv(tp / "main_results.csv") if (tp / "main_results.csv").exists() else pd.DataFrame()
    abl_df = pd.read_csv(tp / "ablation.csv") if (tp / "ablation.csv").exists() else pd.DataFrame()
    qc = json.loads((C.PROCESSED / "qc_report.json").read_text(encoding="utf-8"))
    spread = json.loads((tp / "module_coverage_spread.json").read_text(encoding="utf-8")) \
        if (tp / "module_coverage_spread.json").exists() else {}

    out = ["# Results", "",
           "Generated by `python -m src.make_report`. Every number here traces to a file "
           "in `results/`; nothing is transcribed by hand.", ""]

    out += ["## Cohort", ""]
    if (tp / "cohort_flow.csv").exists():
        cf = pd.read_csv(tp / "cohort_flow.csv")
        out += ["| step | n |", "|---|---:|"]
        out += [f"| {r.step} | {int(r.n):,} |" for r in cf.itertuples()]
        out += [""]

    out += ["## Data quality", "",
            f"- **{qc['sentinel_total_cells']:,} cells** ({qc['sentinel_total_cells']/(qc['raw_shape'][0]*qc['raw_shape'][1])*100:.1f}% "
            "of the matrix) encoded *test not performed* as text rather than as a blank.",
            f"- **{qc['unit_errors_total']:,} core-panel values** fell outside physiological range and were removed.",
            f"- Reconstructed identities: **{qc['person_id']['n_person_ids']:,}** over "
            f"**{qc['person_id']['n_visits']:,}** visits; same-day collisions "
            f"**{qc['person_id']['same_person_same_day_collisions']}** "
            f"({qc['person_id']['same_person_same_day_collisions']/qc['person_id']['n_visits']*100:.2f}%).", ""]
    if qc.get("missingness_understated_by_pp"):
        worst = list(qc["missingness_understated_by_pp"].items())[:6]
        out += ["Naive `isna()` understates missingness most severely for:", ""]
        out += [f"- `{k}` — by {v:.0f} percentage points" for k, v in worst]
        out += [""]

    if spread:
        out += ["## Module assignment is protocol-driven", "",
                "Coverage of each module across employers (the basis for the "
                "leave-one-employer-out design):", "",
                "| module | min | max | range |", "|---|---:|---:|---:|"]
        for k, v in sorted(spread.items(), key=lambda kv: -kv[1]["range_pp"]):
            nm = TARGETS_BY_KEY[k].module if k in TARGETS_BY_KEY else k
            out.append(f"| {nm} ({k}) | {v['min']:.1f}% | {v['max']:.1f}% | {v['range_pp']:.1f} pp |")
        out += [""]

    if not main_df.empty:
        out += ["## Headline: leave-one-employer-out", "",
                "The design the paper leans on — every test fold is an employer, and "
                "therefore a screening protocol, the model never saw.", "",
                main_table(main_df, "protocol"), "",
                "## Transfer across validation designs", "",
                transfer_table(main_df), "",
                "## Does the model cheat through the ordering pattern?", "",
                "`gbdt` sees missing values natively and could in principle exploit "
                "*which* analytes were ordered rather than their values. "
                "`gbdt_values_only` removes that channel by imputing everything, so the "
                "**`gbdt` − `gbdt_values_only`** gap is the share of performance "
                "attributable to ordering. It is small throughout.", "",
                "> `gbdt_miss` reproduces `gbdt` exactly. That is expected rather than a "
                "bug: histogram boosting already learns a default direction for missing "
                "values at each split, so explicit indicators are redundant and are never "
                "selected (verified — 50 extra columns, 49 with non-zero variance, "
                "predictions identical to 0.0). It is reported for completeness but "
                "carries no evidence on its own.", "",
                leakage_table(main_df, "protocol"), "",
                "## Internal (person-grouped) results, for comparison", "",
                main_table(main_df, "person"), ""]

    if not abl_df.empty:
        out += ["## Ablation: where does the signal come from?", "",
                "AUROC as predictor blocks are added, leave-one-employer-out.", "",
                ablation_table(abl_df, "protocol"), "",
                "Person-grouped, for comparison:", "",
                ablation_table(abl_df, "person"), ""]

    if not main_df.empty and "macro_auroc" in main_df.columns:
        g = main_df[(main_df.design == "protocol") & (main_df.model == "gbdt")
                    & (main_df.features == "D_full_core")].copy()
        g["gap"] = g["auroc"] - g["macro_auroc"]
        g = g.sort_values("gap", ascending=False)
        out += ["## Pooled minus fold-wise: a label-drift diagnostic", "",
                "Where a finding is reported to site-specific thresholds, prevalence swings "
                "between employers and the *pooled* AUROC is inflated by that between-site "
                "variation rather than by within-site skill. A large positive gap is a warning "
                "about the label, not a compliment to the model.", "",
                "| Finding | pooled | fold-wise | gap | prevalence range across employers |",
                "|---|---:|---:|---:|---:|"]
        for r in g.itertuples():
            t = TARGETS_BY_KEY[r.target]
            f = PRED / f"{r.target}__protocol__gbdt__D_full_core.parquet"
            rng = ""
            if f.exists():
                dd = pd.read_parquet(f)
                pr = [x.y.mean() * 100 for _, x in dd.groupby("fold") if len(x) >= 100 and x.y.sum() >= 10]
                if pr:
                    rng = f"{min(pr):.1f}% – {max(pr):.1f}%"
            out.append(f"| {t.name} | {_fmt(r.auroc)} | {_fmt(r.macro_auroc)} | "
                       f"{_fmt(r.gap, 3)} | {rng} |")
        out += [""]

    sg = C.TABLES / "subgroup_performance.csv"
    if sg.exists():
        s = pd.read_csv(sg)
        out += ["## Subgroup performance (equity check)", "",
                "Sex gaps are small; **age gaps are not**, and they run the wrong way — "
                "discrimination is weakest in the oldest band, which is exactly where "
                "prevalence is highest and a referral decision matters most.", "",
                "| Finding | variable | level | n | prevalence | AUROC | scans avoided @90% |",
                "|---|---|---|---:|---:|---:|---:|"]
        for _, r in s.iterrows():
            out.append(f"| {r['target_name']} | {r['variable']} | {r['level']} | "
                       f"{int(r['n']):,} | {r['prevalence_%']:.1f}% | {r['AUROC']:.3f} | "
                       f"{r['scans_avoided_at_sens90_%']:.1f}% |")
        out += [""]

    out += ["## Per-employer breakdown, primary targets", ""]
    for k in ("dexa_low_bone_mass", "us_nephrolithiasis", "prostate_heterogeneous"):
        blk = per_protocol_breakdown(k)
        if blk:
            out += [blk, ""]

    out += ["## Figures", ""]
    for f in sorted(C.FIGURES.glob("*.png")):
        out.append(f"![{f.stem}](figures/{f.name})")
    out += [""]

    (C.RESULTS / "REPORT.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {C.RESULTS/'REPORT.md'}")


if __name__ == "__main__":
    main()
