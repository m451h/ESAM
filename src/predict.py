"""Stage 3 of deployment: score new attendees.

    python -m src.predict --inputs                       # what to send
    python -m src.predict --list                         # what can be scored
    python -m src.predict -t dexa_low_bone_mass -i new_patients.csv
    python -m src.predict -t dexa_low_bone_mass --json '{"age":58,"sex":"female",...}'

Takes the cheap panel for one or many people and answers, per expensive module,
"is this person worth scanning?" -- a probability, a decision at the shipped
threshold, and the reasons the answer might be untrustworthy.

The decision is a *referral prioritisation*, not a diagnosis, and not a
prediction of anything prognostic. See paper/LIMITATIONS.md sections 2 and 9.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .deploy import BY_COLUMN, EXCLUDED_FEATURES, INPUT_SPEC, input_table, vectorize

MODEL_DIR = C.ROOT / "models"

# Below this fraction of the 44 inputs the imputer is supplying most of the
# panel and the score collapses toward the base rate.
COMPLETENESS_WARN = 0.80
COMPLETENESS_REFUSE = 0.50


def available() -> pd.DataFrame:
    rows = []
    for p in sorted(MODEL_DIR.glob("*.card.json")):
        c = json.loads(p.read_text(encoding="utf-8"))
        op = c.get("operating_point", {})
        at = op.get("at_this_threshold", {})
        rows.append({
            "target": c["target"], "module": c["module"], "status": c["status"],
            "sex": c.get("sex_restriction") or "any",
            "threshold": round(op["threshold"], 4) if "threshold" in op else None,
            "median_sens_%": round(at["sensitivity_median"] * 100) if at else None,
            "median_scans_avoided_%": round(at["scans_avoided_median_%"]) if at else None,
        })
    return pd.DataFrame(rows)


def load(target: str) -> tuple[object, dict]:
    path = MODEL_DIR / f"{target}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"no fitted model for {target!r} in {MODEL_DIR}. "
            f"Run `python -m src.fit_final` (add --all for unvalidated targets).")
    obj = joblib.load(path)
    return obj["pipeline"], obj["card"]


def triage(records, target: str, threshold: float | None = None,
           allow_unvalidated: bool = False) -> pd.DataFrame:
    """Score records against one expensive module.

    `records`: dict, list of dicts, or DataFrame keyed by the aliases in
    `src.deploy.INPUT_SPEC`. Missing inputs are allowed and imputed; how many
    were missing is reported per row.
    """
    pipe, card = load(target)
    if card["status"] != "validated" and not allow_unvalidated:
        raise ValueError(
            f"{target!r} is marked {card['status']}: {card['evidence']} "
            f"Pass allow_unvalidated=True only if you understand that.")

    op = card.get("operating_point", {})
    if threshold is None:
        if "threshold" not in op:
            raise ValueError(f"{target!r} has no operating threshold: "
                             f"{op.get('error', 'unknown reason')}")
        threshold = op["threshold"]

    X, rep = vectorize(records)
    p = pipe.predict_proba(X.reindex(columns=card["feature_order"]))[:, 1]

    sex_req = card.get("sex_restriction")
    male = X["male"]
    eligible = pd.Series(True, index=X.index)
    if sex_req == "male":
        eligible = male.eq(1)
    elif sex_req == "female":
        eligible = male.eq(0)

    out = pd.DataFrame({
        "target": target,
        "module": card["module"],
        "risk": np.round(p, 4),
        "threshold": threshold,
        "decision": np.where(p >= threshold, "REFER", "skip"),
        "completeness": np.round(rep["completeness"], 2),
    })

    baseline = card["train_prevalence"]
    out["risk_vs_baseline"] = np.round(p / baseline, 2)

    notes = []
    for i in range(len(out)):
        n = []
        if not eligible.iloc[i]:
            n.append(f"NOT ELIGIBLE: module is {sex_req}-only")
        if X["male"].isna().iloc[i]:
            n.append("sex missing or unrecognised")
        if X["age"].isna().iloc[i]:
            n.append("age missing -- the single strongest feature")
        c = rep["completeness"].iloc[i]
        if c < COMPLETENESS_REFUSE:
            n.append(f"UNRELIABLE: only {c*100:.0f}% of inputs supplied")
        elif c < COMPLETENESS_WARN:
            n.append(f"{c*100:.0f}% of inputs supplied; score is partly imputed")
        if rep["rejected"].iloc[i]:
            n.append("out of range, ignored: " + rep["rejected"].iloc[i])
        if rep["unknown_keys"].iloc[i]:
            n.append("unrecognised keys: " + rep["unknown_keys"].iloc[i])
        if rep["ignored_keys"].iloc[i]:
            n.append("accepted but withheld from the model: "
                     + rep["ignored_keys"].iloc[i])
        notes.append("; ".join(n))
    out["notes"] = notes

    out.loc[~eligible, "decision"] = "n/a"
    out.loc[rep["completeness"] < COMPLETENESS_REFUSE, "decision"] = "insufficient data"
    out["missing_inputs"] = rep["missing"]
    return out


# ------------------------------------------------------------------------ CLI
def _read(path: Path):
    if path.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported input file type: {path.suffix}")


def main():
    ap = argparse.ArgumentParser(description="Triage new attendees for expensive modules.")
    ap.add_argument("-t", "--target", help="which module to score for")
    ap.add_argument("-i", "--input", type=Path, help="CSV / JSON / XLSX / parquet of attendees")
    ap.add_argument("--json", dest="inline", help="a single attendee as inline JSON")
    ap.add_argument("--threshold", type=float, help="override the shipped threshold")
    ap.add_argument("--allow-unvalidated", action="store_true")
    ap.add_argument("-o", "--output", type=Path, help="write results to CSV")
    ap.add_argument("--list", action="store_true", help="list scoreable modules")
    ap.add_argument("--inputs", action="store_true", help="print the input contract")
    args = ap.parse_args()

    if args.inputs:
        t = input_table()
        n_used = int(t.used_by_model.sum())
        print(f"{n_used} inputs are used by the model. All are optional -- missing "
              f"values are imputed, but completeness is reported per row and low\n"
              f"completeness is flagged. Values outside the accepted range are "
              f"discarded, not clipped.\n")
        for block in ("demographics", "vitals", "chemistry", "haematology"):
            sub = t[t.block == block]
            n_b = int(sub.used_by_model.sum())
            print(f"--- {block} ({n_b}) " + "-" * (58 - len(block)))
            for _, r in sub.iterrows():
                rng = ("" if pd.isna(r.accepted_min)
                       else f"  [{r.accepted_min:g}..{r.accepted_max:g}]")
                flag = "" if r.used_by_model else "   <- WITHHELD, see below"
                print(f"  {r.alias:22s} {r.unit:12s}{rng}{flag}")
        print("\nDerived automatically, do not send: nlr, ast_alt, tg_hdl, "
              "non_hdl, transferrin_sat, pulse_pressure")
        for col, why in EXCLUDED_FEATURES.items():
            alias = BY_COLUMN[col].alias if col in BY_COLUMN else col
            print(f"\nWITHHELD -- {alias!r} is accepted and ignored:\n")
            for line in textwrap.wrap(why, 76):
                print(f"  {line}")
        return

    if args.list:
        a = available()
        if a.empty:
            print(f"no fitted models in {MODEL_DIR}. Run `python -m src.fit_final`.")
        else:
            print(a.to_string(index=False))
        return

    if not args.target:
        ap.error("--target is required (see --list)")
    if not args.input and not args.inline:
        ap.error("supply --input FILE or --json '{...}'")

    records = json.loads(args.inline) if args.inline else _read(args.input)
    res = triage(records, args.target, args.threshold, args.allow_unvalidated)

    _, card = load(args.target)
    op = card["operating_point"]
    at = op.get("at_this_threshold", {})
    print(f"\n{card['name']}  ({card['module']})")
    if at:
        print(f"At this threshold, across {op['n_employers']} held-out employers: "
              f"sensitivity {at['sensitivity_min']*100:.0f}-{at['sensitivity_max']*100:.0f}%, "
              f"scans avoided {at['scans_avoided_min_%']:.0f}-{at['scans_avoided_max_%']:.0f}%.")
    print(f"Baseline prevalence in training: {card['train_prevalence']*100:.1f}%\n")

    show = res[["risk", "risk_vs_baseline", "threshold", "decision", "completeness", "notes"]]
    print(show.to_string())
    n_refer = int((res.decision == "REFER").sum())
    print(f"\n{n_refer} of {len(res)} referred "
          f"({100*n_refer/len(res):.0f}%); {len(res)-n_refer} not.")
    print("\nReferral prioritisation only -- not a diagnosis. "
          "Validated on Iranian occupational-screening attendees (median age 37); "
          "recalibrate before use elsewhere.")

    if args.output:
        res.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
