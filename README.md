# Equity-Oriented Screening Allocation Model

Can the routine blood panel every screening attendee already receives tell us who actually needs the expensive modules — bone densitometry, abdominal ultrasound, prostate ultrasound?

This repository contains the full analysis pipeline for a study using data from an Iranian corporate occupational health-screening programme: **36,885 visits by 12,246 individuals** across 8 programme years, with 592 recorded fields spanning ~14 modalities.

The key insight: which expensive modules a person receives is determined by their employer's screening contract, not by their health. DEXA coverage ranges from 0.0% to 91.8% across employers. That natural variation enables a leave-one-employer-out validation design where the model is tested on an entire screening protocol it never saw during training.

## Key Results

| Target | AUROC (leave-one-employer-out) | Scans avoided @ 90% sensitivity |
|---|---:|---:|
| Low bone mass (DEXA) | 0.733 | 33% (median) |
| Heterogeneous prostate (US) | 0.775 | 50% (median) |
| Kidney stone (US) | 0.663 | 25% (median) |
| Nephrolithiasis (US) | 0.649 | 37% (median) |
| Severe psychological distress (K6) | 0.660 | 23% (median) |

The deployed model (`gbdt_values_only`) uses only imputed lab values — no ordering-pattern leakage. Cost vs. the full GBDT: ≤ 0.02 AUROC.

## Quick Start

```bash
# Clone
git clone https://github.com/m451h/ESAM.git
cd ESAM

# Install
python -m pip install -r requirements.txt

# Run the pipeline
python -m src.clean              # ~3 min: parse export -> data/processed/visits.parquet
python -m src.run_experiments --jobs 3   # main run (~1 h on 8 cores)
python -m src.checks             # integrity checks
python -m src.make_figures       # publication figures
python -m src.make_report        # -> results/REPORT.md
```

Smoke test (~30 s):

```bash
python -m src.run_experiments --quick --targets dexa_low_bone_mass us_gallstones
```

## Project Structure

```
src/
  config.py            paths, column groups, plausible ranges
  clean.py             raw export -> visits.parquet + QC report
  features.py          50-feature "cheap core" panel
  targets.py           12 expensive-module outcomes + eligibility masks
  splits.py            person / leave-one-employer-out / temporal designs
  models.py            baselines through GBDT, incl. missingness-leakage arms
  evaluate.py          AUROC/AUPRC, calibration, net benefit
  run_experiments.py   experiment driver
  describe_cohort.py   cohort flow, Table 1, coverage matrices
  subgroups.py         performance by sex/age, equity analysis
  checks.py            integrity checks
  make_figures.py      publication figures
  make_report.py       assembles results/REPORT.md
  deploy.py            input contract: 43 named inputs with units and ranges
  fit_final.py         fits + serialises shippable models
  predict.py           scores new attendees against a fitted model
models/                <target>.joblib + <target>.card.json  (generated)
data/processed/        visits.parquet, qc_report.json       (generated)
results/               tables, figures, predictions          (generated)
```

## Scoring New Attendees

```bash
python -m src.fit_final          # -> models/*.joblib + model cards
python -m src.predict --inputs   # show the 43 inputs with units and ranges
python -m src.predict -t dexa_low_bone_mass -i attendees.csv -o triaged.csv
python -m src.predict -t dexa_low_bone_mass --json '{"age":58,"sex":"female",...}'
```

Only the two targets that pass the negative control (gallstones) are fitted by default. Others require `--all` and are marked `not_validated`.

## Validation Design

Three validation designs of increasing external-validity demand:

1. **Person-grouped** — same individuals, different visits (internal)
2. **Leave-one-employer-out** — entire screening protocol held out (the main result)
3. **Temporal** — train on early years, test on later years

All reported intervals use person-clustered bootstrap.

## Built-in Controls

- **Negative control target**: Gallstones — risk is largely not encoded in blood chemistry. If the model "predicts everything," it is fitting site artefacts.
- **Missingness-leakage bracket**: Comparing `gbdt` (native NaN), `gbdt_miss` (explicit ordering indicators), and `gbdt_values_only` (imputed) separates skill in measured values from skill in which tests were ordered.

## Data Notes

Key pitfalls discovered during development (handled in `src/clean.py`):

- "Test not performed" stored as text (`not registered`, Farsi equivalents) in 8% of cells — naive `isna()` massively understates missingness
- Unit-entry errors: platelets recorded up to 150,000, ferritin to 91,118
- DEXA values split across category columns — recovered by `clean.coalesce_dexa`
- No patient identifier — identity reconstructed from (birth date, sex): 12,246 identities, 111 same-day collisions (0.30%)
- `VT D3` runs backwards (37-fold gradient opposite to physiology) — withheld from deployed model

See `results/REPORT.md` for the full data quality report.

## Citation

```bibtex
@software{equity_screening_2026,
  title  = {Equity-Oriented Screening Allocation Model},
  year   = {2026},
  url    = {https://github.com/m451h/ESAM}
}
```

## License

Choose a license (MIT, Apache 2.0, etc.) and add a `LICENSE` file before pushing.

## Ethical Note

This model predicts **findings and referral priority**, never prognosis. The programme records no mortality, no cardiovascular events, and no biopsy confirmation. The spread of performance across employers (e.g., 78–96% sensitivity for DEXA) is the honest estimate for a new site and supports a silent-mode prospective trial rather than a live referral rule.
