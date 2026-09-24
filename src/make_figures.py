"""Publication figures.

Run:  python -m src.make_figures

Fig 1  module coverage heatmap by employer      -- motivates the design
Fig 2  ablation ladder across targets           -- the lift is in the chemistry
Fig 3  decision curves for the primary targets  -- is triage worth doing
Fig 4  tests avoided vs sensitivity             -- the operating trade-off
Fig 5  transfer: person vs protocol vs temporal -- does it survive a new protocol
Fig 6  calibration curves                       -- per design
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config as C
from . import evaluate as E
from .targets import TARGETS, TARGETS_BY_KEY

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 320, "savefig.bbox": "tight",
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "text.color": "#0b0b0b", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.edgecolor": "#c9c8c2",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#d8d7d1", "grid.alpha": 0.7,
    "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "lines.linewidth": 2, "lines.markersize": 5,
})

# Reference categorical palette, light mode, assigned in fixed slot order.
# Validated: worst adjacent CVD dE 9.1, normal-vision dE 19.6 (see dataviz
# validate_palette.js). Three slots fall below 3:1 contrast on this surface, so
# every figure ships a legend and the numbers appear in results/REPORT.md tables.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
ACCENT = SERIES[0]
WARM = SERIES[1]
MUTED = "#8a8a84"
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"])

# Colour follows the finding, not its rank in a sorted list, so a figure that
# drops or reorders targets never repaints the survivors.
COLOR = {
    "dexa_low_bone_mass": SERIES[0],
    "us_nephrolithiasis": SERIES[1],
    "prostate_heterogeneous": SERIES[2],
    "k6_severe_distress": SERIES[3],
    "us_kidney_stone": SERIES[4],
    "echo_mitral_abnormal": SERIES[5],
    "echo_mr_mild_plus": SERIES[6],
    "us_gallstones": SERIES[7],
}
MARKER = {"dexa_low_bone_mass": "o", "us_nephrolithiasis": "s",
          "prostate_heterogeneous": "^", "k6_severe_distress": "D",
          "us_kidney_stone": "v", "us_gallstones": "X"}


def _style(key):
    """Colour, marker and dash for a target; negative controls are dashed."""
    neg = TARGETS_BY_KEY[key].negative_control if key in TARGETS_BY_KEY else False
    return dict(color=COLOR.get(key, SERIES[0]), marker=MARKER.get(key, "o"),
                ls="--" if neg else "-")

PRED = C.RESULTS / "predictions"
PRIMARY = ["dexa_low_bone_mass", "us_nephrolithiasis", "prostate_heterogeneous",
           "k6_severe_distress", "us_gallstones"]
SHORT = {
    "dexa_low_bone_mass": "Low bone mass\n(DEXA)",
    "us_nephrolithiasis": "Nephrolithiasis\n(abdominal US)",
    "prostate_heterogeneous": "Heterogeneous\nprostate (US)",
    "us_gallstones": "Gallstones\n(negative control)",
    "k6_severe_distress": "Severe distress\n(K6)",
    "echo_mitral_abnormal": "Mitral valve\nabnormality (echo)",
    "echo_mr_mild_plus": "MR >= mild (echo)",
    "us_kidney_stone": "Kidney stone (US)",
    "us_hydronephrosis": "Hydronephrosis (US)",
    "carotid_abnormal": "Carotid abnormality",
    "mammo_birads_3plus": "BI-RADS >= 3",
    "tox_any_positive": "Positive drug screen",
}


def _load(target, design, model, fs="D_full_core"):
    p = PRED / f"{target}__{design}__{model}__{fs}.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ------------------------------------------------------------------- figure 1
def fig_coverage():
    f = C.TABLES / "module_coverage_by_employer.csv"
    if not f.exists():
        return
    d = pd.read_csv(f).set_index("protocol")
    n = d.pop("n_visits")
    keys = [t.key for t in TARGETS if t.key in d.columns]
    d = d[keys]
    d.index = [f"{i[:26]}  (n={int(v):,})" for i, v in zip(d.index, n)]

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    im = ax.imshow(d.values, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([SHORT.get(k, k).replace("\n", " ") for k in keys],
                       rotation=38, ha="right")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.index, fontsize=6.8)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.6,
                    color="white" if v > 55 else INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=.62, pad=.015)
    cb.set_label("% of visits receiving the module", fontsize=8)
    ax.set_title("Which expensive modules you receive is set by your employer's contract,\n"
                 "not by your health", loc="left", fontweight="bold", pad=10)
    fig.savefig(C.FIGURES / "fig1_module_coverage.png")
    plt.close(fig)
    print("  fig1_module_coverage.png")


# ------------------------------------------------------------------- figure 2
def fig_ablation(design="protocol"):
    f = C.TABLES / "ablation.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = d[(d.design == design) & (d.model == "gbdt")]
    if d.empty:
        design = "person"
        d = pd.read_csv(f)
        d = d[(d.design == design) & (d.model == "gbdt")]
    order = ["A_demographics", "B_plus_vitals", "C_plus_chemistry", "D_full_core"]
    pretty = ["Age + sex", "+ BMI, blood\npressure", "+ blood\nchemistry", "+ haematology\n(full core)"]
    keys = [k for k in PRIMARY if k in set(d.target)]
    if not keys:
        return

    metric_col = "macro_auroc" if "macro_auroc" in d.columns else "auroc"
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, metric, lab in [(axes[0], metric_col, "AUROC (mean across held-out employers)"),
                            (axes[1], "auprc_lift", "AUPRC / prevalence  (lift over chance)")]:
        for k in keys:
            s = d[d.target == k].set_index("features").reindex(order)[metric]
            ax.plot(range(len(order)), s.values, ms=5, **_style(k),
                    label=SHORT.get(k, k).replace("\n", " ")
                          + (" — negative control" if TARGETS_BY_KEY[k].negative_control else ""))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(pretty, fontsize=7)
        ax.set_ylabel(lab)
        ax.axhline(.5 if metric == metric_col else 1, color=MUTED, lw=.9, ls=":")
    axes[0].set_title("Blood chemistry, not demography, carries the cross-modal signal",
                      loc="left", fontweight="bold")
    axes[1].legend(loc="upper left", ncol=1)
    fig.savefig(C.FIGURES / "fig2_ablation.png")
    plt.close(fig)
    print("  fig2_ablation.png")


# ------------------------------------------------------------------- figure 3
def fig_decision_curves(design="protocol"):
    keys = [k for k in PRIMARY if (PRED / f"{k}__{design}__gbdt__D_full_core.parquet").exists()]
    if not keys:
        design = "person"
        keys = [k for k in PRIMARY if (PRED / f"{k}__{design}__gbdt__D_full_core.parquet").exists()]
    if not keys:
        return
    ncol = min(len(keys), 5)
    fig, axes = plt.subplots(1, ncol, figsize=(2.35 * ncol, 2.9), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys[:ncol]):
        d = _load(k, design, "gbdt")
        dc = E.decision_curve(d.y.values, d.p.values)
        ax.plot(dc.threshold, dc.nb_model, color=COLOR.get(k, SERIES[0]), lw=2, label="Triage on model")
        ax.plot(dc.threshold, dc.nb_test_all, color=MUTED, lw=1.2, ls="--", label="Test everyone")
        ax.axhline(0, color=INK, lw=1, ls=":", label="Test nobody")
        ax.set_ylim(min(-0.01, dc.nb_model.min()), max(dc.nb_model.max(), dc.nb_test_all.max()) * 1.15 + 1e-3)
        ax.set_title(SHORT.get(k, k), fontsize=8)
        ax.set_xlabel("Risk threshold")
    axes[0].set_ylabel("Net benefit")
    axes[0].legend(loc="upper right")
    fig.suptitle("Decision-curve analysis (leave-one-employer-out): triage adds net benefit "
                 "for three modules,\nand none at all for the other two",
                 x=.005, ha="left", fontweight="bold", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, .84])
    fig.savefig(C.FIGURES / "fig3_decision_curves.png")
    plt.close(fig)
    print("  fig3_decision_curves.png")


# ------------------------------------------------------------------- figure 4
def fig_tests_avoided(design="protocol"):
    keys = [k for k in PRIMARY if (PRED / f"{k}__{design}__gbdt__D_full_core.parquet").exists()]
    if not keys:
        design = "person"
        keys = [k for k in PRIMARY if (PRED / f"{k}__{design}__gbdt__D_full_core.parquet").exists()]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    sens = np.arange(.70, .995, .01)
    for k in keys:
        d = _load(k, design, "gbdt")
        vals = [E.tests_avoided_at_sensitivity(d.y.values, d.p.values, s)["tests_avoided_%"] for s in sens]
        st = _style(k); st.pop("marker")
        ax.plot(sens * 100, vals, **st,
                label=SHORT.get(k, k).replace("\n", " ")
                      + (" — neg. control" if TARGETS_BY_KEY[k].negative_control else ""))
    ax.set_xlabel("Sensitivity the programme insists on (% of abnormal findings caught)")
    ax.set_ylabel("% of scans that can be skipped")
    ax.set_title("How much imaging a risk-based referral rule saves", loc="left", fontweight="bold")
    ax.legend(loc="upper right")
    ax.invert_xaxis()
    fig.savefig(C.FIGURES / "fig4_tests_avoided.png")
    plt.close(fig)
    print("  fig4_tests_avoided.png")


# ------------------------------------------------------------------- figure 5
def fig_transfer():
    f = C.TABLES / "main_results.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = d[(d.model == "gbdt") & (d.features == "D_full_core")]
    designs = ["person", "protocol", "temporal"]
    keys = [k for k in PRIMARY if k in set(d.target)]
    if not keys:
        return
    x = np.arange(len(keys)); w = .26
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    metric_col = "macro_auroc" if "macro_auroc" in d.columns else "auroc"
    for i, des in enumerate(designs):
        sub = d[d.design == des].set_index("target").reindex(keys)
        vals = sub[metric_col].values
        err = None
        if {"auroc_lo", "auroc_hi"} <= set(sub.columns) and sub["auroc_lo"].notna().any() \
                and metric_col == "auroc":
            err = np.vstack([vals - sub["auroc_lo"].values, sub["auroc_hi"].values - vals])
            err = np.clip(np.nan_to_num(err), 0, None)
        ax.bar(x + (i - 1) * w, vals, w, yerr=err, capsize=2,
               label={"person": "Grouped by person (internal)",
                      "protocol": "Leave-one-employer-out",
                      "temporal": "Train early years -> test late"}[des],
               color=["#9ec5f4", SERIES[0], "#0d366b"][i],
               edgecolor="#fcfcfb", lw=.8)
    ax.axhline(.5, color=INK, lw=.9, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(k, k) for k in keys], fontsize=7)
    ax.set_ylabel("AUROC")
    ax.set_ylim(.45, 1.0)
    ax.legend(loc="upper right", ncol=1)
    ax.set_title("Transfer to an unseen screening protocol, by validation design",
                 loc="left", fontweight="bold")
    fig.savefig(C.FIGURES / "fig5_transfer.png")
    plt.close(fig)
    print("  fig5_transfer.png")


# ------------------------------------------------------------------- figure 6
def fig_calibration(design="protocol"):
    keys = [k for k in PRIMARY if (PRED / f"{k}__{design}__gbdt__D_full_core.parquet").exists()]
    if not keys:
        design = "person"
        keys = [k for k in PRIMARY if (PRED / f"{k}__{design}__gbdt__D_full_core.parquet").exists()]
    if not keys:
        return
    ncol = min(len(keys), 5)
    fig, axes = plt.subplots(1, ncol, figsize=(2.2 * ncol, 2.5))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys[:ncol]):
        d = _load(k, design, "gbdt")
        q = pd.qcut(d.p, 10, duplicates="drop")
        g = d.groupby(q, observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
        lim = max(g.pred.max(), g.obs.max()) * 1.12
        ax.plot([0, lim], [0, lim], color=MUTED, ls=":", lw=1)
        ax.plot(g.pred, g.obs, marker="o", ms=4, lw=1.6, color=COLOR.get(k, SERIES[0]))
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title(SHORT.get(k, k), fontsize=8)
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("Observed")
    fig.suptitle(f"Calibration by decile of predicted risk ({design} design)",
                 x=.005, ha="left", fontweight="bold", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, .88])
    fig.savefig(C.FIGURES / "fig6_calibration.png")
    plt.close(fig)
    print("  fig6_calibration.png")


# ------------------------------------------------------------------- figure 7
def fig_fold_spread():
    """Per-held-out-employer AUROC, with the pooled figure marked.

    The gap between the pooled estimate and the fold-wise mean is a diagnostic
    for label drift: where a finding is reported to different thresholds at
    different sites, prevalence varies between folds and the pooled AUROC is
    inflated by that between-site variation rather than by within-site skill.
    """
    keys = [k for k in ["prostate_heterogeneous", "dexa_low_bone_mass", "us_kidney_stone",
                        "us_nephrolithiasis", "us_gallstones", "k6_severe_distress"]
            if (PRED / f"{k}__protocol__gbdt__D_full_core.parquet").exists()]
    if not keys:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                  gridspec_kw={"width_ratios": [1.45, 1]})
    rng = np.random.default_rng(0)
    for i, k in enumerate(keys):
        d = _load(k, "protocol", "gbdt")
        vals, prevs = [], []
        for _, g in d.groupby("fold"):
            if len(g) < 100 or g.y.sum() < 10 or g.y.nunique() < 2:
                continue
            vals.append(E.auroc(g.y, g.p)); prevs.append(g.y.mean() * 100)
        if not vals:
            continue
        col = COLOR.get(k, SERIES[0])
        ax.scatter(np.array(vals), i + rng.uniform(-.15, .15, len(vals)),
                   s=22, color=col, alpha=.75, edgecolor="#fcfcfb", lw=.5, zorder=3)
        ax.scatter([np.mean(vals)], [i], marker="|", s=430, color=INK, lw=2.2, zorder=4)
        ax.scatter([E.auroc(d.y, d.p)], [i], marker="D", s=28, facecolor="#fcfcfb",
                   edgecolor=INK, lw=1.3, zorder=5)
        ax2.scatter(prevs, [i] * len(prevs), s=22, color=col, alpha=.75,
                    edgecolor="#fcfcfb", lw=.5)
    ax.axvline(.5, color=MUTED, lw=1, ls=":")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([SHORT.get(k, k).replace("\n", " ")
                        + (" *" if TARGETS_BY_KEY[k].negative_control else "")
                        for k in keys], fontsize=7.5)
    ax.set_xlabel("AUROC in each held-out employer      (* negative control)")
    ax.set_xlim(.35, 1.0)
    ax.set_ylim(len(keys) - 0.4, -0.9)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=MUTED, ms=5, label="one held-out employer"),
        Line2D([], [], marker="|", ls="", color=INK, ms=11, mew=2, label="fold-wise mean"),
        Line2D([], [], marker="D", ls="", mfc="#fcfcfb", mec=INK, ms=5, label="pooled estimate"),
    ], loc="upper left", fontsize=7, ncol=3, columnspacing=1.1, handletextpad=.4)
    ax2.set_yticks(range(len(keys))); ax2.set_yticklabels([])
    ax2.set_xlabel("Prevalence in each held-out employer (%)")
    ax2.set_ylim(len(keys) - 0.4, -0.9)
    fig.suptitle("Where the pooled estimate overstates transfer: findings whose prevalence\n"
                 "swings between sites are reported to site-specific thresholds",
                 x=.005, ha="left", fontweight="bold", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, .86])
    fig.savefig(C.FIGURES / "fig7_fold_spread.png")
    plt.close(fig)
    print("  fig7_fold_spread.png")


def main():
    print("writing figures ->", C.FIGURES)
    for fn in (fig_coverage, fig_ablation, fig_decision_curves,
               fig_tests_avoided, fig_transfer, fig_calibration, fig_fold_spread):
        try:
            fn()
        except Exception as exc:
            print(f"  ! {fn.__name__}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
