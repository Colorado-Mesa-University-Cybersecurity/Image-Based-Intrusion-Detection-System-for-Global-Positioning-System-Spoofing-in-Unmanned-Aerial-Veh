#!/usr/bin/env python3
"""
Generate publication-ready plots for model accuracy vs. training time,
with consistent colors/markers across figures.

Notes:
  - Matplotlib 3.9.2
  - NumPy 1.26.4
"""

__author__ = "Michael A. Nuttall"
__date__ = "2025/09/10"
__license__ = "MIT"
__version__ = "0.5.0"

import math
import sys
from pathlib import Path
import colorsys
import numpy as np

import matplotlib
matplotlib.use("Agg")  # render-to-file (no GUI)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colormaps

# ---- CONFIG ----
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gps_ids.results import (
    aggregate_seeds,
    collapse_variants,
    dataset_short,
    family_for,
    load_rows,
    normalize_label,
    protocol_tag,
    results_files,
)


def _resolve_inputs():
    return results_files()


OUTDIR = Path(__file__).resolve().parent / "plots_out"

METRICS = [
    ("mcc", "Matthews Correlation Coefficient", (-1.05, 1.05)),
    ("balanced_accuracy", "Balanced Accuracy", (0.0, 1.05)),
    ("f1_macro", "Macro F1", (0.0, 1.05)),
    ("g_mean", "G-Mean of Per-Class Recall", (0.0, 1.05)),
    ("accuracy", "Accuracy", (0.0, 1.05)),
]
BASELINE_COL = "majority_baseline_accuracy"
DEGENERATE_COL = "is_degenerate"

# Color assignment policy: "model" (unique per model label) or "family"
COLOR_BY = "model"   # <-- switch to "family" if prefer one color per family

# Figure/legend
FIGSIZE = (10.5, 6.5)
PNG_DPI = 1000
MARKER_SIZE = 5.2
AX_MARGIN_XY = 0.08
FONTS = {"axes.labelsize": 10, "axes.titlesize": 10,
         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9}
plt.rcParams.update(FONTS)


def _as_float(value):
    try:
        f = float(str(value).strip())
    except Exception:
        return None
    return f if math.isfinite(f) else None


DATASET_DISPLAY = {"uav": "UAV Attack", "2d": "2D"}
BENIGN_ONLY_TEST_SET = "real_heldout_benign_plus_malicious"


def to_arrays(rows, use_overview_labels=True, metric="accuracy"):
    X, Y, M_raw, D_raw, SRC, DEG, BASE, KEY = [], [], [], [], [], [], [], []
    for r in rows:
        x = _as_float(r.get("train_time_sec", ""))
        y = _as_float(r.get(metric, ""))
        if x is None or y is None or x <= 0:
            continue
        short = normalize_label(r.get("model", "Unknown"))
        short = collapse_variants(short) if use_overview_labels else short
        short += protocol_tag(r)
        ds_short = dataset_short(r.get("dataset", "Unknown"), r.get("_dataset_key"))
        X.append(x)
        Y.append(y)
        M_raw.append(short)
        D_raw.append(ds_short)
        SRC.append(r.get("_source",""))
        DEG.append(str(r.get(DEGENERATE_COL, "")).strip().lower() in ("true", "1"))
        # The benign-only protocol has its own, much lower majority baseline over a
        # different test set; it is not comparable with the others on one axis.
        base_val = _as_float(r.get(BASELINE_COL, ""))
        if str(r.get("test_set", "")).strip() == BENIGN_ONLY_TEST_SET:
            base_val = None
        BASE.append(base_val)
        KEY.append(r.get("_dataset_key"))
    return np.array(X), np.array(Y), M_raw, D_raw, SRC, DEG, BASE, KEY


# ---- Color + marker styling (GLOBAL & CONSISTENT) ----
def distinct_colors(n):
    """Generate n distinct RGB tuples evenly spaced in HLS."""
    if n <= 20:
        # Prefer tab20 for <=20
        cmap = colormaps.get_cmap("tab20")
        return [cmap(i % cmap.N) for i in range(n)]
    # Evenly spaced hues, fixed lightness/saturation
    cols = []
    for i in range(n):
        h = i / float(n)
        r,g,b = colorsys.hls_to_rgb(h, 0.50, 0.65)  # lightness=0.5, sat=0.65
        cols.append((r,g,b,1.0))
    return cols


def build_style_maps(global_rows):
    # sets of labels for BOTH modes (overview vs variants)
    M_over = to_arrays(global_rows, use_overview_labels=True)[2]
    M_var = to_arrays(global_rows, use_overview_labels=False)[2]

    models_over = sorted(set(M_over))
    models_var = sorted(set(M_var))

    # Datasets (markers shared)
    Dg = to_arrays(global_rows, use_overview_labels=True)[3]
    datasets = sorted(set(Dg))
    markers = ["o","s","D","^","v","<",">","P","X","*","h","H"]
    marker_map = {d: markers[i % len(markers)] for i, d in enumerate(datasets)}

    if COLOR_BY.lower() == "model":
        # Unique color per model label
        cols_over = distinct_colors(len(models_over))
        color_over = {lbl: cols_over[i] for i, lbl in enumerate(models_over)}
        cols_var = distinct_colors(len(models_var))
        color_var = {lbl: cols_var[i] for i, lbl in enumerate(models_var)}
    else:
        # One color per family (old behavior), consistently applied
        fams_over = sorted(set(family_for(m) for m in models_over))
        fam_cols = distinct_colors(len(fams_over))
        fam_color = {fam: fam_cols[i] for i, fam in enumerate(fams_over)}
        color_over = {lbl: fam_color[family_for(lbl)] for lbl in models_over}

        fams_var = sorted(set(family_for(m) for m in models_var))
        fam_cols2 = distinct_colors(len(fams_var))
        fam_color2 = {fam: fam_cols2[i] for i, fam in enumerate(fams_var)}
        color_var = {lbl: fam_color2[family_for(lbl)] for lbl in models_var}

    return color_over, color_var, marker_map


def _legend_layout(models_here, datasets_here):
    n_m, n_d = len(models_here), len(datasets_here)
    if n_m > 18 or n_d > 10:
        right = 0.55
    elif n_m > 12 or n_d > 8:
        right = 0.62
    else:
        right = 0.70
    model_ncol = 2 if n_m > 18 else 1
    ds_ncol = 2 if n_d > 10 else 1
    return right, model_ncol, ds_ncol


def scatter_figure(rows_subset, out_path, title,
                   color_map, marker_map,
                   use_overview_labels=True,
                   trim_percentiles=None,
                   metric="accuracy", metric_label="Accuracy", ylim=None):
    X, Y, M, D, _, DEG, BASE, KEY = to_arrays(
        rows_subset, use_overview_labels=use_overview_labels, metric=metric
    )
    if len(X) == 0:
        print(f"No points for {out_path.name}")
        return

    xlim = None
    if trim_percentiles:
        lo, hi = np.percentile(X, trim_percentiles)
        lo = max(lo, 1e-9)
        if hi > lo:
            xlim = (lo, hi)

    models_here = sorted(set(M))
    datasets_here = sorted(set(D))
    right, model_ncol, ds_ncol = _legend_layout(models_here, datasets_here)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for m_lbl in models_here:
        for d_lbl in datasets_here:
            idx = [i for i, (mm,dd) in enumerate(zip(M,D)) if mm==m_lbl and dd==d_lbl]
            if not idx:
                continue
            xs = X[idx]
            ys = Y[idx]
            ax.plot(xs, ys, linestyle="None",
                    marker=marker_map.get(d_lbl, "o"),
                    markersize=MARKER_SIZE,
                    markerfacecolor=color_map.get(m_lbl, (0.2,0.2,0.2)),
                    markeredgecolor="black", markeredgewidth=0.3, alpha=0.9)

    if metric == "accuracy":
        # Group the majority-class baselines by the real dataset a run belongs to,
        # not by the dataset label, so the generated tables share their parent's
        # baseline. One dashed line is drawn per distinct value; the annotation
        # shows a range when a dataset has more than one (its test sets differ).
        per_key = {}
        for k, b in zip(KEY, BASE):
            if b is not None and k:
                per_key.setdefault(k, set()).add(round(float(b), 4))
        for k in sorted(per_key):
            vals = sorted(per_key[k])
            for b in vals:
                ax.axhline(b, linestyle="--", linewidth=0.9, color="0.35", zorder=0)
            name = DATASET_DISPLAY.get(k, k)
            if len(vals) == 1:
                text = f"{name} majority baseline {vals[0]:.4f}"
            else:
                text = f"{name} majority baseline {vals[0]:.4f}\u2013{vals[-1]:.4f}"
            ax.annotate(text,
                        xy=(0.004, vals[-1]), xycoords=("axes fraction", "data"),
                        va="bottom", ha="left", fontsize=7.5, color="0.25")

    deg_idx = [i for i, g in enumerate(DEG) if g]
    if deg_idx:
        ax.plot(X[deg_idx], Y[deg_idx], linestyle="None", marker="x",
                markersize=MARKER_SIZE + 4, markeredgecolor="crimson",
                markeredgewidth=1.4, zorder=5)

    ax.set_xscale("log")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        lo, hi = ylim
        if len(Y) and float(Y.min()) >= 0.0 and lo < -0.1:
            lo = -0.05
        ax.set_ylim(lo, hi)
    ax.set_xlabel("Training Time (s)")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.margins(AX_MARGIN_XY, AX_MARGIN_XY)

    plt.subplots_adjust(right=right)

    model_handles = [Line2D([0],[0], marker="o", linestyle="",
                            markerfacecolor=color_map.get(m, (0.2,0.2,0.2)),
                            markeredgecolor="none", markersize=7)
                     for m in models_here]
    leg_models = ax.legend(model_handles, models_here, title="model",
                           loc="upper left", bbox_to_anchor=(1.02,1.00),
                           frameon=True, ncol=model_ncol, columnspacing=0.6, handletextpad=0.4)
    ax.add_artist(leg_models)

    dataset_handles = [Line2D([0],[0], marker=marker_map.get(d,"o"), linestyle="",
                              color="black", markersize=6)
                       for d in datasets_here]
    dataset_labels = list(datasets_here)
    if deg_idx:
        dataset_handles.append(Line2D([0],[0], marker="x", linestyle="", color="crimson",
                                      markeredgewidth=1.4, markersize=7))
        dataset_labels.append("degenerate (not every class predicted)")
    ax.legend(dataset_handles, dataset_labels, title="dataset",
              loc="lower left", bbox_to_anchor=(1.02,0.00),
              frameon=True, ncol=ds_ncol, columnspacing=0.8, handletextpad=0.4)

    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".pdf"), dpi=PNG_DPI, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path.with_suffix(".pdf").name, "and .png")


def best_by_dataset_chart(rows, out_path, use_overview_labels=False,
                          metric="accuracy", metric_label="Accuracy"):
    grouped = {}
    for r in rows:
        acc = _as_float(r.get(metric, ""))
        if acc is None:
            continue
        ds = dataset_short(r.get("dataset",""), r.get("_dataset_key"))
        model_short = normalize_label(r.get("model",""))
        if use_overview_labels:
            model_short = collapse_variants(model_short)
        model_short += protocol_tag(r)
        base = _as_float(r.get(BASELINE_COL, ""))
        if ds not in grouped or acc > grouped[ds]["acc"]:
            grouped[ds] = {"acc": acc, "model": model_short, "base": base}

    if not grouped:
        print(f"[WARN] No best-by-dataset data for {out_path.name}")
        return

    items = sorted(grouped.items(), key=lambda kv: kv[0])
    labels = [k for k, _ in items]
    accs = [v["acc"] for _, v in items]
    models = [v["model"] for _, v in items]
    bases = [v["base"] for _, v in items]

    fig_h = max(3.2, 0.42*len(labels))
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], fig_h))
    y = np.arange(len(labels))
    bars = ax.barh(y, accs)
    ax.set_yticks(y, labels)
    ax.set_xlabel(metric_label)
    ax.set_title(f"Best Model Per Dataset (by {metric_label})")

    if metric == "accuracy":
        for yi, b in zip(y, bases):
            if b is None:
                continue
            ax.plot([b, b], [yi - 0.4, yi + 0.4], color="crimson", linewidth=1.6, zorder=4)
        ax.plot([], [], color="crimson", linewidth=1.6, label="majority baseline")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                  fontsize=8, frameon=False)
        ax.set_title(f"Best Model Per Dataset (by {metric_label})", pad=22)

    longest = max((len(f"{m} ({v:.3f})") for m, v in zip(models, accs)), default=10)
    span = max(1.0, max(accs)) - min(0.0, min(accs))
    pad = span * 0.012 * longest + span * 0.03
    for rect, mdl, val in zip(bars, models, accs):
        x = rect.get_width()
        ax.text(x + span * 0.008, rect.get_y() + rect.get_height()/2,
                f"{mdl} ({val:.3f})", va="center", ha="left", fontsize=9)

    ax.set_xlim(min(0, min(accs) - 0.05), max(1.0, max(accs)) + pad)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".pdf"), dpi=PNG_DPI, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path.with_suffix(".pdf").name, "and .png")


# ---- MAIN ----
FULL_FAMILY_METRICS = {"mcc", "accuracy"}


def main():
    files = _resolve_inputs()
    if not files:
        print("No results CSV found. Run the model notebooks first to produce")
        print(f"  {REPO_ROOT/'2D_Results.csv'}")
        print(f"  {REPO_ROOT/'UAVATTACK_Results.csv'}")
        return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Reading:")
    for f in files:
        print(f"  {f}")
    rows_all = aggregate_seeds(load_rows(files))

    # Build consistent colors/markers for both label modes:
    # - overview (collapsed labels)
    # - variants (full labels)
    color_overview, color_variants, marker_map = build_style_maps(rows_all)

    def rows_for_family(rows, fam_key):
        kept = []
        for r in rows:
            short = normalize_label(r.get("model",""))
            fam = family_for(short)
            if fam_key == "MLP+XGB":
                if fam in ("MLP","XGB"): kept.append(r)
            elif fam == fam_key:
                kept.append(r)
        return kept

    zoom_specs = [
        ("LR",        "fig_zoom_LR_features"),
        ("RF",        "fig_zoom_RF_features"),
        ("MLP+XGB",   "fig_zoom_MLP_XGB_features"),
        ("CNN",       "fig_zoom_CNN_family"),
        ("CNN/ResNet-LSTM", "fig_zoom_CLSTM_family"),
        ("Ensemble",  "fig_zoom_Ensemble_RF"),
    ]

    rows_uav = [r for r in rows_all if r.get("_dataset_key") == "uav"]
    rows_2d = [r for r in rows_all if r.get("_dataset_key") == "2d"]

    for metric, metric_label, ylim in METRICS:
        outdir = OUTDIR / metric
        outdir.mkdir(parents=True, exist_ok=True)
        kw = dict(metric=metric, metric_label=metric_label, ylim=ylim)
        print(f"\n[{metric}]")

        scatter_figure(rows_all, outdir/"fig_all",
            title=f"{metric_label} vs Training Time (all models, all datasets)",
            color_map=color_overview, marker_map=marker_map,
            use_overview_labels=True, trim_percentiles=None, **kw)

        best_by_dataset_chart(rows_all, outdir/"fig_best_by_dataset",
            use_overview_labels=False, metric=metric, metric_label=metric_label)

        if metric not in FULL_FAMILY_METRICS:
            continue

        scatter_figure(rows_all, outdir/"fig_trimmed_p1p99",
            title=f"{metric_label} vs Training Time (all models, x trimmed to 1-99th pct)",
            color_map=color_overview, marker_map=marker_map,
            use_overview_labels=True, trim_percentiles=(1,99), **kw)

        if rows_uav:
            scatter_figure(rows_uav, outdir/"fig_uavattack_only",
                title=f"{metric_label} vs Training Time (UAVATTACK only)",
                color_map=color_overview, marker_map=marker_map,
                use_overview_labels=True, trim_percentiles=None, **kw)

        if rows_2d:
            scatter_figure(rows_2d, outdir/"fig_2d_only",
                title=f"{metric_label} vs Training Time (2D only)",
                color_map=color_overview, marker_map=marker_map,
                use_overview_labels=True, trim_percentiles=None, **kw)

        for fam_key, fname in zoom_specs:
            rsub = rows_for_family(rows_all, fam_key)
            if not rsub:
                continue
            scatter_figure(rsub, outdir/fname,
                title=f"{metric_label} vs Training Time ({fam_key} family)",
                color_map=color_variants, marker_map=marker_map,
                use_overview_labels=False, trim_percentiles=None, **kw)

    print(f"\nAll figures written to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
