#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_all_plots import (
    FIGSIZE,
    PNG_DPI,
    REPO_ROOT,
    _as_float,
)
from gps_ids.results import (
    CLASS_LABELS,
    SYNTH_FAMILIES,
    collapse_variants,
    dataset_short,
    load_rows,
    normalize_label,
    protocol_tag,
    results_files as _resolve_inputs,
)

OUTDIR = Path(__file__).resolve().parent / "plots_out" / "metrics"


def save(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path.with_suffix(".pdf").name, "and .png")


def label_of(row, overview=True):
    s = normalize_label(row.get("model", "Unknown"))
    s = collapse_variants(s) if overview else s
    return s + protocol_tag(row)


def is_degenerate(row):
    return str(row.get("is_degenerate", "")).strip().lower() in ("true", "1")


MARKERS = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H"]


def marker_map_for(rows):
    ds = sorted({dataset_short(r.get("dataset", ""), r.get("_dataset_key")) for r in rows})
    return {d: MARKERS[i % len(MARKERS)] for i, d in enumerate(ds)}


def baselines_by_dataset(rows):
    acc = {}
    for r in rows:
        b = _as_float(r.get("majority_baseline_accuracy"))
        if b is None:
            continue
        acc.setdefault(dataset_short(r.get("dataset", ""), r.get("_dataset_key")), []).append(b)
    return {d: float(np.median(v)) for d, v in acc.items()}


def fig_accuracy_vs_mcc(rows):
    fig, ax = plt.subplots(figsize=(FIGSIZE[0] * 0.78, FIGSIZE[1] * 0.78))
    markers = marker_map_for(rows)
    seen_ds = {}
    for r in rows:
        acc = _as_float(r.get("accuracy"))
        mcc = _as_float(r.get("mcc"))
        if acc is None or mcc is None:
            continue
        ds = dataset_short(r.get("dataset", ""), r.get("_dataset_key"))
        deg = is_degenerate(r)
        ax.plot(acc, mcc, linestyle="None", marker=markers.get(ds, "^"),
                markersize=6.5, markerfacecolor="crimson" if deg else "#2a6fb0",
                markeredgecolor="black", markeredgewidth=0.35, alpha=0.9)
        seen_ds[ds] = markers.get(ds, "^")

    bases = {d: b for d, b in baselines_by_dataset(rows).items() if d in seen_ds}
    groups = {}
    for d, b in bases.items():
        groups.setdefault(round(b, 3), []).append(d)
    annotate = len(groups) <= 3
    for i, (b, ds_list) in enumerate(sorted(groups.items())):
        ax.axvline(b, linestyle="--", linewidth=0.9, color="0.4")
        if annotate:
            tag = ds_list[0] if len(ds_list) == 1 else f"{len(ds_list)} datasets"
            ax.annotate(f"{tag} majority baseline {b:.3f}",
                        xy=(b, 0.97 - 0.07 * i), xycoords=("data", "axes fraction"),
                        rotation=90, va="top", ha="right", fontsize=7.5, color="0.25")

    ax.axhline(0.0, linestyle=":", linewidth=0.9, color="0.5")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Matthews Correlation Coefficient")
    ax.set_title("Accuracy vs MCC, against the majority-class baseline")
    ax.grid(True, linestyle=":", alpha=0.4)

    handles = [Line2D([0], [0], marker=m, linestyle="", color="black", markersize=6)
               for m in seen_ds.values()]
    labels = list(seen_ds.keys())
    handles.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor="crimson",
                          markeredgecolor="black", markersize=6))
    labels.append("degenerate (not every class predicted)")
    if not annotate:
        handles.append(Line2D([0], [0], linestyle="--", color="0.4", linewidth=0.9))
        labels.append("majority baseline")
    ax.legend(handles, labels, fontsize=8, loc="lower right", frameon=True)
    save(fig, OUTDIR / "fig_accuracy_vs_mcc")


def fig_metric_disagreement(rows, dataset_key, top_n=18):
    cols = [("accuracy", "Accuracy"), ("f1_weighted", "Weighted F1"),
            ("f1_macro", "Macro F1"), ("mcc", "MCC"), ("g_mean", "G-Mean")]
    sel = []
    for r in rows:
        if dataset_short(r.get("dataset", ""), r.get("_dataset_key")) != dataset_key:
            continue
        vals = [_as_float(r.get(c)) for c, _ in cols]
        if any(v is None for v in vals):
            continue
        sel.append((label_of(r, overview=False), vals, _as_float(r.get("mcc"))))
    if not sel:
        print(f"[WARN] no rows for metric-disagreement ({dataset_key})")
        return
    best = {}
    for name, vals, mcc in sel:
        if name not in best or mcc > best[name][1]:
            best[name] = (vals, mcc)
    items = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)[:top_n]

    names = [k for k, _ in items]
    data = np.array([v[0] for _, v in items])
    y = np.arange(len(names))
    h = 0.78 / len(cols)

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], max(3.6, 0.52 * len(names))))
    cmap = plt.get_cmap("viridis")
    for j, (_, lbl) in enumerate(cols):
        ax.barh(y + j * h - 0.39 + h / 2, data[:, j], height=h,
                color=cmap(j / max(1, len(cols) - 1)), label=lbl,
                edgecolor="black", linewidth=0.25)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_xlim(min(0.0, float(data.min()) - 0.03), 1.02)
    ax.set_title(f"The same model, scored five ways ({dataset_key})", pad=30)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, ncol=len(cols), loc="lower center",
              bbox_to_anchor=(0.5, 1.02), frameon=False)
    save(fig, OUTDIR / f"fig_metric_disagreement_{dataset_key}")


def fig_per_class_recall(rows, dataset_key, top_n=20):
    names = CLASS_LABELS.get(dataset_key)
    if not names:
        return
    k = len(names)
    sel = []
    for r in rows:
        if dataset_short(r.get("dataset", ""), r.get("_dataset_key")) != dataset_key:
            continue
        rec = [_as_float(r.get("recall_c%d" % i)) for i in range(k)]
        if any(v is None for v in rec):
            continue
        mcc = _as_float(r.get("mcc"))
        sel.append((label_of(r, overview=False), rec, mcc if mcc is not None else float("-inf")))
    if not sel:
        print(f"[WARN] no rows for per-class recall ({dataset_key})")
        return
    best = {}
    for nm, rec, mcc in sel:
        if nm not in best or mcc > best[nm][1]:
            best[nm] = (rec, mcc)
    items = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)[:top_n]
    labels = [kv[0] for kv in items]
    mat = np.array([kv[1][0] for kv in items])

    fig, ax = plt.subplots(figsize=(max(5.0, 1.5 * k + 3.0), max(3.4, 0.42 * len(labels))))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(k), names, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(k):
            v = mat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                    color="black" if 0.25 < v < 0.85 else "white")
    ax.set_title(f"Per-class recall ({dataset_key})")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Recall")
    save(fig, OUTDIR / f"fig_per_class_recall_{dataset_key}")


def fig_lift_over_baseline(rows, top_n=26):
    sel = []
    for r in rows:
        lift = _as_float(r.get("accuracy_lift_over_majority"))
        if lift is None:
            continue
        ds = dataset_short(r.get("dataset", ""), r.get("_dataset_key"))
        sel.append(("%s [%s]" % (label_of(r, overview=False), ds), lift, is_degenerate(r)))
    if not sel:
        print("[WARN] no rows for lift chart")
        return
    best = {}
    for nm, lift, deg in sel:
        if nm not in best or lift > best[nm][0]:
            best[nm] = (lift, deg)
    items = sorted(best.items(), key=lambda kv: kv[1][0])[:top_n]
    truncated = len(best) > top_n
    labels = [k for k, _ in items]
    vals = [v[0] for _, v in items]
    degs = [v[1] for _, v in items]

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], max(3.4, 0.42 * len(labels))))
    colors = ["crimson" if d or v <= 0 else "#2a6fb0" for v, d in zip(vals, degs)]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.3)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()

    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    span = (hi - lo) or 1.0
    for yi, v, d in zip(y, vals, degs):
        off = span * 0.012
        ha = "left" if v >= 0 else "right"
        ax.text(v + (off if v >= 0 else -off), yi,
                ("%+.4f" % v) + ("  degenerate" if d else ""),
                va="center", ha=ha, fontsize=8,
                color="crimson" if (d or v <= 0) else "0.25")
    ax.set_xlim(lo - span * 0.22, hi + span * 0.22)

    ax.set_xlabel("Accuracy minus majority-class baseline")
    ax.set_title("Accuracy lift over predicting the majority class"
                 + (" (worst %d shown)" % len(labels) if truncated else ""))
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    save(fig, OUTDIR / "fig_lift_over_baseline")


def fig_prior_shift(rows):
    pts = []
    for r in rows:
        tv = _as_float(r.get("prior_shift_tv"))
        if tv is None:
            continue
        fam = (r.get("family") or "").strip()
        if fam not in SYNTH_FAMILIES:
            continue
        ba = _as_float(r.get("balanced_accuracy"))
        gm = _as_float(r.get("g_mean"))
        acc = _as_float(r.get("accuracy"))
        if ba is None or gm is None or acc is None:
            continue
        pts.append((tv, ba, gm, acc, dataset_short(r.get("dataset", ""), r.get("_dataset_key")), fam))
    if not pts:
        print("[WARN] no synthetic/SMOTE rows for prior-shift chart")
        return

    fams = sorted({p[5] for p in pts})
    palette = ["#2a6fb0", "#e08a1e", "#3a9e5c", "#9467bd", "#8c564b", "#d62728"]
    fam_color = {f: palette[i % len(palette)] for i, f in enumerate(fams)}
    fam_marker = {f: MARKERS[i % len(MARKERS)] for i, f in enumerate(fams)}

    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE[0], FIGSIZE[1] * 0.62), sharex=True)
    for ax, (idx, name) in zip(axes, [(1, "Balanced Accuracy"), (3, "Accuracy")]):
        for f in fams:
            sub = [p for p in pts if p[5] == f]
            ax.scatter([p[0] for p in sub], [p[idx] for p in sub], s=38,
                       c=fam_color[f], marker=fam_marker[f],
                       edgecolors="black", linewidths=0.3, alpha=0.9)
        ax.set_xlabel("Train/test prior shift (total variation)")
        ax.set_ylabel(name)
        ax.grid(True, linestyle=":", alpha=0.4)
    axes[0].set_title("Prior shift vs imbalance-aware score", pad=10)
    axes[1].set_title("Prior shift vs accuracy", pad=10)
    handles = [Line2D([0], [0], marker=fam_marker[f], linestyle="",
                      markerfacecolor=fam_color[f], markeredgecolor="black", markersize=6)
               for f in fams]
    fig.legend(handles, fams, fontsize=8, frameon=False, ncol=min(len(fams), 4),
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, OUTDIR / "fig_prior_shift")


def fig_inference_latency(rows, top_n=20):
    sel = []
    for r in rows:
        b = _as_float(r.get("infer_batch_us_per_row"))
        p50 = _as_float(r.get("infer_online_p50_us"))
        p95 = _as_float(r.get("infer_online_p95_us"))
        if b is None or p50 is None:
            continue
        sel.append(("%s [%s]" % (label_of(r, overview=False),
                                 dataset_short(r.get("dataset", ""), r.get("_dataset_key"))), b, p50, p95))
    if not sel:
        print("[WARN] no inference-timing rows (needs a fresh results CSV)")
        return
    best = {}
    for nm, b, p50, p95 in sel:
        if nm not in best or p50 < best[nm][1]:
            best[nm] = (b, p50, p95)
    items = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)[:top_n]
    labels = [k for k, _ in items]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], max(3.4, 0.44 * len(labels))))
    ax.barh(y - 0.19, [v[0] for _, v in items], height=0.36,
            color="#9ecae1", edgecolor="black", linewidth=0.3, label="batch us/row")
    ax.barh(y + 0.19, [v[1] for _, v in items], height=0.36,
            color="#2a6fb0", edgecolor="black", linewidth=0.3, label="online p50 us")
    for yi, (_, v) in zip(y, items):
        if v[2] is not None:
            ax.plot([v[2]], [yi + 0.19], marker="|", markersize=10,
                    color="crimson", markeredgewidth=1.6)
    ax.plot([], [], marker="|", linestyle="", color="crimson",
            markeredgewidth=1.6, markersize=10, label="online p95 us")
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Microseconds per sample (log scale)")
    ax.set_title("Batch throughput flatters single-sample latency", pad=26)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=3, frameon=False)
    save(fig, OUTDIR / "fig_inference_latency")


def main():
    files = _resolve_inputs()
    if not files:
        print("No results CSV found. Run the model notebooks first to produce")
        print(f"  {REPO_ROOT/'2D_Results.csv'}")
        print(f"  {REPO_ROOT/'UAVATTACK_Results.csv'}")
        return
    print("Reading:")
    for f in files:
        print(f"  {f}")
    rows = load_rows(files)
    if "mcc" not in (rows[0].keys() if rows else {}):
        print("\n[ERROR] These results predate eval_metrics.py (no 'mcc' column).")
        print("Re-run the notebooks to regenerate them with the current schema.")
        return
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig_accuracy_vs_mcc(rows)
    fig_lift_over_baseline(rows)
    fig_prior_shift(rows)
    fig_inference_latency(rows)
    for ds in sorted({dataset_short(r.get("dataset", ""), r.get("_dataset_key")) for r in rows}):
        if ds in CLASS_LABELS:
            fig_metric_disagreement(rows, ds)
            fig_per_class_recall(rows, ds)

    print(f"\nAll figures written to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
