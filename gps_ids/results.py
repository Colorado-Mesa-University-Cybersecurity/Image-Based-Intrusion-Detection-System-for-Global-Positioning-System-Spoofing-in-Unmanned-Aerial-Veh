import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import schema as _schema
from .paths import repo_root

MODELS = {
    "Logistic Regression": "LR",
    "Logistic Regression (SMOTE)": "LR+SM",
    "Logistic Regression (Explainability Features)": "LR+EXPL",
    "Logistic Regression (Xgboost Features)": "LR+XGBF",
    "Logistic Regression (Quadratic Features)": "LR+QUAD",
    "Random Forest": "RF",
    "Random Forest (SMOTE)": "RF+SM",
    "Random Forest (Explainability Features)": "RF+EXPL",
    "Random Forest (Xgboost Features)": "RF+XGBF",
    "XGBoost": "XGB",
    "XGBoost (SMOTE)": "XGB+SM",
    "XGBoost (Explainability Features)": "XGB+EXPL",
    "XGBoost (Xgboost Features)": "XGB+XGBF",
    "MLP Classifier": "MLP",
    "MLP Classifier (SMOTE)": "MLP+SM",
    "MLP Classifier (Explainability Features)": "MLP+EXPL",
    "MLP Classifier (Xgboost Features)": "MLP+XGBF",
    "LSTM Classifier": "LSTM",
    "LSTM Classifier (SMOTE)": "LSTM+SM",
    "LSTM Classifier (Explainability Features)": "LSTM+EXPL",
    "LSTM Classifier (Xgboost Features)": "LSTM+XGBF",
    "CNN": "CNN",
    "CNN-LSTM": "C-LSTM",
    "ResNet-CNN": "ResNet",
    "ResNet-LSTM": "ResNet-LSTM",
    "OneClassSVM": "OCSVM",
    "LocalOutlierFactor": "LOF",
    "StackedEnsemble_RF": "SE-RF",
    "StackedEnsemble_RF (SMOTE)": "SE-RF+SM",
    "StackingClassifier(LogisticRegression RandomForest SVC)": "STACK",
}

FAMILIES = {
    "Regular",
    "SMOTE",
    "Stacking",
    "OneClass",
    "LSTM",
    "FeatureEngineering",
    "Image",
    "Synthetic",
    "Synthetic-LSTM",
    "Synthetic-Stacking",
}

SYNTH_FAMILIES = {"Synthetic", "Synthetic-Stacking", "Synthetic-LSTM", "SMOTE"}

TEST_SET_TAGS = {
    "real_heldout_20pct": "",
    "real_full": " [full-real]",
    "real_heldout_benign_plus_malicious": " [benign-ho]",
}

VARIANT_SUFFIXES = ("+SM", "+EXPL", "+XGBF", "+QUAD")


@dataclass(frozen=True)
class DatasetInfo:
    short: str
    kind: str
    datasets: tuple
    generator: Optional[str] = None
    size: Optional[int] = None


def _sdv_entries():
    entries = {}
    for gen, label in (("gc", "GC"), ("ctgan", "CTGAN"), ("tvae", "TVAE")):
        for size, tag in ((10_000, "10k"), (100_000, "100k"), (1_000_000, "1M")):
            entries[f"spoofing_{gen}_{size}.csv"] = DatasetInfo(
                short=f"{label}-{tag}",
                kind="synthetic",
                datasets=("uav", "2d"),
                generator=gen,
                size=size,
            )
    return entries


DATASETS = {
    "spoofing-merged-gps-only.csv": DatasetInfo(
        short="SPOOF-GPS", kind="real", datasets=("uav",)
    ),
    "GPS_Data_Simplified_2D_Feature_Map.xlsx": DatasetInfo(
        short="GPS-2D", kind="real", datasets=("2d",)
    ),
    "syntheticSpoofing_gan_data.csv": DatasetInfo(
        short="GAN", kind="synthetic", datasets=("uav", "2d"), generator="gan", size=10_000
    ),
    "syntheticSpoofing_cgan_data.csv": DatasetInfo(
        short="CGAN", kind="synthetic", datasets=("uav", "2d"), generator="cgan", size=10_000
    ),
    **_sdv_entries(),
}

DATASET_KEY_SUFFIX = {"uav": " (UAV)", "2d": " (2D)"}

CLASS_LABELS = {
    info.short: list(_schema.get(info.datasets[0]).class_names)
    for info in DATASETS.values()
    if info.kind == "real"
}


def normalize_label(name):
    n = (name or "").strip()
    return MODELS.get(n, n)


def collapse_variants(short):
    for suffix in VARIANT_SUFFIXES:
        if short.endswith(suffix):
            return short[: -len(suffix)]
    return short


def family_for(short):
    s = short
    if s.startswith("ResNet-LSTM") or s.startswith("C-LSTM"):
        return "CNN/ResNet-LSTM"
    if s.startswith("ResNet") or s == "CNN":
        return "CNN"
    if s.startswith("SE-") or s.startswith("STACK"):
        return "Ensemble"
    if s.startswith("RF"):
        return "RF"
    if s.startswith("LR"):
        return "LR"
    if s.startswith("XGB"):
        return "XGB"
    if s.startswith("MLP"):
        return "MLP"
    if s.startswith("OCSVM") or s.startswith("LOF"):
        return "OneClass"
    if s.startswith("LSTM"):
        return "LSTM"
    return s


def dataset_short(name, dataset_key=None):
    n = (name or "").strip()
    info = DATASETS.get(n)
    if info is None:
        return n
    if dataset_key and len(info.datasets) > 1 and dataset_key in DATASET_KEY_SUFFIX:
        return info.short + DATASET_KEY_SUFFIX[dataset_key]
    return info.short


def synthetic_info(name):
    info = DATASETS.get((name or "").strip())
    if info is None or info.kind != "synthetic":
        return None
    return info


def synthetic_files(dataset_key, generators=None):
    return [
        name
        for name, info in DATASETS.items()
        if info.kind == "synthetic"
        and dataset_key in info.datasets
        and (generators is None or info.generator in generators)
    ]


def protocol_tag(row):
    ts = (row.get("test_set", "") or "").strip()
    if ts in TEST_SET_TAGS:
        return TEST_SET_TAGS[ts]
    return " [%s]" % ts if ts and ts != "N/A" else ""


def results_files():
    found = []
    for schema in _schema.SCHEMAS.values():
        path = _schema.results_path(schema)
        if path.exists():
            found.append(path)
    return found


def _dataset_key_for_source(source_name):
    if "UAVATTACK" in source_name:
        return "uav"
    if source_name.startswith("2D"):
        return "2d"
    return None


def load_rows(files=None):
    files = results_files() if files is None else [Path(f) for f in files]
    rows = []
    for f in files:
        with open(f, "r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                row = {
                    (k or "").strip().lstrip("﻿"): (v or "").strip()
                    for k, v in raw.items()
                }
                row["_source"] = Path(f).name
                row["_dataset_key"] = _dataset_key_for_source(Path(f).name)
                rows.append(row)
    return rows


SEED_METRICS = (
    "mcc",
    "balanced_accuracy",
    "f1_macro",
    "accuracy",
    "min_class_recall",
    "train_time_sec",
)


def summarize_seeds(files=None, metrics=None, rows=None):
    import math

    metrics = SEED_METRICS if metrics is None else tuple(metrics)
    rows = load_rows(files) if rows is None else rows
    groups = {}
    for row in rows:
        key = (
            row.get("_dataset_key"),
            row.get("model"),
            row.get("dataset"),
            row.get("test_set"),
        )
        groups.setdefault(key, []).append(row)
    out = []
    for (_dskey, model, dataset, test_set), group in sorted(
        groups.items(), key=lambda item: tuple("" if p is None else str(p) for p in item[0])
    ):
        seeds = sorted({str(r.get("seed")) for r in group})
        entry = {
            "model": model,
            "dataset": dataset,
            "test_set": test_set,
            "n_runs": len(group),
            "n_seeds": len(seeds),
            "seeds": ";".join(seeds),
        }
        for metric in metrics:
            values = []
            for row in group:
                raw = row.get(metric)
                if raw in (None, "", "N/A"):
                    continue
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    continue
            if not values:
                entry[metric + "_mean"] = None
                entry[metric + "_std"] = None
                continue
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                std = math.sqrt(variance)
            else:
                std = 0.0
            entry[metric + "_mean"] = mean
            entry[metric + "_std"] = std
        out.append(entry)
    return out


def format_seed_summary(summary, metric="mcc"):
    lines = []
    for entry in summary:
        mean = entry.get(metric + "_mean")
        std = entry.get(metric + "_std")
        if mean is None:
            continue
        lines.append(
            "%-55s %-22s %s %.4f +/- %.4f  (n=%d)"
            % (
                entry["model"],
                dataset_short(entry["dataset"]) or entry["dataset"],
                metric,
                mean,
                std,
                entry["n_seeds"],
            )
        )
    return "\n".join(lines)


AGGREGATE_KEY = ("_dataset_key", "_source", "model", "dataset", "family", "test_set")

AGGREGATE_VERBATIM = frozenset(
    (
        "model",
        "dataset",
        "family",
        "test_set",
        "seed",
        "class_names",
        "worst_class",
        "parameters",
        "is_degenerate",
    )
)


def _is_true(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def aggregate_seeds(rows):
    import math

    groups = {}
    order = []
    for row in rows:
        key = tuple(row.get(k) for k in AGGREGATE_KEY)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    out = []
    for key in order:
        group = groups[key]
        entry = dict(group[0])
        values = {}
        for row in group:
            for name, raw in row.items():
                if name in AGGREGATE_VERBATIM or name.startswith("_"):
                    continue
                if raw in (None, "", "N/A", "nan"):
                    continue
                try:
                    number = float(str(raw).strip())
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(number):
                    continue
                values.setdefault(name, []).append(number)

        for name, found in values.items():
            mean = sum(found) / len(found)
            entry[name] = "%.17g" % mean
            if len(found) > 1:
                variance = sum((v - mean) ** 2 for v in found) / (len(found) - 1)
                entry[name + "_std"] = "%.17g" % math.sqrt(variance)
            else:
                entry[name + "_std"] = "0"

        seeds = sorted({str(r.get("seed", "")).strip() for r in group if str(r.get("seed", "")).strip()})
        entry["is_degenerate"] = "True" if any(_is_true(r.get("is_degenerate")) for r in group) else "False"
        entry["n_degenerate"] = str(sum(1 for r in group if _is_true(r.get("is_degenerate"))))
        entry["seed"] = "mean(%s)" % ";".join(seeds)
        entry["seeds"] = ";".join(seeds)
        entry["n_seeds"] = str(len(seeds))
        entry["n_runs"] = str(len(group))
        out.append(entry)

    names = []
    seen = set()
    for entry in out:
        for name in entry:
            if name not in seen:
                seen.add(name)
                names.append(name)
    for entry in out:
        for name in names:
            entry.setdefault(name, "")
    return out


def validate_row(row):
    problems = []
    model = row.get("model")
    if model not in MODELS:
        problems.append("model %r is not in gps_ids.results.MODELS" % model)
    dataset = row.get("dataset")
    if dataset not in DATASETS:
        problems.append("dataset %r is not in gps_ids.results.DATASETS" % dataset)
    family = row.get("family")
    if family not in FAMILIES:
        problems.append("family %r is not in gps_ids.results.FAMILIES" % family)
    test_set = row.get("test_set")
    if test_set not in TEST_SET_TAGS:
        problems.append(
            "test_set %r is not in gps_ids.results.TEST_SET_TAGS" % test_set
        )
    if problems:
        raise ValueError(
            "results row failed vocabulary validation:\n  " + "\n  ".join(problems)
        )


def append_validated(row_or_rows, path):
    import eval_metrics

    rows = row_or_rows if isinstance(row_or_rows, list) else [row_or_rows]
    for row in rows:
        validate_row(row)
    return eval_metrics.append_row(rows, path)
