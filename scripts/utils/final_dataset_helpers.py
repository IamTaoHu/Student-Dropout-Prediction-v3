from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from scripts.utils.final_result_helpers import (
    load_json,
    load_table,
    normalize_label,
    save_figure,
    write_markdown,
)


@dataclass(frozen=True)
class DatasetPaths:
    dataset: str
    baseline_exp: Path | None
    threshold_exp: Path | None
    error_audit_exp: Path | None
    boosting_exp: Path | None


def canonical_dataset_id(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if "uct" in text:
        return "uct"
    if "oulad" in text:
        return "oulad"
    return None


def detect_dataset_from_experiment(exp_path: Path | None) -> str | None:
    if exp_path is None:
        return None
    summary = load_json(exp_path / "benchmark_summary.json", default={}) or {}
    by_name = canonical_dataset_id(summary.get("dataset_name"))
    if by_name:
        return by_name
    return canonical_dataset_id(exp_path.name)


def discover_default_dataset_paths(root: Path, dataset: str) -> DatasetPaths:
    results = root / "results"
    if dataset == "uct":
        baseline = results / "exp_bm_uct_3class"
        threshold = results / "exp_uct_3class_threshold_tuning_v1"
        error_audit = results / "exp_uct_3class_error_audit"
        boosting = results / "exp_uct_3class_boosting_optuna_v1"
    else:
        baseline_primary = results / "exp_bm_oulad_binary"
        baseline_fallback = results / "exp_bm_oulad_binary_post_leakage_fix"
        baseline = baseline_primary if baseline_primary.exists() else baseline_fallback
        threshold = results / "exp_oulad_threshold_tuning_v1"
        error_audit = results / "exp_oulad_error_audit"
        boosting = results / "exp_oulad_boosting_optuna_v1"
    return DatasetPaths(
        dataset=dataset,
        baseline_exp=baseline if baseline.exists() else None,
        threshold_exp=threshold if threshold.exists() else None,
        error_audit_exp=error_audit if error_audit.exists() else None,
        boosting_exp=boosting if boosting.exists() else None,
    )


def dataset_label_info(dataset: str) -> dict[str, Any]:
    if dataset == "uct":
        return {
            "fail_label": 1,
            "label_map": {0: "PASS", 1: "FAIL", 2: "DROPOUT"},
            "label_order": [0, 1, 2],
        }
    return {
        "fail_label": 1,
        "label_map": {0: "PASS", 1: "FAIL"},
        "label_order": [0, 1],
    }


def condition_from_role(role: str) -> str:
    mapping = {
        "baseline": "baseline",
        "threshold": "threshold_tuned",
        "boosting": "boosting_optuna",
        "error_audit": "error_audit",
    }
    return mapping.get(role, role)


def _best_report_path(exp_path: Path, model: str, condition: str) -> Path | None:
    candidates = []
    token = str(model)
    if condition == "threshold_tuned":
        candidates.append(exp_path / f"classification_report_after_{token}.json")
    candidates.extend(
        [
            exp_path / f"classification_report_{token}.json",
            exp_path / f"classification_report_before_{token}.json",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _fail_metrics_from_predictions(
    y_true: list[Any] | np.ndarray,
    y_pred: list[Any] | np.ndarray,
    fail_label: Any,
) -> tuple[float, float, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=[fail_label],
        average=None,
        zero_division=0,
    )
    return float(precision[0]), float(recall[0]), float(f1[0])


def _prediction_files(exp_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in exp_path.rglob("*.csv"):
        name = path.name.lower()
        if ("pred" in name) or ("prediction" in name):
            candidates.append(path)
    return sorted(set(candidates))


def _report_files(exp_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in exp_path.rglob("*.json"):
        name = path.name.lower()
        if ("classification_report" in name) or ("report" in name):
            candidates.append(path)
    return sorted(set(candidates))


def _infer_model_from_filename(path: Path) -> str | None:
    token = path.stem.lower()
    known = [
        "catboost",
        "lightgbm",
        "xgboost",
        "gradient_boosting",
        "random_forest",
        "decision_tree",
        "svm",
    ]
    for model in known:
        if model in token:
            return model
    return None


def infer_true_pred_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    true_candidates = ["y_true", "true_label", "label", "target", "actual", "ground_truth", "truth"]
    pred_candidates = ["y_pred", "predicted_label", "prediction", "pred", "yhat", "y_hat"]
    true_col = next((c for c in true_candidates if c in df.columns), None)
    pred_col = next((c for c in pred_candidates if c in df.columns), None)
    return true_col, pred_col


def _infer_model_column(df: pd.DataFrame) -> str | None:
    for col in ["model", "model_name", "estimator", "algorithm"]:
        if col in df.columns:
            return col
    return None


def _filter_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    split_col = next((c for c in ["split", "subset", "fold", "set"] if c in df.columns), None)
    if split_col is None:
        return df, False
    split_values = df[split_col].astype(str).str.lower()
    mask = split_values.str.contains("test")
    if mask.any():
        return df.loc[mask].copy(), True
    return df, True


def infer_fail_label(df: pd.DataFrame, dataset: str, default_fail_label: Any) -> Any:
    values = pd.unique(pd.concat([df.iloc[:, 0], df.iloc[:, 1]], axis=0)) if not df.empty else []
    for value in values:
        if normalize_label(value) == "FAIL":
            return value
    if dataset == "oulad":
        return 1 if any(str(v).strip() in {"1", "1.0"} for v in values) else default_fail_label
    return default_fail_label


def _as_comparable_label(series: pd.Series) -> pd.Series:
    mapped = series.map(lambda v: normalize_label(v))
    return mapped


def _compute_fail_from_dataframe(df: pd.DataFrame, dataset: str, fail_label_default: Any) -> tuple[float, float, float] | None:
    true_col, pred_col = infer_true_pred_columns(df)
    if true_col is None or pred_col is None:
        return None
    pair_df = df[[true_col, pred_col]].dropna().copy()
    if pair_df.empty:
        return None
    fail_label = infer_fail_label(pair_df[[true_col, pred_col]], dataset=dataset, default_fail_label=fail_label_default)
    y_true = _as_comparable_label(pair_df[true_col])
    y_pred = _as_comparable_label(pair_df[pred_col])
    fail_norm = normalize_label(fail_label)
    return _fail_metrics_from_predictions(y_true=y_true.to_numpy(), y_pred=y_pred.to_numpy(), fail_label=fail_norm)


def _backfill_fail_metrics(
    exp_path: Path,
    dataset: str,
    model: str,
    fail_label_default: Any,
    summary: dict[str, Any],
) -> tuple[float | None, float | None, float | None, str, str, list[str]]:
    warnings: list[str] = []

    # Attempt from summary arrays + runtime y_test.
    model_payload = (summary.get("model_results", {}) or {}).get(model, {})
    artifacts = model_payload.get("artifacts", {}) if isinstance(model_payload, dict) else {}
    y_pred = artifacts.get("y_pred_test")
    y_true = artifacts.get("y_true_test")
    if y_true is None:
        y_test_df = load_table(exp_path / "runtime_artifacts" / "y_test.csv")
        if y_test_df is None:
            y_test_df = load_table(exp_path / "runtime_artifacts" / "y_test.parquet")
        if y_test_df is not None and not y_test_df.empty:
            y_true = y_test_df.iloc[:, 0].tolist()
    if y_true is not None and y_pred is not None and len(y_true) == len(y_pred):
        y_true_norm = pd.Series(y_true).map(normalize_label).to_numpy()
        y_pred_norm = pd.Series(y_pred).map(normalize_label).to_numpy()
        fp, fr, ff = _fail_metrics_from_predictions(y_true_norm, y_pred_norm, normalize_label(fail_label_default))
        return fp, fr, ff, "predictions_backfill", "summary:model_results+runtime_artifacts/y_test", warnings

    # Attempt from prediction-like CSVs recursively.
    pred_files = _prediction_files(exp_path)
    for pred_file in pred_files:
        df = load_table(pred_file)
        if df is None or df.empty:
            continue
        df_use, split_detected = _filter_test_split(df)
        if not split_detected:
            warnings.append(f"{model}: split column missing in {pred_file.name}; using all rows.")
        model_col = _infer_model_column(df_use)
        if model_col is not None:
            sub = df_use[df_use[model_col].astype(str).str.lower() == model.lower()].copy()
            if sub.empty:
                continue
            metrics = _compute_fail_from_dataframe(sub, dataset=dataset, fail_label_default=fail_label_default)
            if metrics is not None:
                return metrics[0], metrics[1], metrics[2], "predictions_backfill", str(pred_file), warnings
        else:
            guessed = _infer_model_from_filename(pred_file)
            if guessed and guessed != model:
                continue
            if (summary.get("best_model") == model) or (guessed == model):
                metrics = _compute_fail_from_dataframe(df_use, dataset=dataset, fail_label_default=fail_label_default)
                if metrics is not None:
                    return metrics[0], metrics[1], metrics[2], "predictions_backfill", str(pred_file), warnings

    # Attempt from classification report files recursively.
    for rep_file in _report_files(exp_path):
        guessed = _infer_model_from_filename(rep_file)
        if guessed and guessed != model:
            continue
        payload = load_json(rep_file, default={}) or {}
        fail_key = None
        for key in payload.keys():
            if normalize_label(key) == "FAIL":
                fail_key = key
                break
        if fail_key is None:
            continue
        fp = payload.get(fail_key, {}).get("precision")
        fr = payload.get(fail_key, {}).get("recall")
        ff = payload.get(fail_key, {}).get("f1-score")
        if fp is not None and fr is not None and ff is not None:
            return float(fp), float(fr), float(ff), "classification_report", str(rep_file), warnings

    return None, None, None, "unavailable", "", warnings


def gather_experiment_model_rows(
    dataset: str,
    role: str,
    exp_path: Path | None,
    backfill_missing_fail_metrics: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    columns = [
        "dataset",
        "experiment_name",
        "model",
        "condition",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "fail_precision",
        "fail_recall",
        "fail_f1",
        "fail_metrics_source",
        "selected_as_best",
        "notes",
        "source_experiment_path",
    ]
    if exp_path is None or not exp_path.exists():
        return pd.DataFrame(columns=columns), [f"{dataset}:{role} missing experiment path"], {
            "backfilled_models": [],
            "unavailable_models": [],
            "source_files": [],
        }

    summary = load_json(exp_path / "benchmark_summary.json", default={}) or {}
    leaderboard = load_table(exp_path / "leaderboard.csv")
    if leaderboard is None:
        return pd.DataFrame(columns=columns), [f"{dataset}:{role} missing leaderboard.csv"], {
            "backfilled_models": [],
            "unavailable_models": [],
            "source_files": [],
        }

    label_info = dataset_label_info(dataset)
    fail_label = int(label_info["fail_label"])
    condition = condition_from_role(role)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    backfilled_models: list[str] = []
    unavailable_models: list[str] = []
    source_files: list[str] = []
    for _, lb in leaderboard.iterrows():
        model = str(lb.get("model"))
        accuracy = lb.get("test_accuracy", np.nan)
        macro_f1 = lb.get("test_macro_f1", np.nan)
        weighted_f1 = lb.get("test_weighted_f1", np.nan)
        balanced_accuracy = lb.get("test_balanced_accuracy", np.nan)
        fail_precision = np.nan
        fail_recall = np.nan
        fail_f1 = np.nan
        fail_metrics_source = "unavailable"
        notes = ""

        report_path = _best_report_path(exp_path, model, condition)
        if report_path is not None:
            report = load_json(report_path, default={}) or {}
            fail_key = None
            for key in report.keys():
                if normalize_label(key) == "FAIL":
                    fail_key = key
                    break
            if fail_key is not None:
                fail_precision = float(report.get(fail_key, {}).get("precision", np.nan))
                fail_recall = float(report.get(fail_key, {}).get("recall", np.nan))
                fail_f1 = float(report.get(fail_key, {}).get("f1-score", np.nan))
                fail_metrics_source = "classification_report"

        if np.isnan(fail_f1) and backfill_missing_fail_metrics:
            fp, fr, ff, source, source_file, backfill_warnings = _backfill_fail_metrics(
                exp_path=exp_path,
                dataset=dataset,
                model=model,
                fail_label_default=fail_label,
                summary=summary,
            )
            warnings.extend([f"{dataset}:{role}:{w}" for w in backfill_warnings])
            if source_file:
                source_files.append(source_file)
            if ff is not None:
                fail_precision, fail_recall, fail_f1 = fp, fr, ff
                fail_metrics_source = source
                if source == "predictions_backfill":
                    backfilled_models.append(model)
            else:
                fail_metrics_source = "unavailable"
                unavailable_models.append(model)
                notes = "fail_metrics_unavailable"

        rows.append(
            {
                "dataset": dataset,
                "experiment_name": exp_path.name,
                "model": model,
                "condition": condition,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "balanced_accuracy": balanced_accuracy,
                "fail_precision": fail_precision,
                "fail_recall": fail_recall,
                "fail_f1": fail_f1,
                "fail_metrics_source": fail_metrics_source,
                "selected_as_best": False,
                "notes": notes,
                "source_experiment_path": str(exp_path),
            }
        )
    return pd.DataFrame(rows, columns=columns), warnings, {
        "backfilled_models": sorted(set(backfilled_models)),
        "unavailable_models": sorted(set(unavailable_models)),
        "source_files": sorted(set(source_files)),
    }


def select_best_model_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    sort_df = df.copy()
    for metric in ["macro_f1", "balanced_accuracy", "fail_f1", "accuracy"]:
        sort_df[metric] = pd.to_numeric(sort_df[metric], errors="coerce")
    sort_df = sort_df.sort_values(
        by=["macro_f1", "balanced_accuracy", "fail_f1", "accuracy", "model"],
        ascending=[False, False, False, False, True],
        na_position="last",
    )
    return sort_df.iloc[0]


def get_model_artifacts(exp_path: Path, model: str) -> dict[str, Any]:
    summary = load_json(exp_path / "benchmark_summary.json", default={}) or {}
    payload = (summary.get("model_results", {}) or {}).get(model, {})
    artifacts = payload.get("artifacts", {}) if isinstance(payload, dict) else {}
    y_true = artifacts.get("y_true_test")
    y_pred = artifacts.get("y_pred_test")
    y_proba = artifacts.get("y_proba_test")
    labels = artifacts.get("labels")
    params = artifacts.get("params") or {}
    if y_true is None or y_pred is None:
        pred = load_table(exp_path / "predictions.csv")
        if pred is not None and {"y_true", "y_pred"}.issubset(pred.columns):
            y_true = pred["y_true"].tolist()
            y_pred = pred["y_pred"].tolist()
            proba_cols = [c for c in pred.columns if c.startswith("proba_class_")]
            if proba_cols:
                y_proba = pred[proba_cols].to_numpy(dtype=float).tolist()
                labels = [int(c.replace("proba_class_", "")) for c in proba_cols]
    return {
        "summary": summary,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "labels": labels,
        "params": params,
    }


def find_existing_figure(exp_path: Path, kind: str) -> Path | None:
    candidate = exp_path / "figures" / f"{kind}.png"
    if candidate.exists():
        return candidate
    return None


def copy_or_none(source: Path | None, target: Path) -> bool:
    if source is None or not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def render_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    label_map: dict[int, str],
    normalized: bool,
    title: str,
    output_path: Path,
) -> None:
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = float(((y_true == true_label) & (y_pred == pred_label)).sum())
    if normalized:
        denom = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            txt = f"{matrix[i, j]:.2f}" if normalized else str(int(matrix[i, j]))
            ax.text(j, i, txt, ha="center", va="center")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([label_map.get(v, str(v)) for v in labels], rotation=20, ha="right")
    ax.set_yticklabels([label_map.get(v, str(v)) for v in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    save_figure(output_path)


def save_markdown_table(path: Path, df: pd.DataFrame, title: str) -> None:
    lines = [f"# {title}", "", df.to_markdown(index=False)]
    write_markdown(path, "\n".join(lines))


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
