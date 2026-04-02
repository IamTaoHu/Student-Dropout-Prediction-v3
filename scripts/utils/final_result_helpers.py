from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_NORMALIZATION_MAP = {
    "0": "PASS",
    "1": "FAIL",
    "2": "DROPOUT",
    "graduate": "PASS",
    "enrolled": "FAIL",
    "dropout": "DROPOUT",
    "pass": "PASS",
    "fail": "FAIL",
}


@dataclass(frozen=True)
class BundleContext:
    baseline_exp: Path | None
    threshold_exp: Path | None
    error_audit_exp: Path | None
    boosting_exp: Path | None
    output_dir: Path


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> Path:
    safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def load_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return None


def save_table(path: Path, df: pd.DataFrame) -> Path:
    safe_mkdir(path.parent)
    df.to_csv(path, index=False)
    return path


def write_markdown(path: Path, text: str) -> Path:
    safe_mkdir(path.parent)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def write_markdown_table(path: Path, df: pd.DataFrame, title: str | None = None) -> Path:
    lines: list[str] = []
    if title:
        lines.extend([f"# {title}", ""])
    lines.append(df.to_markdown(index=False))
    return write_markdown(path, "\n".join(lines))


def save_figure(path: Path, dpi: int = 220) -> Path:
    safe_mkdir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def normalize_label(value: Any) -> str:
    key = str(value).strip().lower()
    if key in LABEL_NORMALIZATION_MAP:
        return LABEL_NORMALIZATION_MAP[key]
    if key.endswith(".0") and key[:-2].isdigit():
        return LABEL_NORMALIZATION_MAP.get(key[:-2], str(value))
    return str(value).strip().upper()


def normalize_series_labels(series: pd.Series) -> pd.Series:
    return series.map(normalize_label)


def parse_literal_list(raw_value: Any) -> list[dict[str, Any]]:
    if isinstance(raw_value, list):
        return raw_value
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        return []
    return []


def find_report_class_key(report: dict[str, Any], canonical_name: str) -> str | None:
    canonical_name = canonical_name.upper().strip()
    for key in report.keys():
        if key in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if normalize_label(key) == canonical_name:
            return key
    return None


def metrics_from_classification_report(report: dict[str, Any], fail_key: str | None) -> dict[str, float]:
    macro_avg = report.get("macro avg", {}) if isinstance(report, dict) else {}
    weighted_avg = report.get("weighted avg", {}) if isinstance(report, dict) else {}
    fail_metrics = report.get(fail_key, {}) if fail_key and isinstance(report, dict) else {}
    balanced_accuracy = float(macro_avg.get("recall", np.nan))
    return {
        "accuracy": float(report.get("accuracy", np.nan)),
        "macro_f1": float(macro_avg.get("f1-score", np.nan)),
        "weighted_f1": float(weighted_avg.get("f1-score", np.nan)),
        "balanced_accuracy": balanced_accuracy,
        "fail_precision": float(fail_metrics.get("precision", np.nan)),
        "fail_recall": float(fail_metrics.get("recall", np.nan)),
        "fail_f1": float(fail_metrics.get("f1-score", np.nan)),
        "support_fail": float(fail_metrics.get("support", np.nan)),
    }


def select_best_model(
    leaderboard: pd.DataFrame,
    tie_break_fail_f1: dict[str, float] | None = None,
) -> str | None:
    if leaderboard.empty or "model" not in leaderboard.columns:
        return None
    df = leaderboard.copy()
    if "test_macro_f1" not in df.columns:
        return str(df.iloc[0]["model"])
    df["test_macro_f1"] = pd.to_numeric(df["test_macro_f1"], errors="coerce")
    if "test_balanced_accuracy" in df.columns:
        df["test_balanced_accuracy"] = pd.to_numeric(df["test_balanced_accuracy"], errors="coerce")
    else:
        df["test_balanced_accuracy"] = np.nan
    fail_map = tie_break_fail_f1 or {}
    df["fail_f1"] = df["model"].map(lambda m: float(fail_map.get(str(m), np.nan)))
    df = df.sort_values(
        by=["test_macro_f1", "test_balanced_accuracy", "fail_f1", "model"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    return str(df.iloc[0]["model"])


def flatten_shap_global(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["feature", "class_name", "importance"])
    rows: list[dict[str, Any]] = []
    if {"class_index", "top_features"}.issubset(df.columns):
        for _, row in df.iterrows():
            class_name = normalize_label(row["class_index"])
            top_features = parse_literal_list(row["top_features"])
            for item in top_features:
                rows.append(
                    {
                        "feature": item.get("feature"),
                        "class_name": class_name,
                        "importance": float(item.get("mean_abs_shap", np.nan)),
                    }
                )
    return pd.DataFrame(rows)


def flatten_shap_local(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["instance_index", "predicted_label", "class_name", "feature", "local_importance"])
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        top_features = parse_literal_list(row.get("top_features"))
        predicted = normalize_label(row.get("predicted_label"))
        explained = normalize_label(row.get("explained_class", row.get("predicted_label")))
        for feat in top_features:
            rows.append(
                {
                    "instance_index": int(row.get("instance_index")),
                    "predicted_label": predicted,
                    "class_name": explained,
                    "feature": feat.get("feature"),
                    "local_importance": float(feat.get("shap_value", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def flatten_lime_local(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["instance_index", "predicted_label", "class_name", "feature", "local_importance"])
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        explanation = parse_literal_list(row.get("explanation"))
        predicted = normalize_label(row.get("predicted_label"))
        explained = normalize_label(row.get("explained_label", row.get("predicted_label")))
        for item in explanation:
            rows.append(
                {
                    "instance_index": int(row.get("instance_index")),
                    "predicted_label": predicted,
                    "class_name": explained,
                    "feature": item.get("feature_term"),
                    "local_importance": float(item.get("weight", np.nan)),
                    "rank": int(item.get("rank", 0)),
                }
            )
    return pd.DataFrame(rows)


def cosine_similarity_to_vector(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    vector_norm = float(np.linalg.norm(vector))
    denom = (matrix_norm[:, 0] * vector_norm) + 1e-12
    return (matrix @ vector) / denom


def collect_relative_file_list(root: Path) -> list[str]:
    files = sorted([p for p in root.rglob("*") if p.is_file()])
    return [str(p.relative_to(root)).replace("\\", "/") for p in files]
