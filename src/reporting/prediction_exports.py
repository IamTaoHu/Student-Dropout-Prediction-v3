from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.experiment.eval_validation import _assert_same_length_arrays

def _safe_filename_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_").lower()


def _metric_label_token(raw_label: str) -> str:
    value = str(raw_label).strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _resolve_metric_column(metric_name: str, source: str) -> str:
    token = str(metric_name).strip()
    alias_map = {
        "enrolled_f1": "f1_enrolled",
        "enrolled_recall": "recall_enrolled",
        "enrolled_precision": "precision_enrolled",
    }
    token = alias_map.get(token.lower(), token)
    if token.startswith(("test_", "valid_", "cv_")):
        return token
    if source == "cv":
        return f"cv_{token}_mean"
    return f"test_{token}"


def _build_prediction_export_dataframe(
    y_true: pd.Series,
    y_pred: Any,
    y_proba: Any,
    labels: list[Any] | None,
    class_metadata: dict[str, Any] | None = None,
    extra_columns: dict[str, Any] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true.reset_index(drop=True)})
    if y_pred is not None:
        df["y_pred"] = np.asarray(y_pred)

    index_to_label = (
        (class_metadata or {}).get("class_index_to_label", {})
        if isinstance(class_metadata, dict)
        else {}
    )
    if index_to_label:
        df["true_label"] = df["y_true"].map(lambda value: index_to_label.get(str(int(value)), value))
        if "y_pred" in df.columns:
            df["pred_label"] = df["y_pred"].map(lambda value: index_to_label.get(str(int(value)), value))

    if y_proba is None:
        if extra_columns is not None:
            extra_df = extra_columns if isinstance(extra_columns, pd.DataFrame) else pd.DataFrame(extra_columns)
            _assert_same_length_arrays(
                context="_build_prediction_export_dataframe:no_proba_extra_columns",
                export_df=df,
                extra_columns=extra_df,
            )
            df = pd.concat([df, extra_df.reset_index(drop=True)], axis=1)
        return df
    probs = np.asarray(y_proba, dtype=float)
    if probs.ndim != 2:
        return df

    ordered_labels = list(labels or [])
    if not ordered_labels:
        ordered_labels = list(range(probs.shape[1]))
    if probs.shape[1] != len(ordered_labels):
        raise ValueError(
            "Probability export failed because label/probability dimensions differ: "
            f"n_labels={len(ordered_labels)}, proba_shape={probs.shape}."
        )

    for idx in range(probs.shape[1]):
        class_idx = ordered_labels[idx] if idx < len(ordered_labels) else idx
        label_name = index_to_label.get(str(class_idx), class_idx)
        token = _metric_label_token(str(label_name))
        df[f"proba_class_{class_idx}"] = probs[:, idx]
        if token:
            df[f"prob_{token}"] = probs[:, idx]
    if extra_columns is not None:
        extra_df = extra_columns if isinstance(extra_columns, pd.DataFrame) else pd.DataFrame(extra_columns)
        _assert_same_length_arrays(
            context="_build_prediction_export_dataframe:extra_columns",
            export_df=df,
            extra_columns=extra_df,
        )
        df = pd.concat([df, extra_df.reset_index(drop=True)], axis=1)
    return df


def _add_named_per_class_metrics(
    metrics: dict[str, Any],
    per_class_metrics: dict[str, dict[str, float]] | None,
    class_index_to_label: dict[str, str],
) -> None:
    if not isinstance(per_class_metrics, dict):
        return
    for class_idx, class_metrics in per_class_metrics.items():
        label_name = class_index_to_label.get(str(class_idx), str(class_idx))
        label_token = _metric_label_token(label_name)
        precision_val = float(class_metrics.get("precision", 0.0))
        recall_val = float(class_metrics.get("recall", 0.0))
        f1_val = float(class_metrics.get("f1", 0.0))
        metrics[f"precision_{label_token}"] = precision_val
        metrics[f"recall_{label_token}"] = recall_val
        metrics[f"f1_{label_token}"] = f1_val
        metrics[f"test_precision_{label_token}"] = precision_val
        metrics[f"test_recall_{label_token}"] = recall_val
        metrics[f"test_f1_{label_token}"] = f1_val


def _add_named_per_class_metrics_with_suffix(
    metrics: dict[str, Any],
    per_class_metrics: dict[str, dict[str, float]] | None,
    class_index_to_label: dict[str, str],
    suffix: str,
) -> None:
    if not isinstance(per_class_metrics, dict):
        return
    normalized_suffix = str(suffix).strip()
    if normalized_suffix and not normalized_suffix.startswith("_"):
        normalized_suffix = f"_{normalized_suffix}"
    for class_idx, class_metrics in per_class_metrics.items():
        label_name = class_index_to_label.get(str(class_idx), str(class_idx))
        label_token = _metric_label_token(label_name)
        precision_val = float(class_metrics.get("precision", 0.0))
        recall_val = float(class_metrics.get("recall", 0.0))
        f1_val = float(class_metrics.get("f1", 0.0))
        metrics[f"precision_{label_token}{normalized_suffix}"] = precision_val
        metrics[f"recall_{label_token}{normalized_suffix}"] = recall_val
        metrics[f"f1_{label_token}{normalized_suffix}"] = f1_val
        metrics[f"test_precision_{label_token}{normalized_suffix}"] = precision_val
        metrics[f"test_recall_{label_token}{normalized_suffix}"] = recall_val
        metrics[f"test_f1_{label_token}{normalized_suffix}"] = f1_val


def _add_named_cv_per_class_metrics(
    metrics: dict[str, Any],
    per_class_metrics: dict[str, dict[str, float]] | None,
    class_index_to_label: dict[str, str],
) -> None:
    if not isinstance(per_class_metrics, dict):
        return
    for class_idx, class_metrics in per_class_metrics.items():
        label_name = class_index_to_label.get(str(class_idx), str(class_idx))
        label_token = _metric_label_token(label_name)
        precision_val = float(class_metrics.get("precision_mean", np.nan))
        recall_val = float(class_metrics.get("recall_mean", np.nan))
        f1_val = float(class_metrics.get("f1_mean", np.nan))
        metrics[f"cv_precision_{label_token}"] = precision_val
        metrics[f"cv_recall_{label_token}"] = recall_val
        metrics[f"cv_f1_{label_token}"] = f1_val
        metrics[f"cv_precision_{label_token}_mean"] = precision_val
        metrics[f"cv_recall_{label_token}_mean"] = recall_val
        metrics[f"cv_f1_{label_token}_mean"] = f1_val


def _add_named_validation_per_class_metrics(
    metrics: dict[str, Any],
    per_class_metrics: dict[str, dict[str, float]] | None,
    class_index_to_label: dict[str, str],
) -> None:
    if not isinstance(per_class_metrics, dict):
        return
    for class_idx, class_metrics in per_class_metrics.items():
        label_name = class_index_to_label.get(str(class_idx), str(class_idx))
        label_token = _metric_label_token(label_name)
        metrics[f"valid_precision_{label_token}"] = float(class_metrics.get("precision", 0.0))
        metrics[f"valid_recall_{label_token}"] = float(class_metrics.get("recall", 0.0))
        metrics[f"valid_f1_{label_token}"] = float(class_metrics.get("f1", 0.0))
