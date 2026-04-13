"""Train-only Isolation Forest outlier filtering utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def apply_outlier_filter(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Filter outliers from training data only, returning metadata for auditability."""
    enabled = bool(config.get("enabled", False))
    method = str(config.get("method", "isolation_forest")).lower()
    if not enabled:
        print("[preprocessing][outlier] enabled=false method=disabled")
        return X_train, y_train, {"enabled": False, "method": None}
    if method != "isolation_forest":
        raise ValueError(f"Unsupported outlier method: '{method}'")

    contamination = float(config.get("contamination", 0.1))
    if not 0 < contamination < 0.5:
        raise ValueError("IsolationForest contamination must be in (0, 0.5).")

    random_state = int(config.get("random_state", 42))
    n_estimators = int(config.get("n_estimators", 200))
    max_samples = config.get("max_samples", "auto")
    max_features = float(config.get("max_features", 1.0))
    bootstrap = bool(config.get("bootstrap", False))
    n_jobs = config.get("n_jobs")
    removal_warning_fraction = float(config.get("removal_warning_fraction", 0.25))
    revert_if_removed_fraction_above = config.get("revert_if_removed_fraction_above")
    min_remaining_rows = config.get("min_remaining_rows")
    numeric_columns = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if not numeric_columns:
        print("[preprocessing][outlier] enabled=true method=isolation_forest skipped=no_numeric_columns")
        return X_train, y_train, {
            "enabled": True,
            "method": "isolation_forest",
            "skipped": True,
            "skip_reason": "no_numeric_columns",
            "estimator_parameters": {
                "contamination": contamination,
                "random_state": random_state,
                "n_estimators": n_estimators,
                "max_samples": max_samples,
                "max_features": max_features,
                "bootstrap": bootstrap,
                "n_jobs": n_jobs,
            },
            "original_train_row_count": int(len(X_train)),
            "filtered_train_row_count": int(len(X_train)),
            "removed_row_count": 0,
            "removed_fraction": 0.0,
            "warning_threshold_fraction": removal_warning_fraction,
            "numeric_feature_count_used": 0,
        }
    X_train_numeric = X_train.loc[:, numeric_columns].copy()
    X_train_numeric = X_train_numeric.where(pd.notna(X_train_numeric), np.nan)
    X_train_numeric = X_train_numeric.fillna(X_train_numeric.median(numeric_only=True))
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
    )
    preds = iso.fit_predict(X_train_numeric)
    keep_mask = preds == 1

    X_f = X_train.loc[keep_mask].reset_index(drop=True)
    y_f = y_train.loc[keep_mask].reset_index(drop=True)
    original_train_rows = int(len(X_train))
    filtered_train_rows = int(keep_mask.sum())
    removed_row_count = int((~keep_mask).sum())
    removed_fraction = float(removed_row_count / original_train_rows) if original_train_rows > 0 else 0.0
    estimator_params = {
        "contamination": contamination,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "n_jobs": n_jobs,
    }
    print(
        "[preprocessing][outlier] "
        f"enabled=true method=isolation_forest "
        f"train_rows_before={original_train_rows} "
        f"train_rows_after={filtered_train_rows} "
        f"removed_rows={removed_row_count} "
        f"removed_fraction={removed_fraction:.4f}"
    )
    if removed_fraction > removal_warning_fraction:
        print(
            "[preprocessing][outlier][warning] "
            f"removed_fraction={removed_fraction:.4f} exceeds warning threshold={removal_warning_fraction:.4f}"
        )
    revert_reason: str | None = None
    if revert_if_removed_fraction_above is not None and removed_fraction > float(revert_if_removed_fraction_above):
        revert_reason = "removed_fraction_above_threshold"
    if min_remaining_rows is not None and filtered_train_rows < int(min_remaining_rows):
        revert_reason = "remaining_rows_below_minimum"
    if revert_reason is not None:
        print(
            "[preprocessing][outlier][revert] "
            f"reason={revert_reason} train_rows_before={original_train_rows} train_rows_after={filtered_train_rows}"
        )
        return X_train, y_train, {
            "enabled": True,
            "method": "isolation_forest",
            "reverted": True,
            "revert_reason": revert_reason,
            "estimator_parameters": estimator_params,
            "original_train_row_count": original_train_rows,
            "filtered_train_row_count": original_train_rows,
            "removed_row_count": 0,
            "removed_fraction": 0.0,
            "warning_threshold_fraction": removal_warning_fraction,
            "numeric_feature_count_used": int(len(numeric_columns)),
            "contamination": contamination,
            "random_state": random_state,
            "n_estimators": n_estimators,
            "n_original": original_train_rows,
            "n_removed": 0,
            "n_remaining": original_train_rows,
        }
    metadata = {
        "enabled": True,
        "method": "isolation_forest",
        "reverted": False,
        "estimator_parameters": estimator_params,
        "original_train_row_count": original_train_rows,
        "filtered_train_row_count": filtered_train_rows,
        "removed_row_count": removed_row_count,
        "removed_fraction": removed_fraction,
        "warning_threshold_fraction": removal_warning_fraction,
        "numeric_feature_count_used": int(len(numeric_columns)),
        "contamination": contamination,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "n_original": original_train_rows,
        "n_removed": removed_row_count,
        "n_remaining": filtered_train_rows,
    }
    return X_f, y_f, metadata
