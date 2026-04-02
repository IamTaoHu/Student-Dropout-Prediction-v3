"""Feature builder for UCT Student tabular benchmark experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return (numerator / denom).replace([np.inf, -np.inf], np.nan)


def build_uct_student_features(adapted: dict[str, Any] | pd.DataFrame, feature_config: dict[str, Any]) -> pd.DataFrame:
    """Create a clean UCT feature table while preserving original columns."""
    if isinstance(adapted, dict):
        if "data" not in adapted:
            raise KeyError("UCT feature builder expects adapted schema with 'data' key.")
        df = adapted["data"].copy()
        id_column = adapted.get("id_column")
        target_column = adapted.get("target_column")
    else:
        df = adapted.copy()
        id_column = feature_config.get("id_column")
        target_column = feature_config.get("target_column")

    drop_columns = set(feature_config.get("drop_columns", []))
    if drop_columns:
        existing = [c for c in drop_columns if c in df.columns]
        df = df.drop(columns=existing)

    derive_safe_features = bool(feature_config.get("derive_safe_features", True))
    if derive_safe_features:
        if {"studied_credits", "num_of_prev_attempts"}.issubset(df.columns):
            df["credits_per_previous_attempt"] = _safe_ratio(
                df["studied_credits"].astype(float), df["num_of_prev_attempts"].astype(float).replace(0, 1)
            )
        if {"attendance_rate", "engagement_score"}.issubset(df.columns):
            df["attendance_engagement_interaction"] = (
                pd.to_numeric(df["attendance_rate"], errors="coerce")
                * pd.to_numeric(df["engagement_score"], errors="coerce")
            )

        feature_candidates = [c for c in df.columns if c not in {id_column, target_column}]
        numeric_candidates = [c for c in feature_candidates if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_candidates:
            df["missing_numeric_count"] = df[numeric_candidates].isna().sum(axis=1)

    # Keep native ordering: id/target first, then feature columns.
    ordered_cols: list[str] = []
    for col in [id_column, target_column]:
        if col and col in df.columns:
            ordered_cols.append(col)
    ordered_cols.extend([c for c in df.columns if c not in ordered_cols])
    return df[ordered_cols]
