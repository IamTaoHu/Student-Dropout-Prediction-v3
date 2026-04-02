"""Tabular preprocessing pipeline with feature-name-preserving transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_FORBIDDEN_FEATURE_COLUMNS = {"final_result"}


@dataclass(frozen=True)
class TabularPipelineArtifacts:
    """Preprocessed splits and metadata for downstream modeling."""

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    metadata: dict[str, Any]


def detect_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Detect numeric and categorical columns from a DataFrame."""
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _build_transformer(
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: dict[str, Any],
) -> ColumnTransformer:
    preprocessing_cfg = config or {}
    numeric_imputer_strategy = str(preprocessing_cfg.get("numeric_imputation", "median"))
    categorical_imputer_strategy = str(preprocessing_cfg.get("categorical_imputation", "most_frequent"))
    use_scaler = bool(preprocessing_cfg.get("scaling", True))
    use_onehot = bool(preprocessing_cfg.get("onehot", True))

    num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy=numeric_imputer_strategy))]
    if use_scaler:
        num_steps.append(("scaler", StandardScaler()))
    num_pipeline = Pipeline(steps=num_steps)

    if use_onehot:
        try:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    else:
        onehot = "passthrough"

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=categorical_imputer_strategy, fill_value="missing")),
            ("encoder", onehot),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _extract_features_target(
    df: pd.DataFrame,
    target_column: str,
    id_columns: list[str] | None,
    forbidden_feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    forbidden_cols = set(DEFAULT_FORBIDDEN_FEATURE_COLUMNS)
    if forbidden_feature_columns:
        forbidden_cols.update(forbidden_feature_columns)

    drop_cols = [target_column]
    if id_columns:
        drop_cols.extend([col for col in id_columns if col in df.columns])
    drop_cols.extend([col for col in forbidden_cols if col in df.columns])

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    offending_columns = sorted(set(X.columns).intersection(forbidden_cols | {target_column}))
    if offending_columns:
        raise ValueError(
            "Forbidden target/source columns leaked into feature matrix X: "
            f"{offending_columns}"
        )
    y = df[target_column].copy()
    return X, y


def run_tabular_preprocessing(split_data: dict[str, pd.DataFrame], config: dict[str, Any]) -> TabularPipelineArtifacts:
    """Fit preprocessing on train split and transform train/valid/test consistently."""
    target_column = str(config.get("target_column", "target"))
    id_columns = list(config.get("id_columns", []))
    raw_forbidden = config.get("forbidden_feature_columns", [])
    if isinstance(raw_forbidden, str):
        forbidden_feature_columns = [raw_forbidden]
    else:
        forbidden_feature_columns = list(raw_forbidden or [])

    if "train" not in split_data or "test" not in split_data:
        raise KeyError("split_data must contain at least 'train' and 'test' keys.")
    if target_column not in split_data["train"].columns:
        raise KeyError(f"Target column '{target_column}' not found in train split.")

    train_df = split_data["train"]
    valid_df = split_data.get("valid")
    test_df = split_data["test"]
    if valid_df is None or valid_df.empty:
        valid_df = pd.DataFrame(columns=train_df.columns)

    X_train_raw, y_train = _extract_features_target(
        train_df,
        target_column,
        id_columns,
        forbidden_feature_columns=forbidden_feature_columns,
    )
    X_valid_raw, y_valid = (
        _extract_features_target(
            valid_df,
            target_column,
            id_columns,
            forbidden_feature_columns=forbidden_feature_columns,
        )
        if not valid_df.empty
        else (pd.DataFrame(columns=X_train_raw.columns), pd.Series(dtype=y_train.dtype))
    )
    X_test_raw, y_test = _extract_features_target(
        test_df,
        target_column,
        id_columns,
        forbidden_feature_columns=forbidden_feature_columns,
    )

    numeric_cols, categorical_cols = detect_feature_types(X_train_raw)
    transformer = _build_transformer(numeric_cols, categorical_cols, config=config)
    X_train_arr = transformer.fit_transform(X_train_raw)

    feature_names = transformer.get_feature_names_out().tolist()
    X_train = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train_raw.index)

    if X_valid_raw.empty:
        X_valid = pd.DataFrame(columns=feature_names)
    else:
        X_valid = pd.DataFrame(transformer.transform(X_valid_raw), columns=feature_names, index=X_valid_raw.index)
    X_test = pd.DataFrame(transformer.transform(X_test_raw), columns=feature_names, index=X_test_raw.index)

    metadata = {
        "target_column": target_column,
        "id_columns": id_columns,
        "forbidden_feature_columns": sorted(set(DEFAULT_FORBIDDEN_FEATURE_COLUMNS).union(forbidden_feature_columns)),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "output_feature_names": feature_names,
        "transformer": transformer,
    }
    return TabularPipelineArtifacts(X_train, X_valid, X_test, y_train, y_valid, y_test, metadata)
