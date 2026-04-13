from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.experiment.shared_context import PreprocessingExecutionResult
from src.preprocessing.balancing import apply_balancing
from src.preprocessing.outlier import apply_outlier_filter
from src.preprocessing.tabular_pipeline import detect_feature_types, run_tabular_preprocessing


def _drop_rows_with_missing_values(
    df: pd.DataFrame,
    preprocessing_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing_cfg = preprocessing_cfg.get("missing_values", {})
    drop_rows = bool(preprocessing_cfg.get("drop_missing_rows", False))
    drop_rows = bool(preprocessing_cfg.get("drop_rows_with_missing", drop_rows))
    if isinstance(missing_cfg, dict):
        drop_rows = bool(missing_cfg.get("drop_rows", drop_rows))
    if not drop_rows:
        return df, {"enabled": False, "method": "imputation"}

    original_count = int(len(df))
    dropped_df = df.dropna(axis=0).reset_index(drop=True)
    removed = original_count - int(len(dropped_df))
    return dropped_df, {
        "enabled": True,
        "method": "drop_rows",
        "n_original": original_count,
        "n_removed": removed,
        "n_remaining": int(len(dropped_df)),
    }


def _resolve_categorical_encoding_config(preprocessing_cfg: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = preprocessing_cfg.get("categorical_encoding")
    if isinstance(raw_cfg, dict):
        mode = str(raw_cfg.get("mode", preprocessing_cfg.get("encoding", "onehot"))).strip().lower()
        return {
            "mode": mode,
            "handle_unknown": str(raw_cfg.get("handle_unknown", "ignore")).strip().lower(),
            "drop": raw_cfg.get("drop"),
            "lock_category_vocabulary_from_pre_split_train": bool(
                raw_cfg.get("lock_category_vocabulary_from_pre_split_train", False)
            ),
            "vocabulary_source": str(
                raw_cfg.get("vocabulary_source", "categorical_dtype_or_full_pre_split_train")
            ).strip(),
        }
    return {
        "mode": str(raw_cfg or preprocessing_cfg.get("encoding", "onehot")).strip().lower(),
        "handle_unknown": "ignore",
        "drop": None,
        "lock_category_vocabulary_from_pre_split_train": False,
        "vocabulary_source": "subset_fit_only",
    }


def _build_locked_onehot_vocabulary(
    feature_df: pd.DataFrame,
    preprocess_cfg: dict[str, Any],
) -> dict[str, Any]:
    target_column = str(preprocess_cfg.get("target_column", "target"))
    id_columns = list(preprocess_cfg.get("id_columns", []))
    forbidden_columns = set(preprocess_cfg.get("forbidden_feature_columns", []))
    source_features = feature_df.drop(
        columns=[c for c in [target_column, *id_columns, *sorted(forbidden_columns)] if c in feature_df.columns]
    ).copy()
    source_features = source_features.where(pd.notna(source_features), np.nan)

    numeric_cols, categorical_cols = detect_feature_types(source_features)
    vocabulary: dict[str, list[Any]] = {}
    counts: dict[str, int] = {}
    labels: dict[str, list[str]] = {}
    sources: dict[str, str] = {}

    for column in categorical_cols:
        series = source_features[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            categories = list(series.cat.categories)
            source = "categorical_dtype"
        else:
            categories = pd.Index(series.dropna().astype(object).unique()).tolist()
            source = "full_pre_split_train_unique_values"
        vocabulary[column] = list(categories)
        counts[column] = int(len(categories))
        labels[column] = [str(value) for value in categories]
        sources[column] = source

    total_encoded = int(sum(counts.values()))
    return {
        "categories": vocabulary,
        "source": str(preprocess_cfg.get("onehot_categories_source", "categorical_dtype_or_full_pre_split_train")),
        "column_counts": counts,
        "column_labels": labels,
        "column_sources": sources,
        "categorical_column_count": int(len(categorical_cols)),
        "numeric_column_count": int(len(numeric_cols)),
        "encoded_categorical_feature_count": total_encoded,
        "preprocessed_feature_count": int(total_encoded + len(numeric_cols)),
    }


def _prepare_preprocessing_config(
    exp_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    id_column: str,
    source_target_col: str,
) -> dict[str, Any]:
    p_cfg = exp_cfg.get("preprocessing", {})
    ds_p_cfg = dataset_cfg.get("preprocessing", {})
    categorical_encoding_cfg = _resolve_categorical_encoding_config(p_cfg)
    imputation = str(p_cfg.get("imputation", "median_mode")).lower()
    scaling_raw = str(p_cfg.get("numeric_scaling", p_cfg.get("scaling", "standard"))).strip().lower()
    encoding_raw = str(categorical_encoding_cfg.get("mode", p_cfg.get("encoding", "onehot"))).strip().lower()
    forbidden_columns = {"final_result"}
    forbidden_columns.update(list(p_cfg.get("forbidden_feature_columns", []) or []))
    forbidden_columns.update(list(p_cfg.get("drop_columns", []) or []))
    forbidden_columns.update(list(ds_p_cfg.get("forbidden_feature_columns", []) or []))
    if source_target_col and source_target_col != "target":
        forbidden_columns.add(source_target_col)
    return {
        "target_column": "target",
        "id_columns": [id_column],
        "forbidden_feature_columns": sorted(forbidden_columns),
        "numeric_imputation": "median" if "median" in imputation else "mean",
        "categorical_imputation": "most_frequent",
        "scaling": scaling_raw in {"standard", "true", "1", "enabled"},
        "onehot": encoding_raw == "onehot",
        "onehot_handle_unknown": str(categorical_encoding_cfg.get("handle_unknown", "ignore")).strip().lower(),
        "onehot_drop": categorical_encoding_cfg.get("drop"),
        "lock_category_vocabulary_from_pre_split_train": bool(
            categorical_encoding_cfg.get("lock_category_vocabulary_from_pre_split_train", False)
        ),
        "onehot_categories_source": str(categorical_encoding_cfg.get("vocabulary_source", "subset_fit_only")).strip(),
    }


def _resolve_outlier_config(
    exp_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    preprocessing_cfg = exp_cfg.get("preprocessing", {}) if isinstance(exp_cfg.get("preprocessing", {}), dict) else {}
    outlier_cfg = preprocessing_cfg.get("outlier", {"enabled": False})
    if not isinstance(outlier_cfg, dict):
        outlier_cfg = {"enabled": False}
    isolation_cfg = preprocessing_cfg.get("isolation_forest", {})
    if not isinstance(isolation_cfg, dict):
        isolation_cfg = {}
    enabled = bool(
        outlier_cfg.get(
            "enabled",
            preprocessing_cfg.get("apply_isolation_forest", isolation_cfg.get("enabled", False)),
        )
    )
    resolved = {
        **outlier_cfg,
        **isolation_cfg,
        "enabled": enabled,
        "method": str(outlier_cfg.get("method", isolation_cfg.get("method", "isolation_forest"))).lower(),
    }
    resolved["random_state"] = int(resolved.get("random_state", seed))
    return resolved


def _resolve_balancing_config(
    exp_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    preprocessing_cfg = exp_cfg.get("preprocessing", {}) if isinstance(exp_cfg.get("preprocessing", {}), dict) else {}
    balancing_cfg = preprocessing_cfg.get("balancing", {"enabled": False})
    if not isinstance(balancing_cfg, dict):
        balancing_cfg = {"enabled": False}
    smote_cfg = preprocessing_cfg.get("smote", {})
    if not isinstance(smote_cfg, dict):
        smote_cfg = {}
    enabled = bool(
        balancing_cfg.get(
            "enabled",
            preprocessing_cfg.get("apply_smote_on_train_only", smote_cfg.get("enabled", False)),
        )
    )
    resolved = {
        **balancing_cfg,
        **smote_cfg,
        "enabled": enabled,
        "method": str(balancing_cfg.get("method", smote_cfg.get("method", "smote"))).lower(),
    }
    resolved["random_state"] = int(resolved.get("random_state", seed))
    return resolved


def _resolve_onehot_metadata_and_validate(
    artifacts: Any,
    preprocess_cfg: dict[str, Any],
    preprocessing_exp_cfg: dict[str, Any],
) -> dict[str, Any]:
    metadata = artifacts.metadata if hasattr(artifacts, "metadata") and isinstance(artifacts.metadata, dict) else {}
    categorical_columns = metadata.get("categorical_columns", [])
    numeric_columns = metadata.get("numeric_columns", [])
    if not isinstance(categorical_columns, list):
        categorical_columns = []
    if not isinstance(numeric_columns, list):
        numeric_columns = []
    encoded_categorical_feature_count = int(metadata.get("encoded_categorical_feature_count", 0) or 0)
    per_column_counts = metadata.get("onehot_column_category_counts", {})
    if not isinstance(per_column_counts, dict):
        per_column_counts = {}
    per_column_labels = metadata.get("onehot_column_category_labels", {})
    if not isinstance(per_column_labels, dict):
        per_column_labels = {}
    locked_vocabulary_mode = bool(metadata.get("onehot_categories_locked", False))
    vocabulary_source = metadata.get("onehot_categories_source")

    require_onehot = bool(preprocessing_exp_cfg.get("require_onehot_encoding", False))
    if require_onehot and not bool(preprocess_cfg.get("onehot", False)):
        raise ValueError("preprocessing.require_onehot_encoding=true but categorical one-hot encoding is disabled.")
    if require_onehot and len(categorical_columns) > 0 and encoded_categorical_feature_count <= 0:
        raise ValueError(
            "preprocessing.require_onehot_encoding=true but encoded categorical feature count is zero."
        )

    return {
        "onehot_enabled": bool(preprocess_cfg.get("onehot", False)),
        "require_onehot_encoding": require_onehot,
        "input_numeric_feature_count": int(len(numeric_columns)),
        "input_categorical_feature_count": int(len(categorical_columns)),
        "encoded_categorical_feature_count": int(encoded_categorical_feature_count),
        "preprocessed_feature_count": int(artifacts.X_train.shape[1]),
        "stable_locked_vocabulary_mode": locked_vocabulary_mode,
        "vocabulary_source": str(vocabulary_source) if vocabulary_source else None,
        "per_column_category_counts": {
            str(column): int(count)
            for column, count in per_column_counts.items()
        },
        "per_column_category_labels": {
            str(column): [str(value) for value in values]
            for column, values in per_column_labels.items()
            if isinstance(values, list)
        },
    }


def run_benchmark_preprocessing(
    *,
    exp_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    id_column: str,
    source_target_col: str,
    splits: dict[str, pd.DataFrame],
    pre_split_train_feature_source_for_vocab: pd.DataFrame | None,
    seed: int,
) -> PreprocessingExecutionResult:
    preprocess_cfg = _prepare_preprocessing_config(
        exp_cfg,
        dataset_cfg,
        id_column=id_column,
        source_target_col=source_target_col,
    )
    if (
        bool(preprocess_cfg.get("onehot", False))
        and bool(preprocess_cfg.get("lock_category_vocabulary_from_pre_split_train", False))
        and isinstance(pre_split_train_feature_source_for_vocab, pd.DataFrame)
        and not pre_split_train_feature_source_for_vocab.empty
    ):
        locked_vocabulary = _build_locked_onehot_vocabulary(
            pre_split_train_feature_source_for_vocab,
            preprocess_cfg,
        )
        preprocess_cfg["onehot_categories"] = locked_vocabulary["categories"]
        preprocess_cfg["onehot_categories_source"] = locked_vocabulary["source"]
        print(
            "[preprocessing][onehot] "
            f"stable_locked_vocabulary_mode=true "
            f"vocabulary_source={locked_vocabulary['source']} "
            f"categorical_columns={locked_vocabulary['categorical_column_count']} "
            f"numerical_columns={locked_vocabulary['numeric_column_count']} "
            f"encoded_categorical_feature_count={locked_vocabulary['encoded_categorical_feature_count']} "
            f"preprocessed_feature_count={locked_vocabulary['preprocessed_feature_count']}"
        )
        print(
            "[preprocessing][onehot] "
            f"per_column_category_counts={json.dumps(locked_vocabulary['column_counts'], sort_keys=True)}"
        )
    artifacts = run_tabular_preprocessing(splits, preprocess_cfg)
    onehot_metadata = _resolve_onehot_metadata_and_validate(
        artifacts=artifacts,
        preprocess_cfg=preprocess_cfg,
        preprocessing_exp_cfg=(
            exp_cfg.get("preprocessing", {}) if isinstance(exp_cfg.get("preprocessing", {}), dict) else {}
        ),
    )
    if bool(onehot_metadata.get("onehot_enabled", False)):
        artifacts.metadata["locked_category_vocabulary_summary"] = {
            "stable_locked_vocabulary_mode": bool(onehot_metadata.get("stable_locked_vocabulary_mode", False)),
            "vocabulary_source": onehot_metadata.get("vocabulary_source"),
            "categorical_column_count": int(onehot_metadata.get("input_categorical_feature_count", 0)),
            "numerical_column_count": int(onehot_metadata.get("input_numeric_feature_count", 0)),
            "encoded_categorical_feature_count": int(onehot_metadata.get("encoded_categorical_feature_count", 0)),
            "total_final_feature_count": int(onehot_metadata.get("preprocessed_feature_count", 0)),
            "columns": [
                {
                    "column_name": str(column),
                    "category_count": int(count),
                    "category_labels": onehot_metadata.get("per_column_category_labels", {}).get(str(column), []),
                }
                for column, count in onehot_metadata.get("per_column_category_counts", {}).items()
            ],
        }
        print(
            "[preprocessing][onehot] "
            f"stable_locked_vocabulary_mode={str(onehot_metadata.get('stable_locked_vocabulary_mode', False)).lower()} "
            f"vocabulary_source={onehot_metadata.get('vocabulary_source')} "
            f"categorical_column_count={onehot_metadata.get('input_categorical_feature_count', 0)} "
            f"numerical_column_count={onehot_metadata.get('input_numeric_feature_count', 0)} "
            f"encoded_categorical_feature_count={onehot_metadata.get('encoded_categorical_feature_count', 0)} "
            f"preprocessed_feature_count={onehot_metadata.get('preprocessed_feature_count', 0)}"
        )

    outlier_cfg = _resolve_outlier_config(exp_cfg=exp_cfg, seed=seed)
    print(
        "[preprocessing][train_only] "
        f"outlier_enabled={bool(outlier_cfg.get('enabled', False))} "
        f"outlier_method={outlier_cfg.get('method', 'disabled')}"
    )
    outlier_applied_before_preprocessing = bool(outlier_cfg.get("apply_before_preprocessing", False))
    if outlier_applied_before_preprocessing:
        outlier_meta = dict(artifacts.metadata.get("train_only_outlier", {}))
        X_train_filtered = artifacts.X_train
        y_train_filtered = artifacts.y_train
    else:
        X_train_filtered, y_train_filtered, outlier_meta = apply_outlier_filter(
            artifacts.X_train,
            artifacts.y_train,
            outlier_cfg,
        )
    balancing_cfg = _resolve_balancing_config(exp_cfg=exp_cfg, seed=seed)
    print(
        "[preprocessing][train_only] "
        f"smote_enabled={bool(balancing_cfg.get('enabled', False))} "
        f"smote_method={balancing_cfg.get('method', 'disabled')}"
    )
    X_train_bal, y_train_bal, balancing_meta = apply_balancing(X_train_filtered, y_train_filtered, balancing_cfg)
    pre_outlier_source = (
        splits["train"]["target"]
        if outlier_applied_before_preprocessing and "train" in splits and "target" in splits["train"].columns
        else artifacts.y_train
    )
    pre_outlier_class_distribution = {
        str(k): int(v) for k, v in pre_outlier_source.value_counts(dropna=False).sort_index().to_dict().items()
    }
    post_outlier_class_distribution = {
        str(k): int(v) for k, v in y_train_filtered.value_counts(dropna=False).sort_index().to_dict().items()
    }
    return PreprocessingExecutionResult(
        preprocess_cfg=preprocess_cfg,
        artifacts=artifacts,
        onehot_metadata=onehot_metadata,
        outlier_cfg=outlier_cfg,
        outlier_meta=outlier_meta,
        X_train_filtered=X_train_filtered,
        y_train_filtered=y_train_filtered,
        balancing_cfg=balancing_cfg,
        balancing_meta=balancing_meta,
        X_train_bal=X_train_bal,
        y_train_bal=y_train_bal,
        pre_outlier_class_distribution=pre_outlier_class_distribution,
        post_outlier_class_distribution=post_outlier_class_distribution,
    )
