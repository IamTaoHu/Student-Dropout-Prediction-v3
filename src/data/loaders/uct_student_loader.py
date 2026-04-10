"""Load raw UCT Student tabular data from config-driven paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _resolve_path(path_value: str | Path, base_dir: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if base_dir is None:
        return path
    return (base_dir / path).resolve()


def _resolve_uct_data_source_config(dataset_config: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = dataset_config.get("data_source", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    return {
        "format": str(raw_cfg.get("format", "csv")).strip().lower(),
        "split_mode": str(raw_cfg.get("split_mode", "single_file")).strip().lower(),
        "train_path": raw_cfg.get("train_path"),
        "valid_path": raw_cfg.get("valid_path"),
        "test_path": raw_cfg.get("test_path"),
    }


def _coerce_low_risk_dtypes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    coerced: list[dict[str, Any]] = []
    train_out = train_df.copy()
    test_out = test_df.copy()
    for col in feature_columns:
        train_dtype = train_out[col].dtype
        test_dtype = test_out[col].dtype
        if str(train_dtype) == str(test_dtype):
            continue
        if pd.api.types.is_numeric_dtype(train_dtype) and pd.api.types.is_numeric_dtype(test_dtype):
            train_out[col] = pd.to_numeric(train_out[col], errors="coerce").astype("float64")
            test_out[col] = pd.to_numeric(test_out[col], errors="coerce").astype("float64")
            coerced.append(
                {
                    "column": str(col),
                    "train_dtype_before": str(train_dtype),
                    "test_dtype_before": str(test_dtype),
                    "coerced_to": "float64",
                }
            )
    return train_out, test_out, coerced


def _validate_predefined_split_schema(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if target_col not in train_df.columns:
        raise ValueError(f"Predefined UCI train parquet is missing target column '{target_col}'.")
    if target_col not in test_df.columns:
        raise ValueError(f"Predefined UCI test parquet is missing target column '{target_col}'.")

    train_features = [col for col in train_df.columns if col != target_col]
    test_features = [col for col in test_df.columns if col != target_col]
    train_set = set(train_features)
    test_set = set(test_features)
    if train_set != test_set:
        missing_in_test = [col for col in train_features if col not in test_set]
        extra_in_test = [col for col in test_features if col not in train_set]
        raise ValueError(
            "Predefined UCI parquet schema mismatch after removing the target column. "
            f"missing_in_test={missing_in_test[:10]} extra_in_test={extra_in_test[:10]}"
        )

    reordered_test = test_df.loc[:, train_features + [target_col]].copy()
    reordered_train = train_df.loc[:, train_features + [target_col]].copy()
    dtype_mismatches: list[dict[str, Any]] = []
    for col in train_features:
        train_dtype = reordered_train[col].dtype
        test_dtype = reordered_test[col].dtype
        if str(train_dtype) != str(test_dtype):
            dtype_mismatches.append(
                {
                    "column": str(col),
                    "train_dtype": str(train_dtype),
                    "test_dtype": str(test_dtype),
                }
            )
    reordered_train, reordered_test, coerced_columns = _coerce_low_risk_dtypes(
        reordered_train,
        reordered_test,
        train_features,
    )
    return reordered_train, reordered_test, {
        "target_column": str(target_col),
        "feature_column_count": int(len(train_features)),
        "dtype_mismatches": dtype_mismatches,
        "low_risk_coercions": coerced_columns,
        "schema_validation_passed": True,
    }


def load_uct_student_dataframe(dataset_config: dict[str, Any], base_dir: Path | None = None) -> pd.DataFrame:
    """Load UCT Student CSV as a single DataFrame with basic validation."""
    data_source_cfg = _resolve_uct_data_source_config(dataset_config)
    if data_source_cfg["format"] == "parquet" and data_source_cfg["split_mode"] == "predefined":
        raise ValueError(
            "load_uct_student_dataframe does not support predefined parquet split mode. "
            "Use load_uct_student_predefined_splits for config-driven train/test parquet ingestion."
        )
    paths_cfg = dataset_config.get("paths", {})
    csv_path = paths_cfg.get("raw_file")
    if not csv_path:
        raw_root = paths_cfg.get("raw_root")
        default_file = dataset_config.get("source", {}).get("filename", "uct_student.csv")
        csv_path = str(Path(raw_root) / default_file) if raw_root else default_file

    delimiter = str(dataset_config.get("source", {}).get("delimiter", ","))
    encoding = str(dataset_config.get("source", {}).get("encoding", "utf-8"))
    target_col = dataset_config.get("schema", {}).get("outcome_column")
    if not target_col:
        raise ValueError("UCT dataset config must define schema.outcome_column.")

    resolved_path = _resolve_path(csv_path, base_dir)
    if not resolved_path.exists():
        raise FileNotFoundError(f"UCT Student raw file not found: {resolved_path}")

    df = pd.read_csv(resolved_path, sep=delimiter, encoding=encoding)
    if target_col not in df.columns:
        raise ValueError(
            f"UCT Student target column '{target_col}' missing from raw file. "
            f"Columns available: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError("UCT Student raw dataset is empty.")
    return df


def load_uct_student_predefined_splits(
    dataset_config: dict[str, Any],
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Load predefined UCI parquet train/test splits with schema validation."""
    data_source_cfg = _resolve_uct_data_source_config(dataset_config)
    if data_source_cfg["format"] != "parquet" or data_source_cfg["split_mode"] != "predefined":
        raise ValueError("Predefined split loader requires data_source.format=parquet and data_source.split_mode=predefined.")

    train_path = data_source_cfg.get("train_path")
    valid_path = data_source_cfg.get("valid_path")
    test_path = data_source_cfg.get("test_path")
    if not train_path or not test_path:
        raise ValueError("Predefined UCI parquet mode requires data_source.train_path and data_source.test_path.")

    target_col = dataset_config.get("schema", {}).get("outcome_column")
    if not target_col:
        raise ValueError("UCT dataset config must define schema.outcome_column.")

    resolved_train_path = _resolve_path(train_path, base_dir)
    resolved_valid_path = _resolve_path(valid_path, base_dir) if valid_path else None
    resolved_test_path = _resolve_path(test_path, base_dir)
    if not resolved_train_path.exists():
        raise FileNotFoundError(f"UCI predefined train parquet not found: {resolved_train_path}")
    if resolved_valid_path is not None and not resolved_valid_path.exists():
        raise FileNotFoundError(f"UCI predefined valid parquet not found: {resolved_valid_path}")
    if not resolved_test_path.exists():
        raise FileNotFoundError(f"UCI predefined test parquet not found: {resolved_test_path}")

    train_df = pd.read_parquet(resolved_train_path)
    valid_df = pd.read_parquet(resolved_valid_path) if resolved_valid_path is not None else None
    test_df = pd.read_parquet(resolved_test_path)
    if train_df.empty:
        raise ValueError("UCI predefined train parquet is empty.")
    if valid_df is not None and valid_df.empty:
        raise ValueError("UCI predefined valid parquet is empty.")
    if test_df.empty:
        raise ValueError("UCI predefined test parquet is empty.")

    train_df, test_df, schema_report = _validate_predefined_split_schema(
        train_df,
        test_df,
        target_col=str(target_col),
    )
    if valid_df is not None:
        train_df, valid_df, valid_schema_report = _validate_predefined_split_schema(
            train_df,
            valid_df,
            target_col=str(target_col),
        )
    else:
        valid_schema_report = None
    print("[dataset][uci] source_format=parquet split_mode=predefined")
    print(
        "[dataset][uci] "
        f"loaded_train_rows={int(len(train_df))} "
        f"loaded_valid_rows={int(len(valid_df)) if valid_df is not None else 0} "
        f"loaded_test_rows={int(len(test_df))} "
        f"target_column={target_col}"
    )
    if schema_report["dtype_mismatches"]:
        print(f"[dataset][uci] dtype_mismatches={schema_report['dtype_mismatches']}")
    if schema_report["low_risk_coercions"]:
        print(f"[dataset][uci] low_risk_dtype_coercions={schema_report['low_risk_coercions']}")
    if valid_schema_report and valid_schema_report["dtype_mismatches"]:
        print(f"[dataset][uci] valid_dtype_mismatches={valid_schema_report['dtype_mismatches']}")
    if valid_schema_report and valid_schema_report["low_risk_coercions"]:
        print(f"[dataset][uci] valid_low_risk_dtype_coercions={valid_schema_report['low_risk_coercions']}")
    print("[dataset][uci] predefined parquet schema validation passed")
    return {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
        "schema_report": schema_report,
        "valid_schema_report": valid_schema_report,
        "resolved_paths": {
            "train_path": str(resolved_train_path),
            "valid_path": str(resolved_valid_path) if resolved_valid_path is not None else None,
            "test_path": str(resolved_test_path),
        },
    }


def load_uct_student_tables(dataset_config: dict[str, Any], base_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Backward-compatible wrapper returning UCT data under a logical table key."""
    return {"students": load_uct_student_dataframe(dataset_config, base_dir=base_dir)}
