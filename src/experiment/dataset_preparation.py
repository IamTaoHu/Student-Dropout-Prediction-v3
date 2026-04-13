from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.adapters.oulad_adapter import adapt_oulad_schema
from src.data.adapters.uct_student_adapter import adapt_uct_student_schema
from src.data.feature_builders.oulad_paper_features import build_oulad_paper_features
from src.data.feature_builders.uci_student_paper_style_features import (
    build_uci_student_paper_style_features,
)
from src.data.feature_builders.uct_student_features import build_uct_student_features
from src.data.loaders.oulad_loader import load_oulad_tables
from src.data.loaders.uct_student_loader import (
    load_uct_student_dataframe,
    load_uct_student_predefined_splits,
)
from src.data.splits.stratified_split import SplitConfig, stratified_train_valid_test_split
from src.data.target_mapping.binary import map_binary_target
from src.data.target_mapping.four_class import map_four_class_target
from src.data.target_mapping.three_class import map_three_class_target
from src.experiment.config_resolution import _normalize_dataset_name
from src.experiment.preprocessing_pipeline import _drop_rows_with_missing_values
from src.experiment.schema_validation import align_feature_schema, validate_feature_schema
from src.experiment.shared_context import DatasetPreparationResult

SUPPORTED_DATASETS = {"uct_student", "oulad"}


def _build_uct_feature_table(
    adapted: dict[str, Any] | pd.DataFrame,
    feature_cfg: dict[str, Any],
) -> pd.DataFrame:
    builder_token = str(feature_cfg.get("builder", "uci_student_features")).strip().lower()
    print(f"[features][uci] builder={builder_token}")
    if builder_token == "uci_student_features":
        return build_uct_student_features(adapted, feature_cfg)
    if builder_token == "uci_student_paper_style_features":
        return build_uci_student_paper_style_features(adapted, feature_cfg)
    raise ValueError(
        "Unsupported UCT/UCI feature builder "
        f"'{builder_token}'. Supported builders: "
        "['uci_student_features', 'uci_student_paper_style_features']."
    )


def _build_feature_table(dataset_cfg: dict[str, Any]) -> tuple[pd.DataFrame, str, str, str]:
    raw_dataset_name = str(dataset_cfg.get("dataset", {}).get("name", ""))
    dataset_name = _normalize_dataset_name(raw_dataset_name)
    if not dataset_name:
        raise ValueError(
            "Dataset name is missing in dataset config under dataset.name. "
            "Expected one of: uct_student, oulad."
        )
    schema_cfg = dataset_cfg.get("schema", {})
    if dataset_name == "uct_student":
        raw_df = load_uct_student_dataframe(dataset_cfg)
        adapted = adapt_uct_student_schema(raw_df, schema_cfg)
        features = _build_uct_feature_table(adapted, dataset_cfg.get("features", {}))
        return features, dataset_name, adapted["id_column"], adapted["target_column"]
    if dataset_name == "oulad":
        raw_tables = load_oulad_tables(dataset_cfg)
        adapted = adapt_oulad_schema(raw_tables, schema_cfg)
        features = build_oulad_paper_features(adapted, dataset_cfg.get("features", {}))
        return features, dataset_name, "id_student", schema_cfg.get("outcome_column", "final_result")
    raise ValueError(
        "Unsupported dataset name "
        f"'{raw_dataset_name}' (normalized: '{dataset_name}'). "
        f"Supported datasets: {sorted(SUPPORTED_DATASETS)}."
    )


def _resolve_dataset_source_config(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = dataset_cfg.get("data_source", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    return {
        "format": str(raw_cfg.get("format", "csv")).strip().lower(),
        "split_mode": str(raw_cfg.get("split_mode", "single_file")).strip().lower(),
        "train_path": raw_cfg.get("train_path"),
        "valid_path": raw_cfg.get("valid_path"),
        "test_path": raw_cfg.get("test_path"),
    }


def _map_target(
    df: pd.DataFrame,
    dataset_name: str,
    source_target_col: str,
    formulation: str,
    mapping: dict[str, int] | None,
) -> pd.Series:
    if formulation == "binary":
        return map_binary_target(df, source_target_col, dataset_name=dataset_name, mapping=mapping)
    if formulation == "three_class":
        return map_three_class_target(
            df,
            source_column=source_target_col,
            dataset_name=dataset_name,
            mapping=mapping,
        )
    if formulation == "four_class":
        return map_four_class_target(
            df,
            source_column=source_target_col,
            dataset_name=dataset_name,
            mapping=mapping,
        )
    raise ValueError(f"Unsupported target formulation '{formulation}'.")


def _prepare_feature_df_with_target(
    feature_df: pd.DataFrame,
    *,
    dataset_name: str,
    source_target_col: str,
    formulation: str,
    target_mapping: dict[str, int] | None,
) -> pd.DataFrame:
    prepared = feature_df.copy()
    mapped_target = _map_target(prepared, dataset_name, source_target_col, formulation, target_mapping)
    if mapped_target is None:
        raise ValueError(
            "Target mapping returned None for "
            f"dataset='{dataset_name}', formulation='{formulation}', "
            f"source_target_col='{source_target_col}'."
        )
    if not isinstance(mapped_target, pd.Series):
        raise ValueError(
            "Target mapping must return pandas Series, got "
            f"{type(mapped_target).__name__} for dataset='{dataset_name}', "
            f"source_target_col='{source_target_col}'."
        )
    if len(mapped_target) != len(prepared):
        raise ValueError(
            "Mapped target length mismatch: "
            f"len(mapped_target)={len(mapped_target)} vs len(feature_df)={len(prepared)} "
            f"for dataset='{dataset_name}', source_target_col='{source_target_col}'."
        )
    prepared["target"] = mapped_target
    columns_to_drop = [col for col in [source_target_col] if col and col != "target"]
    if columns_to_drop:
        prepared = prepared.drop(columns=columns_to_drop, errors="ignore")
    if prepared["target"].isna().any():
        raise ValueError(
            "Target mapping produced null values for "
            f"dataset='{dataset_name}', source_target_col='{source_target_col}'."
        )
    return prepared


def _resolve_target_mapping(
    exp_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    formulation: str,
) -> dict[str, int] | None:
    override_cfg = exp_cfg.get("target_mapping_override", {})
    if not isinstance(override_cfg, dict):
        override_cfg = {}
    ds_cfg = dataset_cfg.get("target_mappings", {})
    if not isinstance(ds_cfg, dict):
        ds_cfg = {}
    override_mapping = override_cfg.get(formulation)
    if isinstance(override_mapping, dict):
        return override_mapping
    mapping = ds_cfg.get(formulation)
    if isinstance(mapping, dict):
        return mapping
    return None


def _resolve_class_metadata(
    exp_cfg: dict[str, Any],
    mapping: dict[str, int] | None,
) -> dict[str, Any]:
    if not isinstance(mapping, dict) or not mapping:
        return {
            "class_label_to_index": {},
            "class_index_to_label": {},
            "class_indices": [],
            "class_order": [],
        }
    class_label_to_index = {str(k): int(v) for k, v in mapping.items()}
    sorted_pairs = sorted(class_label_to_index.items(), key=lambda kv: (kv[1], str(kv[0]).lower()))
    class_indices = [int(v) for _, v in sorted_pairs]
    class_index_to_label = {str(v): str(k) for k, v in sorted_pairs}
    configured_order = exp_cfg.get("target", {}).get("class_order", [])
    if isinstance(configured_order, list) and configured_order:
        class_order = [str(label) for label in configured_order]
    else:
        class_order = [str(label) for label, _ in sorted_pairs]
    return {
        "class_label_to_index": class_label_to_index,
        "class_index_to_label": class_index_to_label,
        "class_indices": class_indices,
        "class_order": class_order,
    }


def _build_predefined_uci_feature_splits(
    dataset_cfg: dict[str, Any],
    *,
    formulation: str,
    target_mapping: dict[str, int] | None,
) -> tuple[dict[str, pd.DataFrame], str, str, str, dict[str, Any]]:
    source_cfg = _resolve_dataset_source_config(dataset_cfg)
    if source_cfg["format"] != "parquet" or source_cfg["split_mode"] != "predefined":
        raise ValueError("Predefined UCI feature split builder requires parquet predefined split config.")

    schema_cfg = dataset_cfg.get("schema", {})
    loaded = load_uct_student_predefined_splits(dataset_cfg)
    raw_train = loaded["train"]
    raw_valid = loaded.get("valid")
    raw_test = loaded["test"]

    adapted_train = adapt_uct_student_schema(raw_train, schema_cfg)
    adapted_valid = adapt_uct_student_schema(raw_valid, schema_cfg) if isinstance(raw_valid, pd.DataFrame) else None
    adapted_test = adapt_uct_student_schema(raw_test, schema_cfg)
    source_target_col = str(adapted_train["target_column"])
    if adapted_valid is not None and str(adapted_valid["target_column"]) != source_target_col:
        raise ValueError(
            "Predefined UCI parquet train/valid target column mismatch after schema adaptation. "
            f"train={adapted_train['target_column']} valid={adapted_valid['target_column']}"
        )
    if str(adapted_test["target_column"]) != source_target_col:
        raise ValueError(
            "Predefined UCI parquet train/test target column mismatch after schema adaptation. "
            f"train={adapted_train['target_column']} test={adapted_test['target_column']}"
        )
    id_column = str(adapted_train["id_column"])
    if str(adapted_test["id_column"]) != id_column:
        print(
            "[dataset][uci] id_column differs between train/test after adaptation; "
            f"train={adapted_train['id_column']} test={adapted_test['id_column']}"
        )

    train_features = _build_uct_feature_table(adapted_train, dataset_cfg.get("features", {}))
    valid_features = (
        _build_uct_feature_table(adapted_valid, dataset_cfg.get("features", {}))
        if adapted_valid is not None
        else None
    )
    test_features = _build_uct_feature_table(adapted_test, dataset_cfg.get("features", {}))
    train_prepared = _prepare_feature_df_with_target(
        train_features,
        dataset_name="uct_student",
        source_target_col=source_target_col,
        formulation=formulation,
        target_mapping=target_mapping,
    )
    valid_prepared = (
        _prepare_feature_df_with_target(
            valid_features,
            dataset_name="uct_student",
            source_target_col=source_target_col,
            formulation=formulation,
            target_mapping=target_mapping,
        )
        if isinstance(valid_features, pd.DataFrame)
        else None
    )
    test_prepared = _prepare_feature_df_with_target(
        test_features,
        dataset_name="uct_student",
        source_target_col=source_target_col,
        formulation=formulation,
        target_mapping=target_mapping,
    )
    feature_reference = train_prepared.drop(columns=["target"], errors="ignore")
    if valid_prepared is not None:
        feature_valid = valid_prepared.drop(columns=["target"], errors="ignore")
        if set(feature_reference.columns) != set(feature_valid.columns):
            missing_in_valid = [col for col in feature_reference.columns if col not in feature_valid.columns]
            extra_in_valid = [col for col in feature_valid.columns if col not in feature_reference.columns]
            raise ValueError(
                "UCI predefined parquet valid feature schema mismatch after feature building. "
                f"missing_in_valid={missing_in_valid[:10]} extra_in_valid={extra_in_valid[:10]}"
            )
        aligned_valid_features = align_feature_schema(feature_reference, feature_valid, fill_value=np.nan)
        validate_feature_schema(feature_reference, aligned_valid_features, context="uci_predefined_valid_feature_alignment")
        valid_prepared = pd.concat(
            [aligned_valid_features.reset_index(drop=True), valid_prepared[["target"]].reset_index(drop=True)],
            axis=1,
        )
    feature_test = test_prepared.drop(columns=["target"], errors="ignore")
    if set(feature_reference.columns) != set(feature_test.columns):
        missing_in_test = [col for col in feature_reference.columns if col not in feature_test.columns]
        extra_in_test = [col for col in feature_test.columns if col not in feature_reference.columns]
        raise ValueError(
            "UCI predefined parquet feature schema mismatch after feature building. "
            f"missing_in_test={missing_in_test[:10]} extra_in_test={extra_in_test[:10]}"
        )
    aligned_test_features = align_feature_schema(feature_reference, feature_test, fill_value=np.nan)
    validate_feature_schema(feature_reference, aligned_test_features, context="uci_predefined_test_feature_alignment")
    test_prepared = pd.concat(
        [aligned_test_features.reset_index(drop=True), test_prepared[["target"]].reset_index(drop=True)],
        axis=1,
    )
    return {
        "train": train_prepared.reset_index(drop=True),
        "valid": valid_prepared.reset_index(drop=True) if isinstance(valid_prepared, pd.DataFrame) else None,
        "test": test_prepared.reset_index(drop=True),
    }, "uct_student", id_column, source_target_col, {
        "source_format": source_cfg["format"],
        "split_mode": source_cfg["split_mode"],
        "schema_report": loaded.get("schema_report", {}),
        "valid_schema_report": loaded.get("valid_schema_report", {}),
        "resolved_paths": loaded.get("resolved_paths", {}),
    }


def prepare_benchmark_dataset(
    *,
    exp_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    dataset_cfg_path: str,
    formulation: str,
    seed: int,
) -> DatasetPreparationResult:
    requested_dataset_token = str(dataset_cfg_path)
    target_mapping = _resolve_target_mapping(exp_cfg, dataset_cfg, formulation)
    class_metadata = _resolve_class_metadata(exp_cfg, target_mapping)
    dataset_source_cfg = _resolve_dataset_source_config(dataset_cfg)
    dataset_name = _normalize_dataset_name(str(dataset_cfg.get("dataset", {}).get("name", "")))
    resolved_dataset_token = str(dataset_cfg_path)
    print(f"[dataset] requested={requested_dataset_token}")
    print(f"[dataset] resolved={resolved_dataset_token}")
    print(f"[dataset] config_path={dataset_cfg_path}")
    pre_split_train_feature_source_for_vocab: pd.DataFrame | None = None
    if requested_dataset_token != resolved_dataset_token:
        raise ValueError(
            "Dataset resolution mismatch. "
            f"requested={requested_dataset_token} resolved={resolved_dataset_token} config_path={dataset_cfg_path}"
        )

    if (
        dataset_name == "uct_student"
        and dataset_source_cfg["format"] == "parquet"
        and dataset_source_cfg["split_mode"] == "predefined"
    ):
        if resolved_dataset_token == "uci_student_presplit_parquet":
            raw_dataset_identity = str(dataset_cfg.get("dataset", {}).get("name", "")).strip()
            if raw_dataset_identity != "uci_student_presplit_parquet":
                raise ValueError(
                    "Dataset identity mismatch for uci_student_presplit_parquet: "
                    f"dataset.name={raw_dataset_identity!r} expected='uci_student_presplit_parquet'."
                )
            expected_train_path = "data/processed/uci/uci_12_03_train.parquet"
            expected_test_path = "data/processed/uci/uci_12_03_test.parquet"
            resolved_train_cfg_path = str(dataset_source_cfg.get("train_path", "")).replace("\\", "/")
            resolved_test_cfg_path = str(dataset_source_cfg.get("test_path", "")).replace("\\", "/")
            if resolved_train_cfg_path != expected_train_path or resolved_test_cfg_path != expected_test_path:
                raise ValueError(
                    "Dataset lock violation for uci_student_presplit_parquet: "
                    f"expected train/test paths ({expected_train_path}, {expected_test_path}) but got "
                    f"({resolved_train_cfg_path}, {resolved_test_cfg_path})."
                )
        predefined_splits, dataset_name, id_column, source_target_col, dataset_source_meta = _build_predefined_uci_feature_splits(
            dataset_cfg,
            formulation=formulation,
            target_mapping=target_mapping,
        )
        pre_split_train_feature_source_for_vocab = predefined_splits["train"].reset_index(drop=True).copy()
        train_feature_df, train_missing_meta = _drop_rows_with_missing_values(
            predefined_splits["train"],
            exp_cfg.get("preprocessing", {}),
        )
        test_feature_df, test_missing_meta = _drop_rows_with_missing_values(
            predefined_splits["test"],
            exp_cfg.get("preprocessing", {}),
        )
        if train_feature_df.empty:
            raise ValueError("UCI predefined parquet train split became empty after missing-value filtering.")
        if test_feature_df.empty:
            raise ValueError("UCI predefined parquet test split became empty after missing-value filtering.")
        valid_feature_df_raw = predefined_splits.get("valid")
        internal_valid_source = "predefined_valid" if isinstance(valid_feature_df_raw, pd.DataFrame) else "train_only"
        effective_split_mode = (
            "predefined_train_valid_test"
            if isinstance(valid_feature_df_raw, pd.DataFrame)
            else "train_test_with_internal_valid"
        )
        if isinstance(valid_feature_df_raw, pd.DataFrame):
            valid_feature_df, valid_missing_meta = _drop_rows_with_missing_values(
                valid_feature_df_raw,
                exp_cfg.get("preprocessing", {}),
            )
            if valid_feature_df.empty:
                raise ValueError("UCI predefined parquet valid split became empty after missing-value filtering.")
            splits = {
                "train": train_feature_df.reset_index(drop=True),
                "valid": valid_feature_df.reset_index(drop=True),
                "test": test_feature_df.reset_index(drop=True),
            }
        else:
            validation_size = float(exp_cfg["splits"].get("validation_size", 0.2))
            train_valid_split_cfg = SplitConfig(
                test_size=validation_size,
                validation_size=0.0,
                random_state=seed,
                stratify_column="target",
            )
            train_valid_splits = stratified_train_valid_test_split(train_feature_df, train_valid_split_cfg)
            valid_feature_df = train_valid_splits["test"].reset_index(drop=True)
            train_feature_df = train_valid_splits["train"].reset_index(drop=True)
            valid_missing_meta = {
                "enabled": True,
                "mode": "internal_from_train_only",
                "source_rows_before_split": int(len(predefined_splits["train"])),
                "validation_size": float(validation_size),
                "random_state": int(seed),
                "stratify_column": "target",
            }
            if train_feature_df.empty or valid_feature_df.empty:
                raise ValueError("Internal validation split from predefined train parquet produced an empty train or valid split.")
            splits = {
                "train": train_feature_df,
                "valid": valid_feature_df,
                "test": test_feature_df.reset_index(drop=True),
            }
        missing_value_meta = {
            "enabled": True,
            "mode": "predefined_uci_parquet",
            "train": train_missing_meta,
            "valid": valid_missing_meta,
            "test": test_missing_meta,
            "source": dataset_source_meta,
            "effective_split_mode": effective_split_mode,
            "internal_valid_source": internal_valid_source,
        }
        print(f"[dataset] train_path={dataset_source_cfg.get('train_path')}")
        print(f"[dataset] test_path={dataset_source_cfg.get('test_path')}")
        print(f"[dataset] valid_path={dataset_source_cfg.get('valid_path')}")
        print(f"[dataset] split_mode={effective_split_mode}")
        print(f"[dataset] internal_valid_source={internal_valid_source}")
        print(
            "[dataset][uci] "
            f"source_format={dataset_source_meta.get('source_format')} "
            f"split_mode={dataset_source_meta.get('split_mode')} "
            f"loaded_train_rows={int(len(predefined_splits['train']))} "
            f"loaded_valid_rows={int(len(predefined_splits['valid'])) if isinstance(predefined_splits.get('valid'), pd.DataFrame) else 0} "
            f"loaded_test_rows={int(len(predefined_splits['test']))} "
            f"target_column={source_target_col}"
        )
        print("[dataset][uci] schema_validation_passed=true")
    else:
        feature_df, dataset_name, id_column, source_target_col = _build_feature_table(dataset_cfg)
        if not dataset_name:
            raise ValueError("Resolved dataset name is empty after feature table construction.")
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Resolved dataset name '{dataset_name}' is unsupported. "
                f"Supported datasets: {sorted(SUPPORTED_DATASETS)}."
            )
        feature_df = feature_df.copy()
        feature_df, missing_value_meta = _drop_rows_with_missing_values(feature_df, exp_cfg.get("preprocessing", {}))
        feature_df = _prepare_feature_df_with_target(
            feature_df,
            dataset_name=dataset_name,
            source_target_col=source_target_col,
            formulation=formulation,
            target_mapping=target_mapping,
        )
        split_cfg = SplitConfig(
            test_size=float(exp_cfg["splits"]["test_size"]),
            validation_size=float(exp_cfg["splits"].get("validation_size", 0.2)),
            random_state=seed,
            stratify_column="target",
        )
        splits = stratified_train_valid_test_split(feature_df, split_cfg)

    return DatasetPreparationResult(
        splits=splits,
        dataset_name=dataset_name,
        id_column=id_column,
        source_target_col=source_target_col,
        target_mapping=target_mapping,
        class_metadata=class_metadata,
        dataset_source_cfg=dataset_source_cfg,
        requested_dataset_token=requested_dataset_token,
        resolved_dataset_token=resolved_dataset_token,
        missing_value_meta=missing_value_meta,
        pre_split_train_feature_source_for_vocab=pre_split_train_feature_source_for_vocab,
    )
