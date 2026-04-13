from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiment.finalization.shared_types import (
    BenchmarkExecutionResult,
    BenchmarkFinalizationContext,
)


@dataclass(frozen=True)
class DatasetPreparationResult:
    splits: dict[str, pd.DataFrame]
    dataset_name: str
    id_column: str
    source_target_col: str
    target_mapping: dict[str, int] | None
    class_metadata: dict[str, Any]
    dataset_source_cfg: dict[str, Any]
    requested_dataset_token: str
    resolved_dataset_token: str
    missing_value_meta: dict[str, Any]
    pre_split_train_feature_source_for_vocab: pd.DataFrame | None


@dataclass(frozen=True)
class PreprocessingExecutionResult:
    preprocess_cfg: dict[str, Any]
    artifacts: Any
    onehot_metadata: dict[str, Any]
    outlier_cfg: dict[str, Any]
    outlier_meta: dict[str, Any]
    X_train_filtered: pd.DataFrame
    y_train_filtered: pd.Series
    balancing_cfg: dict[str, Any]
    balancing_meta: dict[str, Any]
    X_train_bal: pd.DataFrame
    y_train_bal: pd.Series
    pre_outlier_class_distribution: dict[str, int]
    post_outlier_class_distribution: dict[str, int]


@dataclass(frozen=True)
class BenchmarkRunnerResult:
    execution: BenchmarkExecutionResult
    finalization_context: BenchmarkFinalizationContext
