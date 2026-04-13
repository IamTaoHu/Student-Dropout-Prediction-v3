from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BenchmarkExecutionResult:
    model_results: dict[str, Any]
    leaderboard_df: pd.DataFrame
    trained_models: dict[str, Any]
    optuna_artifacts: dict[str, dict[str, str]]
    model_decision_configs: dict[str, dict[str, Any]]
    runtime_artifact_overrides_by_model: dict[str, dict[str, Any]]
    stage2_feature_counts_by_model: dict[str, int]
    stage2_advanced_reports_by_model: dict[str, dict[str, Any]]
    successful_models: list[str]
    failed_models: dict[str, str]
    best_model: str | None
    best_by_cv: dict[str, Any]
    best_by_test: dict[str, Any]
    global_balance_guard_report: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkFinalizationContext:
    compact_summary: bool | None
    output_cfg: dict[str, Any]
    output_dir: Path
    experiment_id: str
    dataset_name: str
    dataset_cfg_path: Path
    requested_dataset_token: str
    resolved_dataset_token: str
    dataset_source_cfg: dict[str, Any]
    formulation: str
    class_metadata: dict[str, Any]
    metric_key: str
    seed: int
    exp_cfg: dict[str, Any]
    splits: dict[str, pd.DataFrame]
    artifacts: Any
    X_train_filtered: pd.DataFrame
    X_train_bal: pd.DataFrame
    y_train_bal: pd.Series
    missing_value_meta: dict[str, Any]
    outlier_meta: dict[str, Any]
    balancing_meta: dict[str, Any]
    pre_outlier_class_distribution: dict[str, Any]
    post_outlier_class_distribution: dict[str, Any]
    onehot_metadata: dict[str, Any]
    class_weight_cfg: dict[str, Any] | None
    threshold_tuning_cfg: dict[str, Any]
    cv_reporting_cfg: dict[str, Any]
    decision_rule_cfg: dict[str, Any]
    two_stage_enabled: bool
    two_stage_feature_bundle: dict[str, Any]
    model_candidates: list[str]
    model_selection_cfg: dict[str, Any]
    global_balance_guard_cfg: dict[str, Any]
    runtime_artifact_format: str
    tuning_enabled: bool
    primary_metric: str
    benchmark_summary_version: str
    persist_paper_style_cv_artifacts_fn: Any
    safe_filename_token_fn: Any
