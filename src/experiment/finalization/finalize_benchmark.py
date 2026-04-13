from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiment.finalization.shared_contracts import (
    artifact_status_from_path,
    ensure_explainability_contract,
    update_benchmark_artifact_manifest,
)
from src.experiment.finalization.shared_types import (
    BenchmarkExecutionResult,
    BenchmarkFinalizationContext,
)
from src.reporting.benchmark_summary import save_benchmark_summary
from src.reporting.generate_all_figures import generate_all_figures
from src.reporting.runtime_persistence import (
    _mirror_root_artifacts_to_runtime,
    _persist_per_model_run_outputs,
    _persist_required_contract_outputs,
    _persist_runtime_artifacts,
    _write_benchmark_failure_summary,
)
from src.reporting.standard_artifacts import write_skipped_explainability_report


def _build_stage2_report_paths(
    *,
    output_dir: Path,
    two_stage_enabled: bool,
    two_stage_feature_bundle: dict[str, Any],
    model_candidates: list[str],
    stage2_feature_counts_by_model: dict[str, int],
    stage2_advanced_reports_by_model: dict[str, dict[str, Any]],
) -> tuple[Path | None, Path | None]:
    stage2_feature_sharpening_report_path: Path | None = None
    stage2_advanced_feature_report_path: Path | None = None
    if not two_stage_enabled:
        return stage2_feature_sharpening_report_path, stage2_advanced_feature_report_path

    stage2_feature_report_payload = (
        dict(two_stage_feature_bundle.get("report", {}))
        if isinstance(two_stage_feature_bundle.get("report", {}), dict)
        else {"enabled": False}
    )
    stage2_feature_report_payload["enabled"] = bool(two_stage_feature_bundle.get("enabled", False))
    stage2_feature_report_payload["requested_groups"] = list(two_stage_feature_bundle.get("requested_groups", []))
    stage2_feature_report_payload["feature_separation_groups"] = list(
        two_stage_feature_bundle.get("feature_separation_groups", [])
    )
    stage2_feature_report_payload["interaction_requested_groups"] = list(
        two_stage_feature_bundle.get("advanced_requested_groups", [])
    )
    stage2_feature_report_payload["stage1_candidate_models"] = list(model_candidates)
    stage2_feature_report_payload["stage2_candidate_models"] = list(model_candidates)
    stage2_feature_report_payload["per_model_stage2_feature_count"] = stage2_feature_counts_by_model
    stage2_feature_sharpening_report_path = output_dir / "stage2_feature_sharpening_report.json"
    stage2_feature_sharpening_report_path.write_text(
        json.dumps(stage2_feature_report_payload, indent=2),
        encoding="utf-8",
    )

    prototype_source_columns: list[str] = []
    prototype_feature_columns: list[str] = []
    for model_name in model_candidates:
        model_report = stage2_advanced_reports_by_model.get(model_name, {})
        prototype_report = (
            model_report.get("prototype_distance", {})
            if isinstance(model_report.get("prototype_distance", {}), dict)
            else {}
        )
        for col in prototype_report.get("prototype_source_columns", []):
            if col not in prototype_source_columns:
                prototype_source_columns.append(col)
        for col in prototype_report.get("prototype_feature_columns", []):
            if col not in prototype_feature_columns:
                prototype_feature_columns.append(col)
    stage2_advanced_feature_report_payload = {
        "feature_sharpening_enabled": bool(
            stage2_feature_report_payload.get("feature_sharpening", {}).get("enabled", False)
        ),
        "feature_separation_enabled": bool(stage2_feature_report_payload.get("feature_separation_enabled", False)),
        "interaction_features_enabled": bool(stage2_feature_report_payload.get("interaction_features_enabled", False)),
        "prototype_distance_enabled": bool(stage2_feature_report_payload.get("prototype_distance_enabled", False)),
        "requested_groups": list(stage2_feature_report_payload.get("requested_groups", [])),
        "feature_separation_groups": list(stage2_feature_report_payload.get("feature_separation_groups", [])),
        "interaction_requested_groups": list(stage2_feature_report_payload.get("interaction_requested_groups", [])),
        "created_feature_names": list(stage2_feature_report_payload.get("created_features", [])),
        "skipped_feature_reasons": stage2_feature_report_payload.get("skipped_features_by_group", {}),
        "missing_base_columns": list(stage2_feature_report_payload.get("missing_base_columns", [])),
        "prototype_source_columns": prototype_source_columns,
        "prototype_feature_columns": prototype_feature_columns,
        "per_model_stage2_feature_count": stage2_feature_counts_by_model,
    }
    stage2_advanced_feature_report_path = output_dir / "stage2_advanced_feature_report.json"
    stage2_advanced_feature_report_path.write_text(
        json.dumps(stage2_advanced_feature_report_payload, indent=2),
        encoding="utf-8",
    )
    return stage2_feature_sharpening_report_path, stage2_advanced_feature_report_path


def _build_model_mechanism_audit(
    *,
    model_results: dict[str, Any],
    model_decision_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    model_mechanism_audit: dict[str, Any] = {}
    for model_name, payload in model_results.items():
        if not isinstance(payload, dict) or "metrics" not in payload:
            continue
        class_weight_meta = payload.get("class_weight", {}) if isinstance(payload.get("class_weight", {}), dict) else {}
        threshold_meta = payload.get("threshold_tuning", {}) if isinstance(payload.get("threshold_tuning", {}), dict) else {}
        decision_meta = payload.get("decision_policy_tuning", {}) if isinstance(payload.get("decision_policy_tuning", {}), dict) else {}
        model_decision_cfg = (
            payload.get("decision_rule_config", {})
            if isinstance(payload.get("decision_rule_config", {}), dict)
            else model_decision_configs.get(model_name, {})
        )
        model_mechanism_audit[model_name] = {
            "decision_rule": str(model_decision_cfg.get("decision_rule", "model_predict")),
            "decision_strategy": str(
                model_decision_cfg.get("multiclass_decision", {}).get(
                    "strategy",
                    model_decision_cfg.get("decision_rule", "model_predict"),
                )
            ),
            "multiclass_decision_auto_tuned": bool(decision_meta.get("multiclass_decision_auto_tuned", False)),
            "multiclass_decision_tuning_objective": decision_meta.get("tuning_objective"),
            "multiclass_decision_selected_parameters": decision_meta.get("selected_parameters", {}),
            "validation_objective_score_at_selected_threshold": decision_meta.get(
                "validation_objective_score_at_selected_threshold"
            ),
            "class_weight_requested": bool(class_weight_meta.get("class_weight_requested", False)),
            "class_weight_supported": bool(class_weight_meta.get("class_weight_supported", False)),
            "class_weight_applied": bool(class_weight_meta.get("class_weight_applied", False)),
            "class_weight_effective": str(class_weight_meta.get("effective_mechanism", "none")),
            "class_weight_values": class_weight_meta.get("weight_map", {}),
            "class_weight_backend_note": class_weight_meta.get("class_weight_backend_note"),
            "threshold_tuning_requested": bool(threshold_meta.get("threshold_tuning_requested", False)),
            "threshold_tuning_supported": bool(threshold_meta.get("threshold_tuning_supported", False)),
            "threshold_tuning_applied": bool(threshold_meta.get("threshold_tuning_applied", False)),
            "threshold_selection_split": str(threshold_meta.get("threshold_selection_split", "validation")),
            "threshold_applied_to": str(threshold_meta.get("threshold_applied_to", "none")),
            "selected_thresholds": threshold_meta.get("selected_thresholds", {}),
            "stage2_decision_objective_mode": payload.get("metrics", {}).get("stage2_decision_objective_mode"),
            "stage2_decision_requested": payload.get("metrics", {}).get("stage2_decision_requested"),
            "stage2_decision_executed": payload.get("metrics", {}).get("stage2_decision_executed"),
            "stage2_decision_accepted": payload.get("metrics", {}).get("stage2_decision_accepted"),
            "stage2_decision_reject_reason": payload.get("metrics", {}).get("stage2_decision_reject_reason"),
        }
    return model_mechanism_audit


def finalize_benchmark_run(
    *,
    context: BenchmarkFinalizationContext,
    execution: BenchmarkExecutionResult,
) -> dict[str, Any]:
    stage2_feature_sharpening_report_path, stage2_advanced_feature_report_path = _build_stage2_report_paths(
        output_dir=context.output_dir,
        two_stage_enabled=context.two_stage_enabled,
        two_stage_feature_bundle=context.two_stage_feature_bundle,
        model_candidates=context.model_candidates,
        stage2_feature_counts_by_model=execution.stage2_feature_counts_by_model,
        stage2_advanced_reports_by_model=execution.stage2_advanced_reports_by_model,
    )
    compact_mode = (
        bool(context.output_cfg.get("compact_summary", False))
        if context.compact_summary is None
        else bool(context.compact_summary)
    )
    summary = {
        "experiment_id": context.experiment_id,
        "benchmark_summary_version": context.benchmark_summary_version,
        "schema_version": context.benchmark_summary_version,
        "dataset_name": context.dataset_name,
        "dataset_config": {
            "requested": context.requested_dataset_token,
            "resolved": context.resolved_dataset_token,
            "path": str(context.dataset_cfg_path),
            "source_format": context.dataset_source_cfg.get("format"),
            "split_mode": context.dataset_source_cfg.get("split_mode"),
            "effective_split_mode": context.missing_value_meta.get("effective_split_mode"),
            "internal_valid_source": context.missing_value_meta.get("internal_valid_source"),
            "train_path": context.dataset_source_cfg.get("train_path"),
            "valid_path": context.dataset_source_cfg.get("valid_path"),
            "test_path": context.dataset_source_cfg.get("test_path"),
        },
        "target_formulation": context.formulation,
        "class_metadata": context.class_metadata,
        "primary_metric": context.metric_key,
        "best_model": execution.best_model,
        "output_dir": str(context.output_dir),
        "seed": context.seed,
        "split_config": {
            "test_size": float(context.exp_cfg["splits"]["test_size"]),
            "validation_size": float(context.exp_cfg["splits"].get("validation_size", 0.2)),
            "stratify_column": str(context.exp_cfg["splits"].get("stratify_column", "target")),
        },
        "split_sizes": {
            "train_raw": int(len(context.splits.get("train", pd.DataFrame()))),
            "valid_raw": int(len(context.splits.get("valid", pd.DataFrame()))),
            "test_raw": int(len(context.splits.get("test", pd.DataFrame()))),
            "train_after_preprocessing": int(len(context.artifacts.X_train)),
            "valid_after_preprocessing": int(len(context.artifacts.X_valid)),
            "test_after_preprocessing": int(len(context.artifacts.X_test)),
            "train_after_outlier": int(len(context.X_train_filtered)),
            "train_after_balancing": int(len(context.X_train_bal)),
        },
        "model_results": execution.model_results,
        "leaderboard": execution.leaderboard_df.to_dict(orient="records"),
        "preprocessing": {
            "missing_values": context.missing_value_meta,
            "outlier": context.outlier_meta,
            "balancing": context.balancing_meta,
            "class_distribution_train_before_outlier": context.pre_outlier_class_distribution,
            "class_distribution_train_after_outlier": context.post_outlier_class_distribution,
            "feature_count_after_preprocessing": int(context.artifacts.X_train.shape[1]),
            "preprocessed_feature_count": int(
                context.onehot_metadata.get("preprocessed_feature_count", int(context.artifacts.X_train.shape[1]))
            ),
            "onehot_enabled": bool(context.onehot_metadata.get("onehot_enabled", False)),
            "require_onehot_encoding": bool(context.onehot_metadata.get("require_onehot_encoding", False)),
            "input_numeric_feature_count": int(context.onehot_metadata.get("input_numeric_feature_count", 0)),
            "input_categorical_feature_count": int(context.onehot_metadata.get("input_categorical_feature_count", 0)),
            "encoded_categorical_feature_count": int(
                context.onehot_metadata.get("encoded_categorical_feature_count", 0)
            ),
            "stable_locked_vocabulary_mode": bool(
                context.onehot_metadata.get("stable_locked_vocabulary_mode", False)
            ),
            "vocabulary_source": context.onehot_metadata.get("vocabulary_source"),
            "per_column_category_counts": context.onehot_metadata.get("per_column_category_counts", {}),
        },
        "class_weight": context.class_weight_cfg if isinstance(context.class_weight_cfg, dict) else {},
        "threshold_tuning": context.threshold_tuning_cfg,
        "cross_validation": context.cv_reporting_cfg,
        "decision_policy": context.decision_rule_cfg,
        "summary_mode": "compact" if compact_mode else "full",
        "successful_models": list(execution.successful_models),
        "failed_models": dict(execution.failed_models),
        "successful_candidate_count": int(len(execution.successful_models)),
        "failed_candidate_count": int(len(execution.failed_models)),
        "paper_style_result_note": "Cross-validation metrics (cv_*) are the paper-style reproduction result; holdout metrics (test_*) are separate benchmark outputs.",
    }
    if context.two_stage_enabled:
        summary["two_stage_feature_sharpening"] = (
            dict(context.two_stage_feature_bundle.get("report", {}))
            if isinstance(context.two_stage_feature_bundle.get("report", {}), dict)
            else {"enabled": False}
        )
    if execution.best_by_cv.get("model") is not None:
        summary["best_by_cv"] = execution.best_by_cv
    if execution.best_by_test.get("model") is not None:
        summary["best_by_test"] = execution.best_by_test
    if context.model_selection_cfg.get("enabled", False):
        summary["selection"] = context.model_selection_cfg
    if bool(context.global_balance_guard_cfg.get("enabled", False)):
        summary["global_balance_guard"] = {
            **context.global_balance_guard_cfg,
            **execution.global_balance_guard_report,
        }

    summary["model_mechanism_audit"] = _build_model_mechanism_audit(
        model_results=execution.model_results,
        model_decision_configs=execution.model_decision_configs,
    )
    summary["artifact_paths"] = _persist_runtime_artifacts(
        output_dir=context.output_dir,
        best_model_name=execution.best_model,
        trained_models=execution.trained_models,
        preprocessing_artifacts=context.artifacts,
        summary=summary,
        runtime_artifact_overrides_by_model=execution.runtime_artifact_overrides_by_model,
        file_format=context.runtime_artifact_format,
    )

    contract_paths = _persist_required_contract_outputs(
        output_dir=context.output_dir,
        summary=summary,
        best_model_name=execution.best_model,
        trained_models=execution.trained_models,
        y_test=context.artifacts.y_test,
        class_metadata=context.class_metadata,
    )
    summary["artifact_paths"].update(contract_paths)
    summary["artifact_paths"].update(
        context.persist_paper_style_cv_artifacts_fn(
            output_dir=context.output_dir,
            model_results=execution.model_results,
        )
    )
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary["artifact_paths"].update(
        _persist_per_model_run_outputs(
            output_dir=context.output_dir,
            run_stamp=run_stamp,
            model_results=execution.model_results,
            trained_models=execution.trained_models,
            class_metadata=context.class_metadata,
        )
    )

    figure_status: dict[str, dict[str, Any]] = {}
    summary["artifact_status"] = (
        dict(summary.get("artifact_status", {}))
        if isinstance(summary.get("artifact_status", {}), dict)
        else {}
    )
    print(
        "[finalization][start] "
        f"successful_candidates={len(execution.successful_models)} "
        f"failed_candidates={len(execution.failed_models)} "
        f"best_model={execution.best_model}"
    )
    summary["artifact_paths"]["benchmark_summary"] = str(context.output_dir / "benchmark_summary.json")
    summary["artifact_paths"]["leaderboard"] = str(context.output_dir / "leaderboard.csv")
    summary["artifact_paths"]["summary_csv"] = str(context.output_dir / "summary.csv")
    summary["artifact_paths"]["benchmark_markdown"] = str(context.output_dir / "benchmark_summary.md")
    summary["artifact_paths"]["artifact_manifest"] = str(context.output_dir / "artifact_manifest.json")
    if stage2_feature_sharpening_report_path is not None:
        summary["artifact_paths"]["stage2_feature_sharpening_report"] = str(stage2_feature_sharpening_report_path)
    if stage2_advanced_feature_report_path is not None:
        summary["artifact_paths"]["stage2_advanced_feature_report"] = str(stage2_advanced_feature_report_path)
    for model_name, paths in execution.optuna_artifacts.items():
        model_token = context.safe_filename_token_fn(model_name)
        summary["artifact_paths"][f"optuna_trials_{model_token}"] = paths["trials_csv"]
        summary["artifact_paths"][f"optuna_best_params_{model_token}"] = paths["best_params_json"]
    if execution.failed_models:
        summary["artifact_paths"].update(
            _write_benchmark_failure_summary(
                output_dir=context.output_dir,
                experiment_id=context.experiment_id,
                requested_models=context.model_candidates,
                model_results=execution.model_results,
                failed_models=execution.failed_models,
                successful_models=execution.successful_models,
                reason="partial_candidate_failures",
            )
        )
    print("[finalization][summary_write_begin]")
    try:
        save_benchmark_summary(summary, context.output_dir, compact=compact_mode)
        summary["artifact_paths"]["benchmark_summary"] = str(context.output_dir / "benchmark_summary.json")
        summary["artifact_status"]["benchmark_summary"] = artifact_status_from_path(
            context.output_dir / "benchmark_summary.json"
        )
        summary["artifact_status"]["leaderboard"] = artifact_status_from_path(context.output_dir / "leaderboard.csv")
        summary["artifact_status"]["summary_csv"] = artifact_status_from_path(context.output_dir / "summary.csv")
        print("[finalization][summary_written]")
    except Exception as exc:
        if execution.failed_models:
            _write_benchmark_failure_summary(
                output_dir=context.output_dir,
                experiment_id=context.experiment_id,
                requested_models=context.model_candidates,
                model_results=execution.model_results,
                failed_models=execution.failed_models,
                successful_models=execution.successful_models,
                reason=f"finalization_summary_write_failed:{type(exc).__name__}",
            )
        raise

    mirror_enabled = bool(context.output_cfg.get("mirror_benchmark_outputs_to_runtime", False))
    if mirror_enabled:
        try:
            mirrored = _mirror_root_artifacts_to_runtime(
                output_dir=context.output_dir,
                runtime_dir=context.output_dir / "runtime_artifacts",
                model_candidates=context.model_candidates,
            )
            for name, path in mirrored.items():
                summary["artifact_paths"][f"runtime_{name}"] = path
        except Exception as exc:
            print(f"[finalization][warning] runtime mirroring failed: {type(exc).__name__}: {exc}")
            summary.setdefault("finalization_warnings", []).append(
                f"runtime_mirroring_failed:{type(exc).__name__}: {exc}"
            )

    if execution.best_model and execution.best_model in execution.trained_models and execution.trained_models[execution.best_model] is not None:
        best_payload = execution.model_results.get(execution.best_model, {})
        print("[finalization][figures_begin]")
        try:
            figure_result = generate_all_figures(
                model=execution.trained_models[execution.best_model],
                X_train=context.X_train_bal,
                y_train=context.y_train_bal,
                X_test=context.artifacts.X_test,
                y_test=context.artifacts.y_test,
                y_pred_proba=best_payload.get("artifacts", {}).get("y_proba_test"),
                output_dir=context.output_dir,
                experiment_name=context.experiment_id,
                primary_metric=context.primary_metric,
                random_state=context.seed,
                include_status=True,
            )
            figure_paths = figure_result.get("artifact_paths", {})
            figure_status = figure_result.get("artifact_status", {})
            summary["artifact_paths"].update(figure_paths)
            summary["artifact_status"].update(figure_status)
            print("[finalization][figures_done]")
        except Exception as exc:
            print(f"[finalization][figures_failed] {type(exc).__name__}: {exc}")
            figure_status = {
                "learning_curve": {"status": "failed", "reason": f"figure_generation_failed:{type(exc).__name__}: {exc}"},
                "pr_curve": {"status": "failed", "reason": f"figure_generation_failed:{type(exc).__name__}: {exc}"},
                "shap_beeswarm": {"status": "failed", "reason": f"figure_generation_failed:{type(exc).__name__}: {exc}"},
                "shap_waterfall": {"status": "failed", "reason": f"figure_generation_failed:{type(exc).__name__}: {exc}"},
            }
            summary["artifact_status"].update(figure_status)
    else:
        figure_status["learning_curve"] = {"status": "skipped", "reason": "best_model_missing_or_unavailable"}
        figure_status["pr_curve"] = {"status": "skipped", "reason": "best_model_missing_or_unavailable"}
        figure_status["shap_beeswarm"] = {"status": "skipped", "reason": "best_model_missing_or_unavailable"}
        figure_status["shap_waterfall"] = {"status": "skipped", "reason": "best_model_missing_or_unavailable"}
        summary["artifact_status"].update(figure_status)
    try:
        ensure_explainability_contract(summary)
    except Exception as exc:
        print(f"[finalization][warning] explainability contract compatibility check failed: {type(exc).__name__}: {exc}")
        summary.setdefault("finalization_warnings", []).append(
            f"explainability_contract_check_failed:{type(exc).__name__}: {exc}"
        )

    try:
        explain_json, explain_md = write_skipped_explainability_report(
            output_dir=context.output_dir,
            reason="explainability_not_run_yet",
        )
    except Exception as exc:
        print(f"[finalization][warning] skipped explainability report failed: {type(exc).__name__}: {exc}")
        explain_json = context.output_dir / "explainability" / "explainability_status.json"
        explain_md = context.output_dir / "explainability" / "README.md"
        summary.setdefault("finalization_warnings", []).append(
            f"skipped_explainability_report_failed:{type(exc).__name__}: {exc}"
        )

    print("[finalization][summary_refresh_begin]")
    try:
        save_benchmark_summary(summary, context.output_dir, compact=compact_mode)
        summary["artifact_paths"]["benchmark_summary"] = str(context.output_dir / "benchmark_summary.json")
        summary["artifact_status"]["benchmark_summary"] = artifact_status_from_path(
            context.output_dir / "benchmark_summary.json"
        )
        summary["artifact_status"]["leaderboard"] = artifact_status_from_path(context.output_dir / "leaderboard.csv")
        summary["artifact_status"]["summary_csv"] = artifact_status_from_path(context.output_dir / "summary.csv")
        print("[finalization][summary_refresh_done]")
    except Exception as exc:
        print(f"[finalization][warning] summary refresh failed: {type(exc).__name__}: {exc}")
        summary.setdefault("finalization_warnings", []).append(
            f"summary_refresh_failed:{type(exc).__name__}: {exc}"
        )

    confusion_matrix_paths = sorted(context.output_dir.glob("confusion_matrix_*.png"))
    normalized_confusion_matrix_paths = sorted(context.output_dir.glob("confusion_matrix_*_normalized.png"))
    classification_report_paths = sorted(context.output_dir.glob("classification_report_*.json"))
    mandatory_updates: dict[str, dict[str, Any]] = {
        "benchmark_summary_json": artifact_status_from_path(summary["artifact_paths"]["benchmark_summary"]),
        "benchmark_summary_md": artifact_status_from_path(summary["artifact_paths"]["benchmark_markdown"]),
        "metrics_json": artifact_status_from_path(summary["artifact_paths"]["metrics"]),
        "predictions_csv": artifact_status_from_path(summary["artifact_paths"]["predictions"]),
        "leaderboard_csv": artifact_status_from_path(summary["artifact_paths"]["leaderboard"]),
        "summary_csv": artifact_status_from_path(summary["artifact_paths"]["summary_csv"]),
        "runtime_artifacts_dir": artifact_status_from_path(context.output_dir / "runtime_artifacts"),
        "model_dir": artifact_status_from_path(context.output_dir / "model"),
        "learning_curve_png": figure_status.get(
            "learning_curve",
            {"status": "failed", "reason": "figure_generation_error"},
        ),
        "pr_curve_png": figure_status.get(
            "pr_curve",
            {"status": "failed", "reason": "figure_generation_error"},
        ),
    }
    mandatory_updates["confusion_matrix_artifacts"] = (
        {"status": "generated", "paths": [str(p) for p in confusion_matrix_paths]}
        if confusion_matrix_paths
        else {"status": "skipped", "reason": "not_generated_by_reporting"}
    )
    mandatory_updates["normalized_confusion_matrix_artifacts"] = (
        {"status": "generated", "paths": [str(p) for p in normalized_confusion_matrix_paths]}
        if normalized_confusion_matrix_paths
        else {"status": "skipped", "reason": "not_generated_by_reporting"}
    )
    mandatory_updates["classification_report_artifacts"] = (
        {"status": "generated", "paths": [str(p) for p in classification_report_paths]}
        if classification_report_paths
        else {"status": "skipped", "reason": "not_generated_by_reporting"}
    )
    if execution.optuna_artifacts:
        mandatory_updates["optuna_artifacts"] = {
            "status": "generated",
            "details": execution.optuna_artifacts,
        }
    elif context.tuning_enabled:
        mandatory_updates["optuna_artifacts"] = {
            "status": "failed",
            "reason": "tuning_enabled_but_no_optuna_artifacts_saved",
        }

    optional_updates: dict[str, dict[str, Any]] = {
        "shap_beeswarm_png": figure_status.get(
            "shap_beeswarm",
            {"status": "skipped", "reason": "not_applicable_for_model_type"},
        ),
        "shap_waterfall_pngs": figure_status.get(
            "shap_waterfall",
            {"status": "skipped", "reason": "not_applicable_for_model_type"},
        ),
        "explainability_dir": {"status": "created", "path": str(context.output_dir / "explainability")},
        "explainability_report_json": {
            "status": "created",
            "path": str(explain_json),
            "reason": "explainability_not_run_yet",
        },
        "explainability_report_md": {
            "status": "created",
            "path": str(explain_md),
            "reason": "explainability_not_run_yet",
        },
        "lime_outputs": {"status": "skipped", "reason": "explainability_not_run_yet"},
        "aime_outputs": {"status": "skipped", "reason": "explainability_not_run_yet"},
        "shap_outputs": {"status": "skipped", "reason": "explainability_not_run_yet"},
        "threshold_tuning_inline": {
            "status": "enabled" if bool(context.threshold_tuning_cfg.get("enabled", False)) else "disabled",
            "objective": str(context.threshold_tuning_cfg.get("objective", "macro_f1")),
            "focus_class": str(context.threshold_tuning_cfg.get("focus_class", "Enrolled")),
        },
    }
    if stage2_feature_sharpening_report_path is not None:
        optional_updates["stage2_feature_sharpening_report"] = artifact_status_from_path(
            stage2_feature_sharpening_report_path
        )
    if stage2_advanced_feature_report_path is not None:
        optional_updates["stage2_advanced_feature_report"] = artifact_status_from_path(
            stage2_advanced_feature_report_path
        )
    for artifact_key in ("cv_model_results_csv", "paper_style_cv_summary_csv", "cv_fold_summary_json"):
        artifact_path = summary["artifact_paths"].get(artifact_key)
        if artifact_path:
            optional_updates[artifact_key] = artifact_status_from_path(artifact_path)

    print("[finalization][manifest_begin]")
    try:
        update_benchmark_artifact_manifest(
            output_dir=context.output_dir,
            mandatory_updates=mandatory_updates,
            optional_updates=optional_updates,
            metadata_updates={
                "experiment_id": context.experiment_id,
                "dataset_name": context.dataset_name,
                "target_formulation": context.formulation,
                "best_model": execution.best_model,
                "manifest_scope": "benchmark",
            },
        )
        print("[finalization][manifest_done]")
    except Exception as exc:
        print(f"[finalization][manifest_failed] {type(exc).__name__}: {exc}")
        summary.setdefault("finalization_warnings", []).append(
            f"artifact_manifest_update_failed:{type(exc).__name__}: {exc}"
        )
    if mirror_enabled:
        try:
            _mirror_root_artifacts_to_runtime(
                output_dir=context.output_dir,
                runtime_dir=context.output_dir / "runtime_artifacts",
                model_candidates=context.model_candidates,
            )
        except Exception as exc:
            print(f"[finalization][warning] post-manifest runtime mirroring failed: {type(exc).__name__}: {exc}")
            summary.setdefault("finalization_warnings", []).append(
                f"post_manifest_runtime_mirroring_failed:{type(exc).__name__}: {exc}"
            )
    if summary.get("finalization_warnings"):
        try:
            save_benchmark_summary(summary, context.output_dir, compact=compact_mode)
        except Exception:
            pass
    print("[finalization][complete]")
    return summary
