from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.experiment.config_resolution import _resolve_model_decision_rule_config
from src.reporting.prediction_exports import _add_named_per_class_metrics


def run_two_stage_mode(
    *,
    experiment_mode: str,
    exp_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    model_candidates: list[str],
    formulation: str,
    class_metadata: dict[str, Any],
    decision_rule_cfg: dict[str, Any],
    tuning_cfg: dict[str, Any],
    tuning_backend: str,
    default_n_trials: int,
    per_model_trial_budgets: dict[str, int],
    retrain_on_full_train_split: bool,
    class_weight_cfg: dict[str, Any] | None,
    two_stage_decision_mode: str,
    resolved_two_stage_branch: str,
    resolved_stage2_optuna_cfg: dict[str, Any],
    resolved_stage2_decision_cfg: dict[str, Any],
    two_stage_cfg: dict[str, Any],
    threshold_stage1: float,
    two_stage_class_thresholds: dict[int, float],
    two_stage_threshold_tuning_cfg: dict[str, Any],
    two_stage_calibration_cfg: dict[str, Any],
    outlier_cfg: dict[str, Any],
    balancing_cfg: dict[str, Any],
    stage2_feature_bundle: dict[str, Any],
    output_dir: Path,
    seed: int,
    X_train_bal: pd.DataFrame,
    y_train_bal: pd.Series,
    artifacts: Any,
    param_overrides_cfg: dict[str, Any] | None,
    class_weight_requested_fn: Callable[[dict[str, Any] | None], bool],
    add_class_weight_metadata_metrics_fn: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None],
    run_two_stage_uct_model_fn: Callable[..., Any],
) -> dict[str, Any]:
    model_results: dict[str, Any] = {}
    leaderboard_rows: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    optuna_artifacts: dict[str, dict[str, str]] = {}
    model_decision_configs: dict[str, dict[str, Any]] = {}
    runtime_artifact_overrides_by_model: dict[str, dict[str, Any]] = {}
    stage2_feature_counts_by_model: dict[str, int] = {}
    stage2_advanced_reports_by_model: dict[str, dict[str, Any]] = {}
    successful_models: list[str] = []
    failed_models: dict[str, str] = {}

    for model_name in model_candidates:
        print(
            "[training][start] "
            f"model={model_name} "
            f"experiment_mode={experiment_mode} "
            "two_stage_enabled=True "
            f"branch={resolved_two_stage_branch} "
            f"uci_feature_builder={dataset_cfg.get('features', {}).get('builder', 'uci_student_features')} "
            f"outlier_enabled={bool(outlier_cfg.get('enabled', False))} "
            f"smote_enabled={bool(balancing_cfg.get('enabled', False))} "
            f"stage2_tuning_enabled={bool(resolved_stage2_optuna_cfg.get('enabled', False))} "
            f"stage2_decision_strategy={resolved_stage2_decision_cfg.get('strategy', 'argmax')}"
        )
        model_decision_rule_cfg = _resolve_model_decision_rule_config(
            exp_cfg=exp_cfg,
            base_decision_rule_cfg=decision_rule_cfg,
            model_name=model_name,
            formulation=formulation,
            two_stage_enabled=True,
            class_metadata=class_metadata,
        )
        model_decision_configs[model_name] = copy.deepcopy(model_decision_rule_cfg)
        model_n_trials = int(per_model_trial_budgets.get(model_name, default_n_trials))
        model_tuning_enabled = bool(tuning_backend == "optuna" and model_n_trials > 0)
        model_tuning_cfg = {**tuning_cfg, "n_trials": model_n_trials}
        model_param_overrides = {}
        if isinstance(param_overrides_cfg, dict):
            candidate_override = param_overrides_cfg.get(model_name, {})
            if isinstance(candidate_override, dict):
                model_param_overrides = candidate_override
        tuning_score = None
        tuning_details: dict[str, Any] = {}
        try:
            payload, trained_model, tuning_score, tuning_details, two_stage_optuna_paths = run_two_stage_uct_model_fn(
                model_name=model_name,
                mode_name=experiment_mode,
                decision_mode=str(two_stage_decision_mode),
                params_overrides=model_param_overrides,
                seed=seed,
                tuning_cfg=model_tuning_cfg,
                tuning_enabled=model_tuning_enabled,
                retrain_on_full_train_split=retrain_on_full_train_split,
                two_stage_cfg=two_stage_cfg,
                class_weight_cfg=class_weight_cfg if isinstance(class_weight_cfg, dict) else {},
                class_metadata=class_metadata,
                X_train=X_train_bal,
                y_train=y_train_bal,
                X_train_stage2_base=artifacts.X_train if bool(stage2_feature_bundle.get("enabled", False)) else None,
                y_train_stage2_base=artifacts.y_train if bool(stage2_feature_bundle.get("enabled", False)) else None,
                X_valid=artifacts.X_valid,
                y_valid=artifacts.y_valid,
                X_test=artifacts.X_test,
                y_test=artifacts.y_test,
                threshold_stage1=threshold_stage1,
                class_thresholds=two_stage_class_thresholds,
                threshold_tuning_cfg=two_stage_threshold_tuning_cfg,
                calibration_cfg=two_stage_calibration_cfg,
                outlier_cfg=outlier_cfg,
                balancing_cfg=balancing_cfg,
                stage2_feature_bundle=stage2_feature_bundle,
                output_dir=output_dir,
            )
            runtime_override = payload.pop("_runtime_artifact_overrides", None)
            if isinstance(runtime_override, dict):
                runtime_artifact_overrides_by_model[model_name] = runtime_override
            stage2_feature_counts_by_model[model_name] = int(
                payload.get("metrics", {}).get("stage2_feature_count_seen_at_training", np.nan)
            ) if pd.notna(payload.get("metrics", {}).get("stage2_feature_count_seen_at_training", np.nan)) else 0
            stage2_advanced_reports_by_model[model_name] = (
                dict(payload.get("artifacts", {}).get("two_stage", {}).get("feature_sharpening", {}))
                if isinstance(payload.get("artifacts", {}).get("two_stage", {}).get("feature_sharpening", {}), dict)
                else {}
            )
            payload["class_weight"] = dict(payload.get("class_weight", {}))
            payload["class_weight"]["class_weight_requested"] = bool(
                payload["class_weight"].get("class_weight_requested", False)
                or class_weight_requested_fn(class_weight_cfg)
                or bool(payload.get("metrics", {}).get("auto_balance_search_enabled", 0.0))
            )
            payload["class_weight"]["model_name"] = model_name
            add_class_weight_metadata_metrics_fn(payload["metrics"], payload["class_weight"], class_metadata)
            print(
                "[class_weight][model] "
                f"model={model_name} "
                f"requested={payload['class_weight'].get('class_weight_requested', False)} "
                f"applied={payload['class_weight'].get('class_weight_applied', False)} "
                f"method={payload['class_weight'].get('class_weight_application_method', 'none')} "
                f"mode={payload['class_weight'].get('mode', payload['class_weight'].get('strategy', 'none'))}"
            )
            _add_named_per_class_metrics(
                payload["metrics"],
                payload["artifacts"].get("per_class_metrics_test"),
                class_metadata.get("class_index_to_label", {}),
            )
            if tuning_details:
                payload["tuning"] = tuning_details
            if two_stage_optuna_paths:
                optuna_artifacts[model_name] = dict(two_stage_optuna_paths)
                payload.setdefault("artifact_paths", {}).update(two_stage_optuna_paths)
            model_results[model_name] = payload
            trained_models[model_name] = trained_model
            leaderboard_rows.append({"model": model_name, **payload["metrics"]})
            successful_models.append(model_name)
            print(
                "[training][success] "
                f"model={model_name} "
                f"test_macro_f1={float(payload['metrics'].get('test_macro_f1', np.nan)):.4f} "
                f"test_accuracy={float(payload['metrics'].get('test_accuracy', np.nan)):.4f} "
                f"test_f1_enrolled={float(payload['metrics'].get('test_f1_enrolled', np.nan)):.4f} "
                f"test_f1_graduate={float(payload['metrics'].get('test_f1_graduate', np.nan)):.4f} "
                f"test_f1_dropout={float(payload['metrics'].get('test_f1_dropout', np.nan)):.4f}"
            )
        except Exception as exc:
            model_results[model_name] = {"error": f"Training/evaluation failed: {exc}"}
            failed_models[model_name] = f"{type(exc).__name__}: {exc}"
            print(
                "[training][error] "
                f"model={model_name} "
                f"experiment_mode={experiment_mode} "
                "two_stage_enabled=True "
                f"decision_mode={two_stage_decision_mode} "
                f"stage2_tuning_enabled={bool(resolved_stage2_optuna_cfg.get('enabled', False))} "
                f"stage2_decision_strategy={resolved_stage2_decision_cfg.get('strategy', 'argmax')} "
                f"error={type(exc).__name__}: {exc}"
            )
            print(
                "[two_stage][stage2][error] "
                f"model={model_name} "
                f"reason={type(exc).__name__}: {exc}"
            )

    return {
        "model_results": model_results,
        "leaderboard_rows": leaderboard_rows,
        "trained_models": trained_models,
        "optuna_artifacts": optuna_artifacts,
        "model_decision_configs": model_decision_configs,
        "runtime_artifact_overrides_by_model": runtime_artifact_overrides_by_model,
        "stage2_feature_counts_by_model": stage2_feature_counts_by_model,
        "stage2_advanced_reports_by_model": stage2_advanced_reports_by_model,
        "successful_models": successful_models,
        "failed_models": failed_models,
    }
