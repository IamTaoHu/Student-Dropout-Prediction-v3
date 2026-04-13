from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.experiment.config_resolution import _resolve_model_decision_rule_config
from src.models.train_eval import (
    retrain_on_full_train_and_evaluate_test,
    run_leakage_safe_stratified_cv,
    train_and_evaluate,
    tune_model_with_optuna,
)
from src.reporting.prediction_exports import (
    _add_named_cv_per_class_metrics,
    _add_named_per_class_metrics,
    _add_named_per_class_metrics_with_suffix,
    _add_named_validation_per_class_metrics,
    _safe_filename_token,
)


def run_benchmark_mode(
    *,
    experiment_mode: str,
    exp_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    model_candidates: list[str],
    resolved_two_stage_branch: str,
    formulation: str,
    class_metadata: dict[str, Any],
    decision_rule_cfg: dict[str, Any],
    tuning_cfg: dict[str, Any],
    tuning_backend: str,
    default_n_trials: int,
    per_model_trial_budgets: dict[str, int],
    retrain_on_full_train_split: bool,
    class_weight_cfg: dict[str, Any] | None,
    threshold_tuning_cfg: dict[str, Any],
    paper_reproduction_mode: bool,
    cv_reporting_cfg: dict[str, Any],
    model_selection_cfg: dict[str, Any],
    param_overrides_cfg: dict[str, Any] | None,
    seed: int,
    preprocess_cfg: dict[str, Any],
    outlier_cfg: dict[str, Any],
    balancing_cfg: dict[str, Any],
    splits: dict[str, pd.DataFrame],
    X_train_bal: pd.DataFrame,
    y_train_bal: pd.Series,
    artifacts: Any,
    output_dir: Path,
    primary_metric: str,
    class_weight_requested_fn: Callable[[dict[str, Any] | None], bool],
    add_class_weight_metadata_metrics_fn: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None],
    run_multiclass_decision_autotune_fn: Callable[..., dict[str, Any]],
    run_validation_threshold_tuning_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    model_results: dict[str, Any] = {}
    leaderboard_rows: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    optuna_artifacts: dict[str, dict[str, str]] = {}
    model_decision_configs: dict[str, dict[str, Any]] = {}
    runtime_artifact_overrides_by_model: dict[str, dict[str, Any]] = {}
    successful_models: list[str] = []
    failed_models: dict[str, str] = {}

    for model_name in model_candidates:
        print(
            "[training][start] "
            f"model={model_name} "
            f"experiment_mode={experiment_mode} "
            "two_stage_enabled=False "
            f"branch={resolved_two_stage_branch} "
            f"uci_feature_builder={dataset_cfg.get('features', {}).get('builder', 'uci_student_features')} "
            f"outlier_enabled={bool(outlier_cfg.get('enabled', False))} "
            f"smote_enabled={bool(balancing_cfg.get('enabled', False))} "
            "stage2_tuning_enabled=False "
            "stage2_decision_strategy=argmax"
        )
        model_decision_rule_cfg = _resolve_model_decision_rule_config(
            exp_cfg=exp_cfg,
            base_decision_rule_cfg=decision_rule_cfg,
            model_name=model_name,
            formulation=formulation,
            two_stage_enabled=False,
            class_metadata=class_metadata,
        )
        model_decision_configs[model_name] = copy.deepcopy(model_decision_rule_cfg)
        model_n_trials = int(per_model_trial_budgets.get(model_name, default_n_trials))
        model_tuning_enabled = bool(tuning_backend == "optuna" and model_n_trials > 0)
        model_tuning_cfg = {**tuning_cfg, "n_trials": model_n_trials}
        params: dict[str, Any] = {}
        model_param_overrides = {}
        if isinstance(param_overrides_cfg, dict):
            candidate_override = param_overrides_cfg.get(model_name, {})
            if isinstance(candidate_override, dict):
                model_param_overrides = candidate_override
        tuning_score = None
        tuning_details: dict[str, Any] = {}
        warning_msg: str | None = None
        if model_tuning_enabled:
            try:
                objective_source = str(model_tuning_cfg.get("objective_source", "cv")).strip().lower()
                if paper_reproduction_mode:
                    objective_source = "paper_cv"
                params, tuning_score, tuning_details = tune_model_with_optuna(
                    model_name=model_name,
                    X_train=X_train_bal,
                    y_train=y_train_bal,
                    tuning_cfg={
                        **model_tuning_cfg,
                        "seed": seed,
                        "objective_source": objective_source,
                        "use_class_weights": class_weight_requested_fn(class_weight_cfg),
                        "class_weight": {
                            **(class_weight_cfg if isinstance(class_weight_cfg, dict) else {}),
                            "class_label_to_index": class_metadata.get("class_label_to_index", {}),
                        },
                        "cv_train_df": splits.get("train", pd.DataFrame()).reset_index(drop=True),
                        "cv_preprocess_config": preprocess_cfg,
                        "cv_outlier_config": outlier_cfg,
                        "cv_balancing_config": balancing_cfg,
                        "cv_config": cv_reporting_cfg,
                        "label_order": class_metadata.get("class_indices", []),
                        "decision_rule": model_decision_rule_cfg.get("decision_rule", "model_predict"),
                        "multiclass_decision": model_decision_rule_cfg.get("multiclass_decision", {}),
                        "trial_selection": {
                            "ranking_metrics": model_selection_cfg.get("ranking_metrics", []),
                        },
                    },
                    X_valid=artifacts.X_valid,
                    y_valid=artifacts.y_valid,
                    fixed_params=model_param_overrides,
                )
                model_token = _safe_filename_token(model_name)
                trials_path = output_dir / f"optuna_trials_{model_token}.csv"
                best_params_path = output_dir / f"optuna_best_params_{model_token}.json"
                trials_df = tuning_details.get("trials_dataframe")
                if isinstance(trials_df, pd.DataFrame):
                    trials_df.to_csv(trials_path, index=False)
                else:
                    pd.DataFrame(tuning_details.get("trials", [])).to_csv(trials_path, index=False)
                best_payload = {
                    "model": model_name,
                    "best_params": params,
                    "best_value": tuning_score,
                    "objective_source": tuning_details.get("objective_source"),
                    "best_validation_metrics": tuning_details.get("best_validation_metrics", {}),
                    "best_cv_aggregate_metrics": tuning_details.get("best_cv_aggregate_metrics", {}),
                    "best_per_class_metrics": tuning_details.get("best_per_class_metrics", {}),
                    "best_per_class_f1": tuning_details.get("best_per_class_f1", {}),
                }
                best_params_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
                optuna_artifacts[model_name] = {
                    "trials_csv": str(trials_path),
                    "best_params_json": str(best_params_path),
                }
            except Exception as exc:
                warning_msg = f"Tuning failed; using defaults. Reason: {exc}"
                params = dict(model_param_overrides)
                tuning_score = None
                tuning_details = {}
        elif model_param_overrides:
            params = dict(model_param_overrides)
        try:
            eval_cfg = {
                "seed": seed,
                "class_weight": {
                    **(class_weight_cfg if isinstance(class_weight_cfg, dict) else {}),
                    "class_label_to_index": class_metadata.get("class_label_to_index", {}),
                },
                "label_order": class_metadata.get("class_indices", []),
                "decision_rule": model_decision_rule_cfg.get("decision_rule", "model_predict"),
                "multiclass_decision": model_decision_rule_cfg.get("multiclass_decision", {}),
            }
            if retrain_on_full_train_split:
                prefit_result = train_and_evaluate(
                    model_name=model_name,
                    params=params,
                    X_train=X_train_bal,
                    y_train=y_train_bal,
                    X_valid=artifacts.X_valid,
                    y_valid=artifacts.y_valid,
                    X_test=artifacts.X_test,
                    y_test=artifacts.y_test,
                    eval_config=eval_cfg,
                )
                X_train_full = pd.concat([X_train_bal, artifacts.X_valid], axis=0).reset_index(drop=True)
                y_train_full = pd.concat([y_train_bal, artifacts.y_valid], axis=0).reset_index(drop=True)
                retrained_result = retrain_on_full_train_and_evaluate_test(
                    model_name=model_name,
                    params=params,
                    X_train_full=X_train_full,
                    y_train_full=y_train_full,
                    X_test=artifacts.X_test,
                    y_test=artifacts.y_test,
                    eval_config=eval_cfg,
                )
                merged_metrics = dict(prefit_result.metrics)
                merged_metrics.update(retrained_result.metrics)
                merged_artifacts = dict(retrained_result.artifacts)
                for key in ("y_true_valid", "y_pred_valid", "y_proba_valid"):
                    if key in prefit_result.artifacts:
                        merged_artifacts[key] = prefit_result.artifacts[key]
                prefit_runtime_override = prefit_result.artifacts.get("runtime_artifact_override")
                retrained_runtime_override = retrained_result.artifacts.get("runtime_artifact_override")
                if isinstance(retrained_runtime_override, dict):
                    merged_runtime_override = dict(retrained_runtime_override)
                    if isinstance(prefit_runtime_override, dict) and isinstance(prefit_runtime_override.get("X_valid"), pd.DataFrame):
                        merged_runtime_override["X_valid"] = prefit_runtime_override.get("X_valid").copy()
                    merged_artifacts["runtime_artifact_override"] = merged_runtime_override
                result = retrained_result
                result = type(result)(metrics=merged_metrics, artifacts=merged_artifacts)
            else:
                result = train_and_evaluate(
                    model_name=model_name,
                    params=params,
                    X_train=X_train_bal,
                    y_train=y_train_bal,
                    X_valid=artifacts.X_valid,
                    y_valid=artifacts.y_valid,
                    X_test=artifacts.X_test,
                    y_test=artifacts.y_test,
                    eval_config=eval_cfg,
                )
            payload = {
                "metrics": result.metrics,
                "artifacts": {k: v for k, v in result.artifacts.items() if k != "model"},
                "params": result.artifacts.get("params", params),
                "tuning_score": tuning_score,
            }
            runtime_override = payload["artifacts"].pop("runtime_artifact_override", None)
            if isinstance(runtime_override, dict):
                runtime_artifact_overrides_by_model[model_name] = runtime_override
            payload["class_weight"] = dict(result.artifacts.get("class_weight_info", {}))
            payload["class_weight"]["class_weight_requested"] = class_weight_requested_fn(class_weight_cfg)
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

            if cv_reporting_cfg.get("enabled", False):
                cv_eval = run_leakage_safe_stratified_cv(
                    model_name=model_name,
                    params=params,
                    train_df=splits.get("train", pd.DataFrame()).reset_index(drop=True),
                    preprocess_config=preprocess_cfg,
                    outlier_config=outlier_cfg,
                    balancing_config=balancing_cfg,
                    cv_config=cv_reporting_cfg,
                    eval_config={
                        "seed": seed,
                        "class_weight": {
                            **(class_weight_cfg if isinstance(class_weight_cfg, dict) else {}),
                            "class_label_to_index": class_metadata.get("class_label_to_index", {}),
                        },
                        "label_order": class_metadata.get("class_indices", []),
                        "decision_rule": model_decision_rule_cfg.get("decision_rule", "model_predict"),
                        "multiclass_decision": model_decision_rule_cfg.get("multiclass_decision", {}),
                    },
                )
                payload["cv_results"] = cv_eval
                aggregate_metrics = cv_eval.get("aggregate_metrics", {})
                if isinstance(aggregate_metrics, dict):
                    for key, value in aggregate_metrics.items():
                        if str(key) == "cv_num_folds":
                            payload["metrics"][str(key)] = int(value)
                        else:
                            payload["metrics"][str(key)] = float(value)
                per_class_aggregate = cv_eval.get("per_class_aggregate_metrics", {})
                if isinstance(per_class_aggregate, dict):
                    _add_named_cv_per_class_metrics(
                        payload["metrics"],
                        per_class_aggregate,
                        class_metadata.get("class_index_to_label", {}),
                    )
                payload["metrics"]["optuna_trials"] = float(int(model_n_trials)) if model_tuning_enabled else 0.0
                fold_rows = cv_eval.get("folds", []) if isinstance(cv_eval.get("folds", []), list) else []
                smote_skip_count = sum(
                    1
                    for fold in fold_rows
                    if bool(((fold.get("balancing", {}) if isinstance(fold.get("balancing", {}), dict) else {}).get("skipped", False)))
                )
                outlier_revert_count = sum(
                    1
                    for fold in fold_rows
                    if bool(((fold.get("outlier", {}) if isinstance(fold.get("outlier", {}), dict) else {}).get("reverted", False)))
                )
                print(
                    "[cv][completed] "
                    f"model={model_name} "
                    f"folds={int(payload['metrics'].get('cv_num_folds', 0))} "
                    f"cv_macro_f1={float(payload['metrics'].get('cv_macro_f1', np.nan)):.4f} "
                    f"cv_macro_precision={float(payload['metrics'].get('cv_macro_precision', np.nan)):.4f} "
                    f"cv_macro_recall={float(payload['metrics'].get('cv_macro_recall', np.nan)):.4f} "
                    f"smote_skipped_folds={smote_skip_count} "
                    f"if_reverted_folds={outlier_revert_count}"
                )
            if model_tuning_enabled:
                payload["optuna_summary"] = {
                    "model": model_name,
                    "objective_source": tuning_details.get("objective_source"),
                    "n_trials": int(model_n_trials),
                    "best_value": tuning_score,
                    "best_validation_metrics": tuning_details.get("best_validation_metrics", {}),
                    "best_cv_aggregate_metrics": tuning_details.get("best_cv_aggregate_metrics", {}),
                }

            payload["metrics"]["multiclass_decision_auto_tuned"] = 0.0
            payload["metrics"]["multiclass_decision_strategy"] = str(
                model_decision_rule_cfg.get("decision_rule", "model_predict")
            )
            payload["metrics"]["tuning_objective"] = str(
                model_decision_rule_cfg.get("multiclass_decision", {}).get("auto_tune", {}).get("objective", "macro_f1")
            )
            decision_auto_tune_result = run_multiclass_decision_autotune_fn(
                payload=payload,
                decision_rule_cfg=model_decision_rule_cfg,
                class_metadata=class_metadata,
            )
            payload["decision_policy_tuning"] = decision_auto_tune_result
            payload["decision_rule_config"] = copy.deepcopy(model_decision_rule_cfg)
            payload["metrics"]["multiclass_decision_auto_tuned"] = (
                1.0 if bool(decision_auto_tune_result.get("multiclass_decision_auto_tuned", False)) else 0.0
            )
            print(
                "[decision_policy][model] "
                f"model={model_name} "
                f"strategy={model_decision_rule_cfg.get('decision_rule')} "
                f"auto_tune={decision_auto_tune_result.get('multiclass_decision_auto_tuned', False)} "
                f"grid_size={decision_auto_tune_result.get('search_grid_size', 0)} "
                f"selected={decision_auto_tune_result.get('selected_parameters', {})} "
                f"best_valid={decision_auto_tune_result.get('validation_objective_score_at_selected_threshold')}"
            )

            threshold_result: dict[str, Any] = {"status": "skipped", "reason": "disabled"}
            if bool(threshold_tuning_cfg.get("enabled", False)):
                threshold_result = run_validation_threshold_tuning_fn(
                    payload=payload,
                    class_metadata=class_metadata,
                    threshold_cfg=threshold_tuning_cfg,
                )
            else:
                threshold_result = {
                    "status": "skipped",
                    "reason": "disabled",
                    "threshold_tuning_requested": False,
                    "threshold_tuning_supported": bool(payload["artifacts"].get("y_proba_valid") is not None),
                    "threshold_selection_split": "validation",
                    "threshold_applied_to": "none",
                    "default_decision_rule": "argmax",
                }
            payload["threshold_tuning"] = threshold_result

            if threshold_result.get("status") == "applied":
                baseline_metrics = {
                    k: float(v)
                    for k, v in payload["metrics"].items()
                    if isinstance(v, (int, float)) and k.startswith("test_")
                }
                for key, value in baseline_metrics.items():
                    payload["metrics"][f"{key}_default"] = float(value)
                baseline_per_class = payload["artifacts"].get("per_class_metrics_test")
                _add_named_per_class_metrics_with_suffix(
                    payload["metrics"],
                    baseline_per_class,
                    class_metadata.get("class_index_to_label", {}),
                    "default",
                )
                payload["artifacts"]["per_class_metrics_test_default"] = baseline_per_class
                payload["artifacts"]["y_pred_test_default"] = payload["artifacts"].get("y_pred_test")
                payload["artifacts"]["confusion_matrix_default"] = payload["artifacts"].get("confusion_matrix")

                tuned_metrics = threshold_result.get("test_tuned_metrics", {})
                payload["metrics"].update({f"test_{k}": float(v) for k, v in tuned_metrics.items()})
                payload["artifacts"]["per_class_metrics_test"] = threshold_result.get("test_tuned_per_class", {})
                if str(threshold_result.get("threshold_applied_to", "test")) == "test":
                    payload["artifacts"]["y_pred_test"] = threshold_result.get("y_pred_test_tuned", [])
                    payload["artifacts"]["confusion_matrix"] = threshold_result.get("confusion_matrix_tuned", [])
                payload["threshold_tuning"]["threshold_tuning_applied"] = bool(
                    str(payload["threshold_tuning"].get("threshold_applied_to", "none")) == "test"
                )
            else:
                payload["threshold_tuning"]["threshold_tuning_applied"] = False

            _add_named_per_class_metrics(
                payload["metrics"],
                payload["artifacts"].get("per_class_metrics_test"),
                class_metadata.get("class_index_to_label", {}),
            )
            _add_named_validation_per_class_metrics(
                payload["metrics"],
                payload["artifacts"].get("per_class_metrics_valid"),
                class_metadata.get("class_index_to_label", {}),
            )
            if tuning_details:
                payload["tuning"] = {
                    "scoring": str(model_tuning_cfg.get("scoring", "f1_macro")),
                    "cv_folds": int(model_tuning_cfg.get("cv_folds", 3)),
                    "objective_source": tuning_details.get("objective_source"),
                    "objective_metric": primary_metric,
                    "best_validation_metrics": tuning_details.get("best_validation_metrics", {}),
                    "best_per_class_metrics": tuning_details.get("best_per_class_metrics", {}),
                    "best_per_class_f1": tuning_details.get("best_per_class_f1", {}),
                    "n_trials": int(model_n_trials),
                }
            if model_name in optuna_artifacts:
                payload.setdefault("artifact_paths", {}).update(optuna_artifacts[model_name])
            if warning_msg:
                payload["warning"] = warning_msg
            model_results[model_name] = payload
            trained_models[model_name] = result.artifacts.get("model")
            leaderboard_rows.append({"model": model_name, **payload["metrics"]})
            successful_models.append(model_name)
            print(
                "[training][success] "
                f"model={model_name} "
                f"test_macro_f1={float(payload['metrics'].get('test_macro_f1', np.nan)):.4f} "
                f"test_accuracy={float(payload['metrics'].get('test_accuracy', np.nan)):.4f}"
            )
        except Exception as exc:
            model_results[model_name] = {"error": f"Training/evaluation failed: {exc}"}
            failed_models[model_name] = f"{type(exc).__name__}: {exc}"
            print(
                "[training][error] "
                f"model={model_name} "
                f"experiment_mode={experiment_mode} "
                "two_stage_enabled=False "
                f"error={type(exc).__name__}: {exc}"
            )

    return {
        "model_results": model_results,
        "leaderboard_rows": leaderboard_rows,
        "trained_models": trained_models,
        "optuna_artifacts": optuna_artifacts,
        "model_decision_configs": model_decision_configs,
        "runtime_artifact_overrides_by_model": runtime_artifact_overrides_by_model,
        "stage2_feature_counts_by_model": {},
        "stage2_advanced_reports_by_model": {},
        "successful_models": successful_models,
        "failed_models": failed_models,
    }
