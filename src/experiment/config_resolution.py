from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from src.data.feature_builders.uct_stage2_advanced_features import (
    DEFAULT_INTERACTION_GROUPS,
    DEFAULT_PROTOTYPE_METRIC_SET,
)
from src.data.feature_builders.uct_stage2_feature_separation import (
    DEFAULT_ADVANCED_ENROLLED_FEATURE_SEPARATION_GROUPS,
)
from src.data.feature_builders.uct_stage2_feature_sharpening import (
    DEFAULT_STAGE2_FEATURE_GROUPS,
)

DATASET_NAME_ALIASES = {
    "uct": "uct_student",
    "uci": "uct_student",
    "uci_student": "uct_student",
    "uci_student_presplit_parquet": "uct_student",
    "uci-student": "uct_student",
    "uct_student": "uct_student",
    "uct-student": "uct_student",
    "oulad": "oulad",
    "open university learning analytics dataset": "oulad",
}

def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load experiment configs. Install with `pip install pyyaml`.") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_experiment_config_schema(exp_cfg: dict[str, Any]) -> dict[str, Any]:
    """Accept both native and lightweight analysis-only experiment schemas."""
    if "experiment" in exp_cfg:
        return exp_cfg

    experiment_name = exp_cfg.get("experiment_name")
    task_type = str(exp_cfg.get("task_type", "")).strip().lower()
    dataset_name = str(exp_cfg.get("dataset", {}).get("name", "")).strip().lower()
    input_cfg = exp_cfg.get("input", {})
    output_cfg = exp_cfg.get("output", {})
    analysis_cfg = exp_cfg.get("analysis", {})
    evaluation_cfg = exp_cfg.get("evaluation", {})
    preprocessing_cfg = exp_cfg.get("preprocessing", {})

    if not experiment_name:
        return exp_cfg

    mode = "benchmark"
    if task_type == "analysis_only":
        mode = "error_audit"
    elif task_type in {"threshold_tuning", "posthoc_threshold_tuning"}:
        mode = "threshold_tuning"
    target_mode = str(exp_cfg.get("target", {}).get("mode", "")).strip().lower()
    target_formulation = "binary"
    if "3class" in dataset_name or "three_class" in dataset_name or target_mode in {"3class", "three_class"}:
        target_formulation = "three_class"
    elif target_mode in {"4class", "four_class"}:
        target_formulation = "four_class"

    dataset_config = "configs/datasets/uct_student.yaml"
    if dataset_name == "oulad":
        dataset_config = "configs/datasets/oulad.yaml"

    model_aliases = {
        "xgboost_optuna": "xgboost",
        "lightgbm_optuna": "lightgbm",
        "catboost_optuna": "catboost",
    }
    raw_models = exp_cfg.get("models", [])
    candidates: list[str] = []
    if isinstance(raw_models, list):
        for model in raw_models:
            token = str(model).strip().lower()
            candidates.append(model_aliases.get(token, token.removesuffix("_optuna")))
    if not candidates:
        candidates = ["xgboost", "lightgbm", "catboost"]

    optimization_cfg = exp_cfg.get("optimization", {})
    threshold_tuning_cfg = exp_cfg.get("threshold_tuning", {})
    eval_metrics = [str(m) for m in evaluation_cfg.get("metrics", [])]
    objective_metric = str(optimization_cfg.get("objective_metric", "macro_f1"))
    if mode == "threshold_tuning":
        objective_metric = str(threshold_tuning_cfg.get("objective_metric", "macro_f1"))
    scoring_name = "f1_macro" if objective_metric == "macro_f1" else objective_metric
    secondary_metrics = [m for m in eval_metrics if m and m != objective_metric]
    split_cfg = exp_cfg.get("split", {})
    random_state = int(split_cfg.get("random_state", 42))

    output_results_dir = output_cfg.get("results_dir") or output_cfg.get("dir") or f"results/{experiment_name}"

    training_class_weight_cfg = (
        exp_cfg.get("training", {}).get("class_weight", {})
        if isinstance(exp_cfg.get("training", {}).get("class_weight", {}), dict)
        else {}
    )
    class_weight_payload: dict[str, Any] = {"enabled": bool(exp_cfg.get("training", {}).get("use_class_weights", False))}
    if training_class_weight_cfg:
        class_weight_payload = {"enabled": True, **training_class_weight_cfg}

    normalized: dict[str, Any] = {
        "experiment": {
            "id": str(experiment_name),
            "mode": mode,
            "seed": random_state,
            "dataset_config": dataset_config,
            "target_formulation": target_formulation,
        },
        "splits": {
            "test_size": 0.2,
            "validation_size": 0.2,
            "stratify_column": "target",
        },
        "preprocessing": {
            "imputation": preprocessing_cfg.get("imputation", "median_mode"),
            "encoding": preprocessing_cfg.get("encoding", "onehot"),
            "scaling": preprocessing_cfg.get("scaling", "standard"),
            "outlier": preprocessing_cfg.get("outlier", {"enabled": True, "method": "isolation_forest"}),
            "balancing": preprocessing_cfg.get("balancing", {"enabled": True, "method": "smote"}),
        },
        "inputs": {
            "benchmark_summary_path": input_cfg.get("benchmark_summary"),
            "benchmark_results_root": "results",
        },
        "analysis": {
            "top_k_models": analysis_cfg.get("top_k_models"),
            "models": analysis_cfg.get("models"),
            "threshold_grid": analysis_cfg.get("threshold_grid") or threshold_tuning_cfg.get("threshold_grid"),
            "split_source": split_cfg.get("source", "validation"),
        },
        "models": {
            "candidates": candidates,
            "tuning": {
                "backend": str(optimization_cfg.get("engine", "none")),
                "n_trials": int(optimization_cfg.get("n_trials", 0)),
                "cv_folds": 3,
                "scoring": scoring_name,
                "objective_source": "validation",
            },
            "class_weight": class_weight_payload,
            "retrain_on_full_train_split": True,
        },
        "metrics": {
            "primary": objective_metric,
            "secondary": secondary_metrics,
        },
        "evaluation": {
            "metrics": eval_metrics,
            "compare_default_argmax": bool(evaluation_cfg.get("compare_default_argmax", False)),
            "decision_rule": str(evaluation_cfg.get("decision_rule", exp_cfg.get("training", {}).get("decision_rule", "model_predict"))),
            "cross_validation": evaluation_cfg.get("cross_validation", {}),
            "save_confusion_matrix": bool(evaluation_cfg.get("save_confusion_matrix", True)),
            "save_classification_report": bool(evaluation_cfg.get("save_classification_report", True)),
        },
        "inference": exp_cfg.get("inference", {}) if isinstance(exp_cfg.get("inference", {}), dict) else {},
        "artifact_policy": exp_cfg.get("artifact_policy", {}),
        "outputs": {
            "results_dir": output_results_dir,
            "save_artifact_manifest": bool(output_cfg.get("save_artifact_manifest", True)),
        },
    }
    if mode == "threshold_tuning":
        # Preserve defaults in threshold-tuning runner if grid is not explicitly provided.
        normalized["analysis"]["models"] = analysis_cfg.get("models")
        normalized["analysis"]["threshold_grid"] = (
            analysis_cfg.get("threshold_grid")
            or threshold_tuning_cfg.get("threshold_grid")
            or {}
        )
    return normalized


def _normalize_dataset_name(raw_name: str) -> str:
    normalized = raw_name.strip().lower()
    return DATASET_NAME_ALIASES.get(normalized, normalized)


def _resolve_decision_rule_config(
    exp_cfg: dict[str, Any],
    formulation: str,
    two_stage_enabled: bool,
    class_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation_cfg = exp_cfg.get("evaluation", {}) if isinstance(exp_cfg.get("evaluation", {}), dict) else {}
    training_cfg = exp_cfg.get("training", {}) if isinstance(exp_cfg.get("training", {}), dict) else {}
    inference_cfg = exp_cfg.get("inference", {}) if isinstance(exp_cfg.get("inference", {}), dict) else {}
    multiclass_decision_cfg = (
        inference_cfg.get("multiclass_decision", {})
        if isinstance(inference_cfg.get("multiclass_decision", {}), dict)
        else {}
    )

    training_mode = str(training_cfg.get("mode", "")).strip().lower()
    requested_rule_legacy = str(
        evaluation_cfg.get("decision_rule", training_cfg.get("decision_rule", "model_predict"))
    ).strip().lower()
    requested_strategy = str(multiclass_decision_cfg.get("strategy", "")).strip().lower()
    if requested_strategy:
        requested_rule = requested_strategy
    elif training_mode in {"multiclass_argmax", "paper_multiclass_argmax"}:
        requested_rule = "argmax"
    else:
        requested_rule = requested_rule_legacy

    supported_rules = {"model_predict", "argmax", "enrolled_margin", "enrolled_middle_band", "enrolled_push"}
    if requested_rule not in supported_rules:
        raise ValueError(
            "Unsupported decision rule strategy. Use evaluation.decision_rule in "
            "{'model_predict','argmax'} or inference.multiclass_decision.strategy in "
            "{'argmax','enrolled_margin','enrolled_middle_band','enrolled_push'} "
            f"(got '{requested_rule}')."
        )

    label_indices = set()
    if isinstance(class_metadata, dict):
        raw_indices = class_metadata.get("class_indices", [])
        if isinstance(raw_indices, list):
            label_indices = {int(v) for v in raw_indices}

    strategy_params: dict[str, Any] = {"strategy": requested_rule}
    if requested_rule == "enrolled_margin":
        if formulation != "three_class":
            raise ValueError(
                "inference.multiclass_decision.strategy='enrolled_margin' is only supported for "
                "target_formulation='three_class'."
            )
        if label_indices and label_indices != {0, 1, 2}:
            raise ValueError(
                "strategy='enrolled_margin' requires class indices {0,1,2} "
                "(dropout/enrolled/graduate) in class metadata."
            )
        margin_threshold_raw = multiclass_decision_cfg.get("enrolled_margin_threshold", 0.10)
        margin_threshold = float(margin_threshold_raw)
        if margin_threshold < 0.0 or margin_threshold > 1.0:
            raise ValueError("inference.multiclass_decision.enrolled_margin_threshold must be within [0.0, 1.0].")
        strategy_params["enrolled_margin_threshold"] = margin_threshold

    if requested_rule == "enrolled_middle_band":
        if formulation != "three_class":
            raise ValueError(
                "inference.multiclass_decision.strategy='enrolled_middle_band' is only supported for "
                "target_formulation='three_class'."
            )
        if label_indices and label_indices != {0, 1, 2}:
            raise ValueError(
                "strategy='enrolled_middle_band' requires class indices {0,1,2} "
                "(dropout/enrolled/graduate) in class metadata."
            )
        dropout_threshold_raw = multiclass_decision_cfg.get("dropout_threshold", 0.55)
        graduate_threshold_raw = multiclass_decision_cfg.get("graduate_threshold", 0.55)
        dropout_threshold = float(dropout_threshold_raw)
        graduate_threshold = float(graduate_threshold_raw)
        if dropout_threshold < 0.0 or dropout_threshold > 1.0:
            raise ValueError("inference.multiclass_decision.dropout_threshold must be within [0.0, 1.0].")
        if graduate_threshold < 0.0 or graduate_threshold > 1.0:
            raise ValueError("inference.multiclass_decision.graduate_threshold must be within [0.0, 1.0].")
        strategy_params["dropout_threshold"] = dropout_threshold
        strategy_params["graduate_threshold"] = graduate_threshold

        class_label_to_index = (
            class_metadata.get("class_label_to_index", {})
            if isinstance(class_metadata, dict) and isinstance(class_metadata.get("class_label_to_index", {}), dict)
            else {}
        )
        canonical_label_lookup = {str(key).strip().lower(): int(value) for key, value in class_label_to_index.items()}

        def _resolve_named_label(raw_value: Any, *, field_name: str, default_index: int) -> int:
            if raw_value is None:
                return int(default_index)
            token = str(raw_value).strip()
            if not token:
                return int(default_index)
            if token.lstrip("-").isdigit():
                return int(token)
            resolved = canonical_label_lookup.get(token.lower())
            if resolved is None:
                raise ValueError(
                    f"inference.multiclass_decision.enrolled_decision_tuning.{field_name} "
                    f"must match a mapped class label or class index (got {raw_value!r})."
                )
            return int(resolved)

        tuning_raw = multiclass_decision_cfg.get("enrolled_decision_tuning", {})
        if not isinstance(tuning_raw, dict):
            tuning_raw = {}
        tuning_enabled = bool(tuning_raw.get("enabled", False))
        tuning_payload: dict[str, Any] = {"enabled": tuning_enabled}
        if tuning_enabled:
            enrolled_label = _resolve_named_label(
                tuning_raw.get("enrolled_label", "Enrolled"),
                field_name="enrolled_label",
                default_index=1,
            )
            dropout_label = _resolve_named_label(
                tuning_raw.get("dropout_label", "Dropout"),
                field_name="dropout_label",
                default_index=0,
            )
            graduate_label = _resolve_named_label(
                tuning_raw.get("graduate_label", "Graduate"),
                field_name="graduate_label",
                default_index=2,
            )
            if len({enrolled_label, dropout_label, graduate_label}) != 3:
                raise ValueError(
                    "inference.multiclass_decision.enrolled_decision_tuning labels must resolve to three distinct classes."
                )

            enrolled_min_proba = float(tuning_raw.get("enrolled_min_proba", 0.30))
            enrolled_margin_gap = float(tuning_raw.get("enrolled_margin_gap", 0.08))
            ambiguity_max_gap = float(tuning_raw.get("ambiguity_max_gap", 0.12))
            high_confidence_guard = float(tuning_raw.get("high_confidence_guard", 0.62))
            graduate_guard_max = float(tuning_raw.get("graduate_guard_max", 0.62))
            dropout_guard_max = float(tuning_raw.get("dropout_guard_max", 0.62))

            for field_name, value in [
                ("enrolled_min_proba", enrolled_min_proba),
                ("enrolled_margin_gap", enrolled_margin_gap),
                ("ambiguity_max_gap", ambiguity_max_gap),
                ("high_confidence_guard", high_confidence_guard),
                ("graduate_guard_max", graduate_guard_max),
                ("dropout_guard_max", dropout_guard_max),
            ]:
                if value < 0.0 or value > 1.0:
                    raise ValueError(
                        f"inference.multiclass_decision.enrolled_decision_tuning.{field_name} must be within [0.0, 1.0]."
                    )

            tuning_payload.update(
                {
                    "enabled": True,
                    "enrolled_label": int(enrolled_label),
                    "dropout_label": int(dropout_label),
                    "graduate_label": int(graduate_label),
                    "enrolled_min_proba": float(enrolled_min_proba),
                    "enrolled_margin_gap": float(enrolled_margin_gap),
                    "ambiguity_max_gap": float(ambiguity_max_gap),
                    "high_confidence_guard": float(high_confidence_guard),
                    "graduate_guard_max": float(graduate_guard_max),
                    "dropout_guard_max": float(dropout_guard_max),
                    "require_enrolled_above_baseline": bool(tuning_raw.get("require_enrolled_above_baseline", True)),
                }
            )
        strategy_params["enrolled_decision_tuning"] = tuning_payload

    if requested_rule == "enrolled_push":
        if formulation != "three_class":
            raise ValueError(
                "inference.multiclass_decision.strategy='enrolled_push' is only supported for "
                "target_formulation='three_class'."
            )
        if label_indices and label_indices != {0, 1, 2}:
            raise ValueError(
                "strategy='enrolled_push' requires class indices {0,1,2} "
                "(dropout/enrolled/graduate) in class metadata."
            )
        enrolled_class_name = str(multiclass_decision_cfg.get("enrolled_class_name", "Enrolled")).strip() or "Enrolled"
        threshold_block = multiclass_decision_cfg.get("enrolled_probability_threshold", {})
        if not isinstance(threshold_block, dict):
            threshold_block = {"enabled": threshold_block is not None, "value": threshold_block}
        threshold_enabled = bool(threshold_block.get("enabled", False))
        threshold_value_raw = threshold_block.get("value", threshold_block.get("threshold", 0.40))
        threshold_payload: dict[str, Any] = {"enabled": threshold_enabled}
        if threshold_enabled:
            threshold_value = float(threshold_value_raw)
            if threshold_value < 0.0 or threshold_value > 1.0:
                raise ValueError("inference.multiclass_decision.enrolled_probability_threshold must be within [0.0, 1.0].")
            threshold_payload["value"] = threshold_value

        middle_band_raw = multiclass_decision_cfg.get("enrolled_middle_band", {})
        if not isinstance(middle_band_raw, dict):
            middle_band_raw = {}
        middle_band_enabled = bool(middle_band_raw.get("enabled", False))
        middle_band_payload: dict[str, Any] = {"enabled": middle_band_enabled}
        if middle_band_enabled:
            min_prob = float(middle_band_raw.get("min_enrolled_prob", 0.30))
            max_gap = float(middle_band_raw.get("max_top2_gap", 0.05))
            if min_prob < 0.0 or min_prob > 1.0:
                raise ValueError("inference.multiclass_decision.enrolled_middle_band.min_enrolled_prob must be within [0.0, 1.0].")
            if max_gap < 0.0 or max_gap > 1.0:
                raise ValueError("inference.multiclass_decision.enrolled_middle_band.max_top2_gap must be within [0.0, 1.0].")
            middle_band_payload.update(
                {
                    "min_enrolled_prob": min_prob,
                    "max_top2_gap": max_gap,
                }
            )
        strategy_params.update(
            {
                "enrolled_class_name": enrolled_class_name,
                "fallback": str(multiclass_decision_cfg.get("fallback", "argmax")).strip().lower() or "argmax",
                "enrolled_probability_threshold": threshold_payload,
                "enrolled_middle_band": middle_band_payload,
            }
        )

    if requested_rule == "argmax":
        # Always keep strategy payload explicit for downstream auditing and summaries.
        strategy_params = {"strategy": "argmax"}
    if requested_rule == "model_predict":
        strategy_params = {"strategy": "model_predict"}

    auto_tune_cfg_raw = multiclass_decision_cfg.get("auto_tune", {})
    auto_tune_cfg = auto_tune_cfg_raw if isinstance(auto_tune_cfg_raw, dict) else {}
    auto_tune_enabled = bool(auto_tune_cfg.get("enabled", False))
    if auto_tune_enabled:
        if requested_rule not in {"enrolled_margin", "enrolled_middle_band", "enrolled_push"}:
            raise ValueError(
                "inference.multiclass_decision.auto_tune.enabled=true is only supported for "
                "strategies {'enrolled_margin','enrolled_middle_band','enrolled_push'}."
            )
        objective = str(auto_tune_cfg.get("objective", "macro_f1")).strip().lower()
        if objective not in {"macro_f1", "enrolled_f1", "enrolled_recall", "balanced_accuracy"}:
            raise ValueError(
                "inference.multiclass_decision.auto_tune.objective must be one of "
                "{'macro_f1','enrolled_f1','enrolled_recall','balanced_accuracy'}."
            )
        split_name = str(auto_tune_cfg.get("split", "validation")).strip().lower()
        if split_name != "validation":
            raise ValueError("inference.multiclass_decision.auto_tune.split must be 'validation'.")
        search_cfg = auto_tune_cfg.get("search", {}) if isinstance(auto_tune_cfg.get("search", {}), dict) else {}
        search_method = str(search_cfg.get("method", "grid")).strip().lower()
        if search_method != "grid":
            raise ValueError("inference.multiclass_decision.auto_tune.search.method must be 'grid'.")
        if requested_rule == "enrolled_margin":
            grid_vals = search_cfg.get("enrolled_margin_thresholds", [])
            if not isinstance(grid_vals, list) or len(grid_vals) == 0:
                raise ValueError(
                    "inference.multiclass_decision.auto_tune.search.enrolled_margin_thresholds "
                    "must be a non-empty list."
                )
            for raw in grid_vals:
                val = float(raw)
                if val < 0.0 or val > 1.0:
                    raise ValueError("enrolled_margin_threshold search values must be within [0.0, 1.0].")
        if requested_rule == "enrolled_middle_band":
            drop_vals = search_cfg.get("dropout_thresholds", [])
            grad_vals = search_cfg.get("graduate_thresholds", [])
            if not isinstance(drop_vals, list) or len(drop_vals) == 0:
                raise ValueError(
                    "inference.multiclass_decision.auto_tune.search.dropout_thresholds must be a non-empty list."
                )
            if not isinstance(grad_vals, list) or len(grad_vals) == 0:
                raise ValueError(
                    "inference.multiclass_decision.auto_tune.search.graduate_thresholds must be a non-empty list."
                )
            for raw in [*drop_vals, *grad_vals]:
                val = float(raw)
                if val < 0.0 or val > 1.0:
                    raise ValueError("middle-band search threshold values must be within [0.0, 1.0].")
        if requested_rule == "enrolled_push":
            threshold_cfg = strategy_params.get("enrolled_probability_threshold", {})
            middle_band_cfg = strategy_params.get("enrolled_middle_band", {})
            if bool(threshold_cfg.get("enabled", False)):
                threshold_search = search_cfg.get("enrolled_probability_thresholds", [])
                if not isinstance(threshold_search, list) or len(threshold_search) == 0:
                    raise ValueError(
                        "inference.multiclass_decision.auto_tune.search.enrolled_probability_thresholds "
                        "must be a non-empty list when enrolled_probability_threshold is enabled."
                    )
                for raw in threshold_search:
                    val = float(raw)
                    if val < 0.0 or val > 1.0:
                        raise ValueError("enrolled_probability_threshold search values must be within [0.0, 1.0].")
            if bool(middle_band_cfg.get("enabled", False)):
                min_prob_search = search_cfg.get("min_enrolled_probs", [])
                max_gap_search = search_cfg.get("max_top2_gaps", [])
                if not isinstance(min_prob_search, list) or len(min_prob_search) == 0:
                    raise ValueError(
                        "inference.multiclass_decision.auto_tune.search.min_enrolled_probs must be a non-empty list "
                        "when enrolled_middle_band is enabled."
                    )
                if not isinstance(max_gap_search, list) or len(max_gap_search) == 0:
                    raise ValueError(
                        "inference.multiclass_decision.auto_tune.search.max_top2_gaps must be a non-empty list "
                        "when enrolled_middle_band is enabled."
                    )
                for raw in [*min_prob_search, *max_gap_search]:
                    val = float(raw)
                    if val < 0.0 or val > 1.0:
                        raise ValueError("enrolled_push middle-band search values must be within [0.0, 1.0].")
        strategy_params["auto_tune"] = {
            "enabled": True,
            "objective": objective,
            "split": split_name,
            "search": {
                **search_cfg,
                "method": "grid",
            },
        }
    else:
        strategy_params["auto_tune"] = {"enabled": False}

    if two_stage_enabled and requested_rule != "model_predict":
        return {
            "decision_rule": "model_predict",
            "requested_decision_rule": requested_rule,
            "requested_decision_rule_legacy": requested_rule_legacy,
            "multiclass_decision": {"strategy": "model_predict"},
            "overridden_reason": "two_stage_runner_controls_decision_logic",
            "training_mode": training_mode,
            "target_formulation": formulation,
        }
    return {
        "decision_rule": requested_rule,
        "requested_decision_rule": requested_rule,
        "requested_decision_rule_legacy": requested_rule_legacy,
        "multiclass_decision": strategy_params,
        "overridden_reason": None,
        "training_mode": training_mode,
        "target_formulation": formulation,
    }


def _resolve_cv_reporting_config(
    exp_cfg: dict[str, Any],
    seed: int,
    paper_reproduction_mode: bool,
) -> dict[str, Any]:
    evaluation_cfg = exp_cfg.get("evaluation", {}) if isinstance(exp_cfg.get("evaluation", {}), dict) else {}
    cv_cfg = evaluation_cfg.get("cross_validation", {})
    if not isinstance(cv_cfg, dict):
        cv_cfg = {}

    enabled = bool(cv_cfg.get("enabled", False))
    if paper_reproduction_mode:
        enabled = True

    n_splits = int(cv_cfg.get("n_splits", cv_cfg.get("folds", 5)))
    n_splits = max(2, n_splits)
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = int(cv_cfg.get("random_state", seed))

    return {
        "enabled": enabled,
        "n_splits": n_splits,
        "shuffle": shuffle,
        "random_state": random_state,
        "optuna_objective_metric": "cv_macro_f1_mean",
    }


def _resolve_model_selection_config(exp_cfg: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = exp_cfg.get("selection", {}) if isinstance(exp_cfg.get("selection", {}), dict) else {}
    enabled = bool(raw_cfg)
    evaluation_cfg = exp_cfg.get("evaluation", {}) if isinstance(exp_cfg.get("evaluation", {}), dict) else {}
    primary = str(evaluation_cfg.get("primary_selection_metric", raw_cfg.get("primary", "macro_f1"))).strip()
    secondary = str(raw_cfg.get("secondary", "balanced_accuracy")).strip()
    tertiary = str(raw_cfg.get("tertiary", "accuracy")).strip()
    tie_breakers_raw = raw_cfg.get("tie_breakers", [])
    tie_breakers: list[str] = []
    if isinstance(tie_breakers_raw, list):
        tie_breakers = [str(metric).strip() for metric in tie_breakers_raw if str(metric).strip()]
    ranking_metrics = [metric for metric in [primary or "macro_f1", *tie_breakers] if metric]
    if len(ranking_metrics) == 1:
        for fallback in [secondary or "balanced_accuracy", tertiary or "accuracy"]:
            if fallback and fallback not in ranking_metrics:
                ranking_metrics.append(fallback)
    return {
        "enabled": enabled,
        "primary": primary or "macro_f1",
        "primary_selection_metric": primary or "macro_f1",
        "secondary": secondary or "balanced_accuracy",
        "tertiary": tertiary or "accuracy",
        "tie_breakers": tie_breakers,
        "ranking_metrics": ranking_metrics,
    }


def _resolve_experiment_feature_config(
    exp_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
) -> dict[str, Any]:
    resolved = copy.deepcopy(dataset_cfg)
    experiment_feature_cfg = (
        exp_cfg.get("feature_engineering", {})
        if isinstance(exp_cfg.get("feature_engineering", {}), dict)
        else {}
    )
    if not experiment_feature_cfg:
        return resolved
    dataset_feature_overrides = (
        experiment_feature_cfg.get("dataset_features", {})
        if isinstance(experiment_feature_cfg.get("dataset_features", {}), dict)
        else {}
    )
    if dataset_feature_overrides:
        resolved["features"] = _deep_merge_dicts(
            resolved.get("features", {}) if isinstance(resolved.get("features", {}), dict) else {},
            dataset_feature_overrides,
        )
    return resolved


def _resolve_two_stage_stage2_feature_sharpening_config(two_stage_cfg: dict[str, Any]) -> dict[str, Any]:
    stage2_cfg = two_stage_cfg.get("stage2", {}) if isinstance(two_stage_cfg.get("stage2", {}), dict) else {}
    raw_cfg = (
        stage2_cfg.get("feature_sharpening", {})
        if isinstance(stage2_cfg.get("feature_sharpening", {}), dict)
        else {}
    )
    enabled = bool(raw_cfg.get("enabled", False))
    raw_groups = raw_cfg.get("groups", [])
    groups: list[str] = []
    if isinstance(raw_groups, list):
        for item in raw_groups:
            group = str(item).strip().lower()
            if group and group not in groups:
                groups.append(group)
    if enabled and not groups:
        groups = list(DEFAULT_STAGE2_FEATURE_GROUPS)
    return {
        "enabled": enabled,
        "groups": groups,
        "default_groups": list(DEFAULT_STAGE2_FEATURE_GROUPS),
    }


def _resolve_two_stage_stage2_advanced_config(two_stage_cfg: dict[str, Any]) -> dict[str, Any]:
    stage2_cfg = two_stage_cfg.get("stage2", {}) if isinstance(two_stage_cfg.get("stage2", {}), dict) else {}
    raw_cfg = (
        stage2_cfg.get("advanced_enrolled_separation", {})
        if isinstance(stage2_cfg.get("advanced_enrolled_separation", {}), dict)
        else {}
    )
    enabled = bool(raw_cfg.get("enabled", False))

    interaction_raw = (
        raw_cfg.get("interaction_features", {})
        if isinstance(raw_cfg.get("interaction_features", {}), dict)
        else {}
    )
    interaction_enabled = enabled and bool(interaction_raw.get("enabled", False))
    interaction_groups: list[str] = []
    if isinstance(interaction_raw.get("groups", []), list):
        for item in interaction_raw.get("groups", []):
            group = str(item).strip().lower()
            if group and group not in interaction_groups:
                interaction_groups.append(group)
    if interaction_enabled and not interaction_groups:
        interaction_groups = list(DEFAULT_INTERACTION_GROUPS)

    prototype_raw = (
        raw_cfg.get("prototype_distance", {})
        if isinstance(raw_cfg.get("prototype_distance", {}), dict)
        else {}
    )
    stage2_robust_prototype_raw = (
        stage2_cfg.get("robust_prototypes", {})
        if isinstance(stage2_cfg.get("robust_prototypes", {}), dict)
        else {}
    )
    if stage2_robust_prototype_raw:
        enabled = True
        prototype_raw = stage2_robust_prototype_raw
    prototype_enabled = enabled and bool(prototype_raw.get("enabled", False))
    prototype_metric_set: list[str] = []
    if isinstance(prototype_raw.get("metric_set", []), list):
        for item in prototype_raw.get("metric_set", []):
            metric = str(item).strip().lower()
            if metric and metric not in prototype_metric_set:
                prototype_metric_set.append(metric)
    if prototype_enabled and not prototype_metric_set and not stage2_robust_prototype_raw:
        prototype_metric_set = list(DEFAULT_PROTOTYPE_METRIC_SET)

    return {
        "enabled": enabled,
        "interaction_features": {
            "enabled": interaction_enabled,
            "groups": interaction_groups,
            "default_groups": list(DEFAULT_INTERACTION_GROUPS),
        },
        "prototype_distance": {
            "enabled": prototype_enabled,
            "metric_set": prototype_metric_set,
            "default_metric_set": list(DEFAULT_PROTOTYPE_METRIC_SET),
        },
    }


def _resolve_two_stage_stage2_feature_separation_config(two_stage_cfg: dict[str, Any]) -> dict[str, Any]:
    stage2_cfg = two_stage_cfg.get("stage2", {}) if isinstance(two_stage_cfg.get("stage2", {}), dict) else {}
    raw_cfg = (
        stage2_cfg.get("advanced_enrolled_feature_separation", {})
        if isinstance(stage2_cfg.get("advanced_enrolled_feature_separation", {}), dict)
        else {}
    )
    enabled = bool(raw_cfg.get("enabled", False))
    strict_mode = bool(raw_cfg.get("strict_mode", False))
    create_composite_scores = bool(raw_cfg.get("create_composite_scores", True))
    apply_only_to_stage2 = bool(raw_cfg.get("apply_only_to_stage2", True))
    groups: list[str] = []
    raw_groups = raw_cfg.get("feature_groups", raw_cfg.get("groups", []))
    if isinstance(raw_groups, list):
        for item in raw_groups:
            group = str(item).strip().lower()
            if group and group in DEFAULT_ADVANCED_ENROLLED_FEATURE_SEPARATION_GROUPS and group not in groups:
                groups.append(group)
    if enabled and not groups:
        groups = list(DEFAULT_ADVANCED_ENROLLED_FEATURE_SEPARATION_GROUPS)
    return {
        "enabled": enabled,
        "strict_mode": strict_mode,
        "feature_groups": groups,
        "default_feature_groups": list(DEFAULT_ADVANCED_ENROLLED_FEATURE_SEPARATION_GROUPS),
        "create_composite_scores": create_composite_scores,
        "apply_only_to_stage2": apply_only_to_stage2,
    }


def _resolve_two_stage_stage2_selective_interactions_config(two_stage_cfg: dict[str, Any]) -> dict[str, Any]:
    stage2_cfg = two_stage_cfg.get("stage2", {}) if isinstance(two_stage_cfg.get("stage2", {}), dict) else {}
    raw_cfg = (
        stage2_cfg.get("selective_interactions", {})
        if isinstance(stage2_cfg.get("selective_interactions", {}), dict)
        else {}
    )
    allowlist: list[str] = []
    if isinstance(raw_cfg.get("feature_allowlist", []), list):
        for item in raw_cfg.get("feature_allowlist", []):
            feature_name = str(item).strip().lower()
            if feature_name and feature_name not in allowlist:
                allowlist.append(feature_name)
    return {
        "enabled": bool(raw_cfg.get("enabled", False)),
        "feature_allowlist": allowlist,
    }


def _resolve_two_stage_stage2_finite_sanitation_config(two_stage_cfg: dict[str, Any]) -> dict[str, Any]:
    stage2_cfg = two_stage_cfg.get("stage2", {}) if isinstance(two_stage_cfg.get("stage2", {}), dict) else {}
    raw_cfg = (
        stage2_cfg.get("finite_sanitation", {})
        if isinstance(stage2_cfg.get("finite_sanitation", {}), dict)
        else {}
    )
    return {
        "enabled": bool(raw_cfg.get("enabled", False)),
        "replace_inf": bool(raw_cfg.get("replace_inf", True)),
        "impute_missing": bool(raw_cfg.get("impute_missing", True)),
        "fail_if_non_finite_after_impute": bool(raw_cfg.get("fail_if_non_finite_after_impute", True)),
        "strategy": str(raw_cfg.get("strategy", "median")).strip().lower(),
    }


def _resolve_global_balance_guard_config(exp_cfg: dict[str, Any]) -> dict[str, Any]:
    evaluation_cfg = exp_cfg.get("evaluation", {}) if isinstance(exp_cfg.get("evaluation", {}), dict) else {}
    raw_cfg = (
        evaluation_cfg.get("global_balance_guard", {})
        if isinstance(evaluation_cfg.get("global_balance_guard", {}), dict)
        else {}
    )
    return {
        "enabled": bool(raw_cfg.get("enabled", False)),
        "reference_source": str(raw_cfg.get("reference_source", "baseline_stage2")).strip().lower(),
        "max_graduate_f1_drop": None if raw_cfg.get("max_graduate_f1_drop") is None else float(raw_cfg.get("max_graduate_f1_drop")),
        "min_macro_f1": None if raw_cfg.get("min_macro_f1") is None else float(raw_cfg.get("min_macro_f1")),
        "min_graduate_f1": None if raw_cfg.get("min_graduate_f1") is None else float(raw_cfg.get("min_graduate_f1")),
        "penalty_weight": float(raw_cfg.get("penalty_weight", 0.5)),
        "fallback_to_plain_macro_f1_if_no_candidate_passes": bool(
            raw_cfg.get("fallback_to_plain_macro_f1_if_no_candidate_passes", True)
        ),
    }


def _resolve_per_model_trial_budgets(
    tuning_cfg: dict[str, Any],
    model_candidates: list[str],
) -> dict[str, int]:
    if not isinstance(tuning_cfg, dict):
        return {}
    raw_overrides = (
        tuning_cfg.get("per_model_n_trials")
        or tuning_cfg.get("per_model_optuna")
        or tuning_cfg.get("tuning_overrides")
        or {}
    )
    if not isinstance(raw_overrides, dict):
        return {}

    resolved: dict[str, int] = {}
    candidate_set = set(model_candidates)
    for raw_model_name, raw_value in raw_overrides.items():
        model_name = str(raw_model_name).strip().lower()
        if model_name not in candidate_set:
            continue
        try:
            trials = int(raw_value)
        except (TypeError, ValueError):
            continue
        if trials > 0:
            resolved[model_name] = trials
    return resolved


def _resolve_model_decision_rule_config(
    exp_cfg: dict[str, Any],
    base_decision_rule_cfg: dict[str, Any],
    model_name: str,
    formulation: str,
    two_stage_enabled: bool,
    class_metadata: dict[str, Any],
) -> dict[str, Any]:
    inference_cfg = exp_cfg.get("inference", {}) if isinstance(exp_cfg.get("inference", {}), dict) else {}
    multiclass_cfg = (
        inference_cfg.get("multiclass_decision", {})
        if isinstance(inference_cfg.get("multiclass_decision", {}), dict)
        else {}
    )
    per_model_cfg = multiclass_cfg.get("per_model", {}) if isinstance(multiclass_cfg.get("per_model", {}), dict) else {}
    raw_override = per_model_cfg.get(model_name, {})
    if not isinstance(raw_override, dict) or not raw_override:
        return copy.deepcopy(base_decision_rule_cfg)

    override_enabled = raw_override.get("enabled")
    if override_enabled is False:
        temp_cfg = copy.deepcopy(exp_cfg)
        temp_cfg.setdefault("evaluation", {})
        temp_cfg["evaluation"]["decision_rule"] = "argmax"
        temp_cfg.setdefault("inference", {})
        temp_cfg["inference"]["multiclass_decision"] = {"strategy": "argmax"}
        return _resolve_decision_rule_config(
            exp_cfg=temp_cfg,
            formulation=formulation,
            two_stage_enabled=two_stage_enabled,
            class_metadata=class_metadata,
        )

    model_override = {k: v for k, v in raw_override.items() if k != "enabled"}
    if not model_override:
        return copy.deepcopy(base_decision_rule_cfg)

    temp_cfg = copy.deepcopy(exp_cfg)
    temp_cfg.setdefault("evaluation", {})
    temp_cfg.setdefault("inference", {})
    base_multiclass_cfg = copy.deepcopy(multiclass_cfg)
    base_multiclass_cfg.pop("per_model", None)
    merged_multiclass_cfg = _deep_merge_dicts(base_multiclass_cfg, model_override)
    temp_cfg["inference"]["multiclass_decision"] = merged_multiclass_cfg
    if "decision_rule" in model_override:
        temp_cfg["evaluation"]["decision_rule"] = str(model_override.get("decision_rule", "argmax"))
    elif "strategy" in merged_multiclass_cfg:
        temp_cfg["evaluation"]["decision_rule"] = str(merged_multiclass_cfg.get("strategy", "argmax"))

    return _resolve_decision_rule_config(
        exp_cfg=temp_cfg,
        formulation=formulation,
        two_stage_enabled=two_stage_enabled,
        class_metadata=class_metadata,
    )
