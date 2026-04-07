"""Optuna-oriented search space definitions per model family."""

from __future__ import annotations

from typing import Any


DEFAULT_SVM_SEARCH_SPACE: dict[str, Any] = {
    "kernel_choices": ["rbf", "linear", "poly", "sigmoid"],
    "kernel_params": {
        "default": {
            "C": {"low": 1e-4, "high": 1e4, "log": True},
            "gamma": {"low": 1e-7, "high": 1e1, "log": True},
            "degree": {"low": 2, "high": 5},
            "coef0": {"low": -1.0, "high": 1.0},
        }
    },
    "shrinking_choices": [True, False],
    "tol": {"low": 1e-5, "high": 1e-2, "log": True},
    "class_weight_choices": [None, "balanced"],
}


def _resolve_svm_search_space(search_space_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    overrides = search_space_overrides if isinstance(search_space_overrides, dict) else {}
    svm_overrides = overrides.get("svm", {}) if isinstance(overrides.get("svm", {}), dict) else {}

    kernel_choices = svm_overrides.get("kernel_choices", DEFAULT_SVM_SEARCH_SPACE["kernel_choices"])
    if not isinstance(kernel_choices, list) or not kernel_choices:
        kernel_choices = list(DEFAULT_SVM_SEARCH_SPACE["kernel_choices"])

    kernel_params_raw = svm_overrides.get("kernel_params", {})
    kernel_params = kernel_params_raw if isinstance(kernel_params_raw, dict) else {}
    default_kernel_params = DEFAULT_SVM_SEARCH_SPACE["kernel_params"]["default"]

    shrinking_choices = svm_overrides.get("shrinking_choices", DEFAULT_SVM_SEARCH_SPACE["shrinking_choices"])
    if not isinstance(shrinking_choices, list) or not shrinking_choices:
        shrinking_choices = list(DEFAULT_SVM_SEARCH_SPACE["shrinking_choices"])

    tol = svm_overrides.get("tol", DEFAULT_SVM_SEARCH_SPACE["tol"])
    if not isinstance(tol, dict):
        tol = dict(DEFAULT_SVM_SEARCH_SPACE["tol"])

    class_weight_choices = svm_overrides.get(
        "class_weight_choices",
        DEFAULT_SVM_SEARCH_SPACE["class_weight_choices"],
    )
    if not isinstance(class_weight_choices, list) or not class_weight_choices:
        class_weight_choices = list(DEFAULT_SVM_SEARCH_SPACE["class_weight_choices"])

    return {
        "kernel_choices": list(kernel_choices),
        "kernel_params": kernel_params,
        "default_kernel_params": default_kernel_params,
        "shrinking_choices": list(shrinking_choices),
        "tol": {
            "low": float(tol.get("low", DEFAULT_SVM_SEARCH_SPACE["tol"]["low"])),
            "high": float(tol.get("high", DEFAULT_SVM_SEARCH_SPACE["tol"]["high"])),
            "log": bool(tol.get("log", DEFAULT_SVM_SEARCH_SPACE["tol"]["log"])),
        },
        "class_weight_choices": list(class_weight_choices),
    }


def _resolve_numeric_param_spec(
    space: dict[str, Any],
    kernel: str,
    param_name: str,
) -> dict[str, Any] | None:
    kernel_params = space.get("kernel_params", {}) if isinstance(space.get("kernel_params", {}), dict) else {}
    kernel_specific = kernel_params.get(kernel, {}) if isinstance(kernel_params.get(kernel, {}), dict) else {}
    if param_name in kernel_specific:
        spec = kernel_specific.get(param_name)
        return spec if isinstance(spec, dict) else None

    default_params = (
        space.get("default_kernel_params", {})
        if isinstance(space.get("default_kernel_params", {}), dict)
        else {}
    )
    spec = default_params.get(param_name)
    return spec if isinstance(spec, dict) else None


def get_search_space(model_name: str, search_backend: str = "optuna") -> dict[str, Any]:
    """Return declarative search-space metadata for a model."""
    if search_backend != "optuna":
        raise ValueError(f"Unsupported search backend: {search_backend}")

    spaces: dict[str, dict[str, Any]] = {
        "decision_tree": {
            "max_depth": {"type": "int", "low": 2, "high": 40},
            "min_samples_split": {"type": "int", "low": 2, "high": 40},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
            "criterion": {"type": "categorical", "choices": ["gini", "entropy"]},
        },
        "random_forest": {
            "n_estimators": {"type": "int", "low": 100, "high": 1200},
            "max_depth": {"type": "int", "low": 3, "high": 40},
            "min_samples_split": {"type": "int", "low": 2, "high": 40},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
            "bootstrap": {"type": "categorical", "choices": [True, False]},
        },
        "svm": {
            "C": {"type": "float", "low": 1e-4, "high": 1e4, "log": True},
            "kernel": {"type": "categorical", "choices": list(DEFAULT_SVM_SEARCH_SPACE["kernel_choices"])},
            "gamma": {"type": "float", "low": 1e-7, "high": 1e1, "log": True},
            "degree": {"type": "int", "low": 2, "high": 5, "conditional_on": {"kernel": "poly"}},
            "coef0": {"type": "float", "low": -1.0, "high": 1.0, "conditional_on": {"kernel": ["poly", "sigmoid"]}},
            "shrinking": {"type": "categorical", "choices": list(DEFAULT_SVM_SEARCH_SPACE["shrinking_choices"])},
            "tol": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "class_weight": {"type": "categorical", "choices": list(DEFAULT_SVM_SEARCH_SPACE["class_weight_choices"])},
        },
        "gradient_boosting": {
            "n_estimators": {"type": "int", "low": 100, "high": 1000},
            "max_depth": {"type": "int", "low": 2, "high": 12},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.5, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "min_samples_split": {"type": "int", "low": 2, "high": 30},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 15},
        },
        "xgboost": {
            "n_estimators": {"type": "int", "low": 100, "high": 1500},
            "max_depth": {"type": "int", "low": 2, "high": 14},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.5, "log": True},
            "min_child_weight": {"type": "float", "low": 0.1, "high": 20.0, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-8, "high": 20.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-5, "high": 50.0, "log": True},
        },
        "lightgbm": {
            "n_estimators": {"type": "int", "low": 100, "high": 1500},
            "max_depth": {"type": "int", "low": -1, "high": 32},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.5, "log": True},
            "num_leaves": {"type": "int", "low": 8, "high": 256},
            "min_child_samples": {"type": "int", "low": 5, "high": 200},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-8, "high": 20.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 50.0, "log": True},
        },
        "catboost": {
            "iterations": {"type": "int", "low": 150, "high": 1500},
            "depth": {"type": "int", "low": 3, "high": 12},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.5, "log": True},
            "l2_leaf_reg": {"type": "float", "low": 1e-2, "high": 30.0, "log": True},
            "random_strength": {"type": "float", "low": 1e-8, "high": 20.0, "log": True},
            "bagging_temperature": {"type": "float", "low": 0.0, "high": 10.0},
        },
        "mlp": {
            "hidden_layer_sizes": {"type": "categorical", "choices": [(64,), (128,), (256,), (128, 64), (64, 32)]},
            "alpha": {"type": "float", "low": 1e-7, "high": 1e0, "log": True},
            "learning_rate_init": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
        },
    }
    if model_name not in spaces:
        raise KeyError(f"No search space defined for model '{model_name}'.")
    return {"model": model_name, "space": spaces[model_name]}


def suggest_optuna_params(
    trial: Any,
    model_name: str,
    n_classes: int,
    random_state: int,
    search_space_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Suggest model-specific parameters using an Optuna trial object."""
    if model_name == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "random_state": random_state,
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_jobs": -1,
            "random_state": random_state,
        }
    if model_name == "svm":
        svm_space = _resolve_svm_search_space(search_space_overrides)
        kernel = trial.suggest_categorical("kernel", svm_space["kernel_choices"])
        c_spec = _resolve_numeric_param_spec(svm_space, kernel, "C") or {"low": 1e-4, "high": 1e4, "log": True}
        params: dict[str, Any] = {
            "C": trial.suggest_float("C", float(c_spec["low"]), float(c_spec["high"]), log=bool(c_spec.get("log", False))),
            "kernel": kernel,
            "shrinking": trial.suggest_categorical("shrinking", svm_space["shrinking_choices"]),
            "tol": trial.suggest_float(
                "tol",
                float(svm_space["tol"]["low"]),
                float(svm_space["tol"]["high"]),
                log=bool(svm_space["tol"].get("log", False)),
            ),
            "class_weight": trial.suggest_categorical("class_weight", svm_space["class_weight_choices"]),
            "probability": True,
            "random_state": random_state,
        }
        gamma_spec = _resolve_numeric_param_spec(svm_space, kernel, "gamma")
        if gamma_spec is not None and kernel != "linear":
            params["gamma"] = trial.suggest_float(
                "gamma",
                float(gamma_spec["low"]),
                float(gamma_spec["high"]),
                log=bool(gamma_spec.get("log", False)),
            )
        if kernel == "poly":
            degree_spec = _resolve_numeric_param_spec(svm_space, kernel, "degree")
            if degree_spec is not None:
                params["degree"] = trial.suggest_int("degree", int(degree_spec["low"]), int(degree_spec["high"]))
        if kernel in {"poly", "sigmoid"}:
            coef0_spec = _resolve_numeric_param_spec(svm_space, kernel, "coef0")
            if coef0_spec is not None:
                params["coef0"] = trial.suggest_float("coef0", float(coef0_spec["low"]), float(coef0_spec["high"]))
        return params
    if model_name == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "random_state": random_state,
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": trial.suggest_int("max_depth", 2, 14),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 50.0, log=True),
            "objective": "multi:softprob" if n_classes > 2 else "binary:logistic",
            "num_class": n_classes if n_classes > 2 else None,
            "random_state": random_state,
            "n_jobs": -1,
        }
    if model_name == "lightgbm":
        objective = "multiclass" if n_classes > 2 else "binary"
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": trial.suggest_int("max_depth", -1, 32),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 50.0, log=True),
            "objective": objective,
            "num_class": n_classes if n_classes > 2 else None,
            "random_state": random_state,
        }
    if model_name == "catboost":
        objective = "MultiClass" if n_classes > 2 else "Logloss"
        return {
            "iterations": trial.suggest_int("iterations", 150, 1500),
            "depth": trial.suggest_int("depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 20.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "loss_function": objective,
            "random_seed": random_state,
            "verbose": 0,
        }
    if model_name == "mlp":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes",
                [(64,), (128,), (256,), (128, 64), (64, 32)],
            ),
            "alpha": trial.suggest_float("alpha", 1e-7, 1e0, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-1, log=True),
            "max_iter": 300,
            "early_stopping": True,
            "random_state": random_state,
        }
    raise KeyError(f"Unsupported model for optuna parameter suggestion: '{model_name}'")


def get_default_model_params(model_name: str, random_state: int, n_classes: int) -> dict[str, Any]:
    """Return conservative baseline parameters per model family."""
    defaults: dict[str, dict[str, Any]] = {
        "decision_tree": {"random_state": random_state},
        "random_forest": {"n_estimators": 200, "random_state": random_state, "n_jobs": -1},
        "svm": {"probability": True, "random_state": random_state},
        "gradient_boosting": {"random_state": random_state},
        "xgboost": {
            "n_estimators": 300,
            "objective": "multi:softprob" if n_classes > 2 else "binary:logistic",
            "num_class": n_classes if n_classes > 2 else None,
            "random_state": random_state,
            "n_jobs": -1,
        },
        "lightgbm": {
            "n_estimators": 300,
            "objective": "multiclass" if n_classes > 2 else "binary",
            "num_class": n_classes if n_classes > 2 else None,
            "random_state": random_state,
        },
        "catboost": {
            "iterations": 300,
            "loss_function": "MultiClass" if n_classes > 2 else "Logloss",
            "random_seed": random_state,
            "verbose": 0,
        },
        "mlp": {
            "hidden_layer_sizes": (128,),
            "activation": "relu",
            "solver": "adam",
            "learning_rate_init": 1e-3,
            "alpha": 1e-4,
            "max_iter": 300,
            "early_stopping": True,
            "random_state": random_state,
        },
    }
    if model_name not in defaults:
        raise KeyError(f"No defaults available for model '{model_name}'.")
    return {k: v for k, v in defaults[model_name].items() if v is not None}
