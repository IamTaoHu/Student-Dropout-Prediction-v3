"""Optuna-oriented search space definitions per model family."""

from __future__ import annotations

from typing import Any


def get_search_space(model_name: str, search_backend: str = "optuna") -> dict[str, Any]:
    """Return declarative search-space metadata for a model."""
    if search_backend != "optuna":
        raise ValueError(f"Unsupported search backend: {search_backend}")

    spaces: dict[str, dict[str, Any]] = {
        "decision_tree": {
            "max_depth": {"type": "int", "low": 2, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "criterion": {"type": "categorical", "choices": ["gini", "entropy"]},
        },
        "random_forest": {
            "n_estimators": {"type": "int", "low": 100, "high": 500},
            "max_depth": {"type": "int", "low": 4, "high": 30},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
            "bootstrap": {"type": "categorical", "choices": [True, False]},
        },
        "svm": {
            "C": {"type": "float", "low": 1e-3, "high": 1e3, "log": True},
            "kernel": {"type": "categorical", "choices": ["linear", "rbf", "poly"]},
            "gamma": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
            "degree": {"type": "int", "low": 2, "high": 5, "conditional_on": {"kernel": "poly"}},
            "class_weight": {"type": "categorical", "choices": [None, "balanced"]},
        },
        "gradient_boosting": {
            "n_estimators": {"type": "int", "low": 100, "high": 400},
            "max_depth": {"type": "int", "low": 2, "high": 8},
            "learning_rate": {"type": "float", "low": 1e-2, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "min_samples_split": {"type": "int", "low": 2, "high": 10},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 5},
        },
        "xgboost": {
            "n_estimators": {"type": "int", "low": 100, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 1e-2, "high": 0.3, "log": True},
            "min_child_weight": {"type": "float", "low": 1.0, "high": 10.0},
            "subsample": {"type": "float", "low": 0.7, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.7, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-4, "high": 5.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-3, "high": 20.0, "log": True},
        },
        "lightgbm": {
            "n_estimators": {"type": "int", "low": 100, "high": 500},
            "max_depth": {"type": "int", "low": -1, "high": 16},
            "learning_rate": {"type": "float", "low": 1e-2, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 15, "high": 120},
            "min_child_samples": {"type": "int", "low": 5, "high": 80},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        },
        "catboost": {
            "iterations": {"type": "int", "low": 150, "high": 700},
            "depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 1e-2, "high": 0.3, "log": True},
            "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 12.0},
            "random_strength": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "bagging_temperature": {"type": "float", "low": 0.0, "high": 5.0},
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
) -> dict[str, Any]:
    """Suggest model-specific parameters using an Optuna trial object."""
    if model_name == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "random_state": random_state,
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_jobs": -1,
            "random_state": random_state,
        }
    if model_name == "svm":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        params: dict[str, Any] = {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "kernel": kernel,
            "gamma": trial.suggest_float("gamma", 1e-4, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "probability": True,
            "random_state": random_state,
        }
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
        return params
    if model_name == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state": random_state,
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "objective": "multi:softprob" if n_classes > 2 else "binary:logistic",
            "num_class": n_classes if n_classes > 2 else None,
            "random_state": random_state,
            "n_jobs": -1,
        }
    if model_name == "lightgbm":
        objective = "multiclass" if n_classes > 2 else "binary"
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 120),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": objective,
            "num_class": n_classes if n_classes > 2 else None,
            "random_state": random_state,
        }
    if model_name == "catboost":
        objective = "MultiClass" if n_classes > 2 else "Logloss"
        return {
            "iterations": trial.suggest_int("iterations", 150, 700),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 12.0),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "loss_function": objective,
            "random_seed": random_state,
            "verbose": 0,
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
    }
    if model_name not in defaults:
        raise KeyError(f"No defaults available for model '{model_name}'.")
    return {k: v for k, v in defaults[model_name].items() if v is not None}
