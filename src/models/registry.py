"""Model registry and factories for benchmark-compatible classifiers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ModelFactory = Callable[[dict[str, Any]], Any]


def _decision_tree_factory(params: dict[str, Any]) -> Any:
    return DecisionTreeClassifier(**params)


def _random_forest_factory(params: dict[str, Any]) -> Any:
    return RandomForestClassifier(**params)


def _svm_factory(params: dict[str, Any]) -> Any:
    payload = {"probability": True, **params}
    return SVC(**payload)


def _gradient_boosting_factory(params: dict[str, Any]) -> Any:
    return GradientBoostingClassifier(**params)


def _xgboost_factory(params: dict[str, Any]) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Install it to use the xgboost model.") from exc
    payload = {
        "eval_metric": "mlogloss",
        **params,
    }
    return XGBClassifier(**payload)


def _lightgbm_factory(params: dict[str, Any]) -> Any:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("lightgbm is not installed. Install it to use the lightgbm model.") from exc
    return LGBMClassifier(**params)


def _catboost_factory(params: dict[str, Any]) -> Any:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("catboost is not installed. Install it to use the catboost model.") from exc
    payload = {"verbose": 0, **params}
    return CatBoostClassifier(**payload)


def _mlp_factory(params: dict[str, Any]) -> Any:
    return MLPClassifier(**params)


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "decision_tree": _decision_tree_factory,
    "random_forest": _random_forest_factory,
    "svm": _svm_factory,
    "gradient_boosting": _gradient_boosting_factory,
    "xgboost": _xgboost_factory,
    "lightgbm": _lightgbm_factory,
    "catboost": _catboost_factory,
    "mlp": _mlp_factory,
}


def list_available_models() -> list[str]:
    """Return registered model names. Availability is checked at build time."""
    return sorted(MODEL_REGISTRY.keys())


def build_model(model_name: str, params: dict[str, Any]) -> Any:
    """Construct a model instance from the registry."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Available: {list_available_models()}")
    return MODEL_REGISTRY[model_name](params)
