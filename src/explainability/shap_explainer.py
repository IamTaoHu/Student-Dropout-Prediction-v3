"""SHAP explainability utilities for tabular benchmark experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


def _to_dataframe(x: pd.DataFrame | np.ndarray, feature_names: list[str] | None = None) -> pd.DataFrame:
    """Ensure feature matrix is a DataFrame with stable feature names."""
    if isinstance(x, pd.DataFrame):
        return x.copy()

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(x.shape[1])]

    return pd.DataFrame(x, columns=feature_names)


def _predict_proba_df(model: Any, feature_names: list[str]):
    """Wrap predict_proba so SHAP always receives DataFrame inputs with feature names."""

    def _fn(x: np.ndarray) -> np.ndarray:
        x_df = pd.DataFrame(x, columns=feature_names)
        return model.predict_proba(x_df)

    return _fn


def _build_shap_explainer(model: Any, X_bg: pd.DataFrame):
    """
    Try fast/model-specific SHAP explainer first.
    Fall back to model-agnostic SHAP for unsupported models such as
    multiclass sklearn GradientBoostingClassifier.
    """
    feature_names = X_bg.columns.tolist()

    # First try TreeExplainer for tree models
    try:
        return shap.TreeExplainer(model), "tree"
    except Exception:
        pass

    # Then try generic Explainer on predict_proba
    try:
        masker = shap.maskers.Independent(X_bg)
        return shap.Explainer(_predict_proba_df(model, feature_names), masker), "generic"
    except Exception:
        pass

    # Last fallback: KernelExplainer on a small background sample
    bg_small = X_bg.head(min(50, len(X_bg))).copy()
    return shap.KernelExplainer(_predict_proba_df(model, feature_names), bg_small), "kernel"


def _normalize_shap_values(shap_values: Any, X_explain: pd.DataFrame) -> tuple[list[np.ndarray], int]:
    """
    Normalize SHAP outputs into a list of per-class arrays with shape:
    [n_classes][n_samples, n_features]
    """
    n_samples, n_features = X_explain.shape

    # shap.Explanation object
    if hasattr(shap_values, "values"):
        values = shap_values.values
    else:
        values = shap_values

    # Case 1: list of arrays, one per class
    if isinstance(values, list):
        arrays = [np.asarray(v) for v in values]
        return arrays, len(arrays)

    values = np.asarray(values)

    # Case 2: binary or single-output: [n_samples, n_features]
    if values.ndim == 2 and values.shape == (n_samples, n_features):
        return [values], 1

    # Case 3: multiclass Explanation: [n_samples, n_features, n_classes]
    if values.ndim == 3 and values.shape[0] == n_samples and values.shape[1] == n_features:
        arrays = [values[:, :, class_idx] for class_idx in range(values.shape[2])]
        return arrays, values.shape[2]

    # Case 4: sometimes [n_classes, n_samples, n_features]
    if values.ndim == 3 and values.shape[1] == n_samples and values.shape[2] == n_features:
        arrays = [values[class_idx, :, :] for class_idx in range(values.shape[0])]
        return arrays, values.shape[0]

    raise ValueError(
        f"Unsupported SHAP value shape: {values.shape}. "
        f"Expected 2D or 3D output aligned to X_explain={X_explain.shape}."
    )


def run_shap_explanations(
    model: Any,
    X_background: pd.DataFrame | np.ndarray,
    X_explain: pd.DataFrame | np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run SHAP explanations with safe fallback behavior.

    Supports:
    - tree models via TreeExplainer
    - unsupported multiclass models via generic/Kernel SHAP fallback
    """
    if shap is None:
        return {"status": "skipped", "reason": "shap is not installed"}

    config = config or {}

    if isinstance(X_background, pd.DataFrame):
        feature_names = X_background.columns.tolist()
    elif isinstance(X_explain, pd.DataFrame):
        feature_names = X_explain.columns.tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(X_background.shape[1])]

    X_bg = _to_dataframe(X_background, feature_names)
    X_exp = _to_dataframe(X_explain, feature_names)

    max_background = int(config.get("max_background", min(100, len(X_bg))))
    max_explain = int(config.get("max_explain", min(20, len(X_exp))))

    X_bg = X_bg.head(max_background).copy()
    X_exp = X_exp.head(max_explain).copy()

    try:
        explainer, explainer_type = _build_shap_explainer(model, X_bg)
    except Exception as e:
        return {
            "status": "skipped",
            "reason": f"Unable to construct SHAP explainer: {type(e).__name__}: {e}",
        }

    try:
        shap_values = explainer(X_exp)
    except Exception:
        try:
            shap_values = explainer.shap_values(X_exp)
        except Exception as e:
            return {
                "status": "skipped",
                "reason": f"Unable to compute SHAP values: {type(e).__name__}: {e}",
            }

    try:
        class_arrays, n_outputs = _normalize_shap_values(shap_values, X_exp)
    except Exception as e:
        return {
            "status": "skipped",
            "reason": f"Unable to normalize SHAP values: {type(e).__name__}: {e}",
        }

    global_importance = []
    for class_idx, arr in enumerate(class_arrays):
        mean_abs = np.mean(np.abs(arr), axis=0)
        top_idx = np.argsort(mean_abs)[::-1][: min(20, len(feature_names))]
        global_importance.append(
            {
                "class_index": int(class_idx),
                "top_features": [
                    {
                        "feature": feature_names[i],
                        "mean_abs_shap": float(mean_abs[i]),
                    }
                    for i in top_idx
                ],
            }
        )

    local_explanations = []
    preds = model.predict(X_exp)

    for row_idx in range(len(X_exp)):
        predicted_label = int(preds[row_idx]) if np.isscalar(preds[row_idx]) or isinstance(preds[row_idx], (np.integer, int)) else 0

        # choose class-specific SHAP row when possible
        if n_outputs == 1:
            row_values = class_arrays[0][row_idx]
            explained_class = 0
        else:
            class_idx = predicted_label if 0 <= predicted_label < n_outputs else 0
            row_values = class_arrays[class_idx][row_idx]
            explained_class = class_idx

        top_idx = np.argsort(np.abs(row_values))[::-1][: min(10, len(feature_names))]
        local_explanations.append(
            {
                "instance_index": int(row_idx),
                "predicted_label": int(predicted_label),
                "explained_class": int(explained_class),
                "top_features": [
                    {
                        "feature": feature_names[i],
                        "shap_value": float(row_values[i]),
                    }
                    for i in top_idx
                ],
            }
        )

    return {
        "status": "ok",
        "method": "shap",
        "explainer_type": explainer_type,
        "num_instances": len(local_explanations),
        "global_importance": global_importance,
        "local_explanations": local_explanations,
    }