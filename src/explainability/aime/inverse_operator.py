"""AIME inverse-operator approximation for tabular classifiers.

Assumption:
- The model-probability response can be approximated locally by a linear mapping
  from standardized features to class probabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        if scores.ndim == 1:
            probs_pos = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1.0 - probs_pos, probs_pos])
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
    raise ValueError("Model must expose predict_proba or decision_function for AIME.")


def compute_inverse_operator(model: Any, X_reference: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Estimate the AIME inverse operator from standardized X and model probabilities."""
    if X_reference.empty:
        raise ValueError("X_reference is empty; cannot compute AIME inverse operator.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reference)
    probs = _predict_proba(model, X_reference)

    X_mean = X_scaled.mean(axis=0, keepdims=True)
    P_mean = probs.mean(axis=0, keepdims=True)
    X_centered = X_scaled - X_mean
    P_centered = probs - P_mean

    # Linear operator A that maps feature deviations to probability deviations.
    # P_centered ~= X_centered @ A
    operator = np.linalg.pinv(X_centered) @ P_centered

    return {
        "feature_names": X_reference.columns.tolist(),
        "x_scaled": X_scaled,
        "x_centered": X_centered,
        "x_mean": X_mean.ravel(),
        "probs": probs,
        "probs_mean": P_mean.ravel(),
        "operator": operator,
        "scaler": scaler,
        "classes": list(getattr(model, "classes_", list(range(probs.shape[1])))),
        "assumptions": [
            "Local linear approximation in standardized feature space.",
            "Inverse operator estimated via pseudo-inverse (least-squares).",
        ],
        "config": config,
    }
