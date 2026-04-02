"""LIME explainability utilities for tabular benchmark experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:  # pragma: no cover
    LimeTabularExplainer = None


def _as_dataframe(x: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Convert ndarray to DataFrame with stable feature names."""
    return pd.DataFrame(x, columns=feature_names)


def run_lime_explanations(
    model: Any,
    X_train: pd.DataFrame | np.ndarray,
    X_explain: pd.DataFrame | np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run tabular LIME explanations on selected rows.

    Robust against:
    - binary / multiclass label mismatches
    - models fitted with feature names but explained with ndarray inputs
    """
    if LimeTabularExplainer is None:
        return {"status": "skipped", "reason": "lime is not installed"}

    config = config or {}

    # Normalize inputs
    if isinstance(X_train, pd.DataFrame):
        train_df = X_train.copy()
        feature_names = train_df.columns.tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        train_df = _as_dataframe(np.asarray(X_train), feature_names)

    if isinstance(X_explain, pd.DataFrame):
        explain_df = X_explain.copy()
        # align order if needed
        explain_df = explain_df[feature_names]
    else:
        explain_df = _as_dataframe(np.asarray(X_explain), feature_names)

    class_names = config.get("class_names")
    if class_names is None and hasattr(model, "classes_"):
        class_names = [str(c) for c in model.classes_]

    num_features = int(config.get("num_features", min(10, len(feature_names))))
    max_instances = int(config.get("max_instances", min(5, len(explain_df))))
    discretize_continuous = bool(config.get("discretize_continuous", True))
    random_state = int(config.get("random_state", 42))

    # Predict wrapper that preserves feature names
    def predict_fn(x: np.ndarray) -> np.ndarray:
        x_df = _as_dataframe(x, feature_names)
        return model.predict_proba(x_df)

    explainer = LimeTabularExplainer(
        training_data=train_df.to_numpy(),
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=discretize_continuous,
        random_state=random_state,
    )

    results: list[dict[str, Any]] = []

    for row_idx in range(min(max_instances, len(explain_df))):
        row_series = explain_df.iloc[row_idx]
        row_array = row_series.to_numpy(dtype=float)

        predicted_label = int(model.predict(row_series.to_frame().T)[0])

        exp = explainer.explain_instance(
            data_row=row_array,
            predict_fn=predict_fn,
            num_features=num_features,
            top_labels=max(2, len(class_names) if class_names else 2),
        )

        available_labels = list(exp.available_labels())

        # Prefer predicted label when present; otherwise use first available label
        if predicted_label in available_labels:
            label_to_use = predicted_label
        elif available_labels:
            label_to_use = int(available_labels[0])
        else:
            # Extremely defensive fallback
            label_to_use = predicted_label

        explanation_rows = []
        for rank, (feature_term, weight) in enumerate(exp.as_list(label=label_to_use), start=1):
            explanation_rows.append(
                {
                    "rank": rank,
                    "feature_term": feature_term,
                    "weight": float(weight),
                }
            )

        results.append(
            {
                "instance_index": int(row_idx),
                "predicted_label": int(predicted_label),
                "explained_label": int(label_to_use),
                "available_labels": [int(x) for x in available_labels],
                "explanation": explanation_rows,
            }
        )

    return {
        "status": "ok",
        "method": "lime",
        "num_instances": len(results),
        "results": results,
    }