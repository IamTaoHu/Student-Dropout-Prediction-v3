"""AIME local importance using class-conditional linear contributions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_aime_local_importance(
    aime_state: dict[str, Any],
    X_instances: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute local feature contributions for selected instances."""
    if X_instances.empty:
        return {"local_importance": pd.DataFrame()}

    feature_names = aime_state["feature_names"]
    operator = np.asarray(aime_state["operator"], dtype=float)
    scaler = aime_state["scaler"]
    probs_mean = np.asarray(aime_state["probs_mean"], dtype=float)
    classes = np.asarray(aime_state["classes"])
    suppress_zero_onehot = bool(config.get("suppress_zero_onehot", True))
    top_k = int(config.get("top_k", 15))

    X_scaled = scaler.transform(X_instances[feature_names])
    records: list[dict[str, Any]] = []
    for row_idx, (index_label, row_scaled) in enumerate(zip(X_instances.index.tolist(), X_scaled)):
        row_centered = row_scaled - np.asarray(aime_state["x_mean"], dtype=float)
        logits_proxy = probs_mean + row_centered @ operator
        pred_class_idx = int(np.argmax(logits_proxy))
        pred_class = classes[pred_class_idx] if classes.size else pred_class_idx
        contributions = row_centered * operator[:, pred_class_idx]

        if suppress_zero_onehot:
            row_raw = X_instances.loc[index_label, feature_names]
            for j, fname in enumerate(feature_names):
                if "_" in fname and float(row_raw.iloc[j]) == 0.0:
                    contributions[j] = 0.0

        ranked = np.argsort(np.abs(contributions))[::-1][:top_k]
        for rank, feat_idx in enumerate(ranked, start=1):
            records.append(
                {
                    "instance_index": int(row_idx),
                    "instance_label": str(index_label),
                    "predicted_class": str(pred_class),
                    "rank": rank,
                    "feature": feature_names[feat_idx],
                    "contribution": float(contributions[feat_idx]),
                    "abs_contribution": float(abs(contributions[feat_idx])),
                }
            )

    local_df = pd.DataFrame.from_records(records)
    return {"local_importance": local_df}
