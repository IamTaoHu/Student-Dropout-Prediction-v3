"""Train-only class balancing helpers designed for leakage-safe workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd


def apply_balancing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
    categorical_feature_indices: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Apply train-only resampling (SMOTE/SMOTENC) with reproducible settings."""
    enabled = bool(config.get("enabled", False))
    method = str(config.get("method", "smote")).lower()
    if not enabled:
        return X_train, y_train, {"enabled": False, "method": None}

    try:
        from imblearn.over_sampling import SMOTE, SMOTENC
    except ImportError as exc:
        raise ImportError("imbalanced-learn is required for balancing methods.") from exc

    random_state = int(config.get("random_state", 42))
    k_neighbors = int(config.get("k_neighbors", 5))

    if method == "smote":
        sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    elif method == "smotenc":
        if not categorical_feature_indices:
            raise ValueError("SMOTENC requires categorical_feature_indices.")
        sampler = SMOTENC(
            categorical_features=categorical_feature_indices,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )
    else:
        raise ValueError(f"Unsupported balancing method: '{method}'")

    before = y_train.value_counts(dropna=False).sort_index().to_dict()
    try:
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    except Exception as exc:
        raise ValueError(
            "Balancing failed during fit_resample. "
            "Ensure train features are fully numeric/encoded and compatible with the selected balancing method."
        ) from exc

    X_out = pd.DataFrame(X_res, columns=X_train.columns)
    y_out = pd.Series(y_res, name=y_train.name)
    after = y_out.value_counts(dropna=False).sort_index().to_dict()

    metadata = {
        "enabled": True,
        "method": method,
        "random_state": random_state,
        "k_neighbors": k_neighbors,
        "class_distribution_before": {str(k): int(v) for k, v in before.items()},
        "class_distribution_after": {str(k): int(v) for k, v in after.items()},
    }
    return X_out, y_out, metadata
