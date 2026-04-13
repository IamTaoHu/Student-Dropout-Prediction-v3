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
        print("[preprocessing][smote] enabled=false method=disabled")
        return X_train, y_train, {"enabled": False, "method": None}

    try:
        from imblearn.over_sampling import SMOTE, SMOTENC
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for SMOTE balancing. "
            "Install it before running experiments with preprocessing.balancing.enabled=true."
        ) from exc

    random_state = int(config.get("random_state", 42))
    requested_k_neighbors = int(config.get("k_neighbors", 5))
    before = y_train.value_counts(dropna=False).sort_index().to_dict()
    before_serialized = {str(k): int(v) for k, v in before.items()}
    min_class_count = int(min(before.values())) if before else 0
    allow_skip_on_failure = bool(config.get("allow_skip_on_failure", False))
    effective_k_neighbors = requested_k_neighbors
    if min_class_count > 0:
        effective_k_neighbors = min(requested_k_neighbors, max(1, min_class_count - 1))
    if min_class_count < 2:
        if allow_skip_on_failure:
            print(
                "[preprocessing][smote][skip] "
                f"reason=minority_count_lt_2 class_counts_before={before_serialized}"
            )
            return X_train, y_train, {
                "enabled": True,
                "method": method,
                "skipped": True,
                "skip_reason": "minority_count_lt_2",
                "random_state": random_state,
                "requested_k_neighbors": requested_k_neighbors,
                "k_neighbors": None,
                "class_counts_before_smote": before_serialized,
                "class_counts_after_smote": before_serialized,
                "class_distribution_before": before_serialized,
                "class_distribution_after": before_serialized,
            }
        raise ValueError(
            "SMOTE requires at least 2 samples in every class of the training data. "
            f"Observed class counts: {before_serialized}"
        )
    if effective_k_neighbors != requested_k_neighbors:
        print(
            "[preprocessing][smote][adjust] "
            f"requested_k_neighbors={requested_k_neighbors} "
            f"effective_k_neighbors={effective_k_neighbors} "
            f"min_class_count={min_class_count}"
        )

    if method == "smote":
        sampler = SMOTE(random_state=random_state, k_neighbors=effective_k_neighbors)
    elif method == "smotenc":
        if not categorical_feature_indices:
            raise ValueError("SMOTENC requires categorical_feature_indices.")
        sampler = SMOTENC(
            categorical_features=categorical_feature_indices,
            random_state=random_state,
            k_neighbors=effective_k_neighbors,
        )
    else:
        raise ValueError(f"Unsupported balancing method: '{method}'")

    try:
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    except Exception as exc:
        if allow_skip_on_failure:
            print(
                "[preprocessing][smote][skip] "
                f"reason=fit_resample_failed error={type(exc).__name__}: {exc}"
            )
            return X_train, y_train, {
                "enabled": True,
                "method": method,
                "skipped": True,
                "skip_reason": f"fit_resample_failed:{type(exc).__name__}",
                "random_state": random_state,
                "requested_k_neighbors": requested_k_neighbors,
                "k_neighbors": effective_k_neighbors,
                "class_counts_before_smote": before_serialized,
                "class_counts_after_smote": before_serialized,
                "class_distribution_before": before_serialized,
                "class_distribution_after": before_serialized,
            }
        raise ValueError(
            "Balancing failed during fit_resample. "
            "Ensure train features are fully numeric/encoded and compatible with the selected balancing method."
        ) from exc

    X_out = pd.DataFrame(X_res, columns=X_train.columns)
    y_out = pd.Series(y_res, name=y_train.name)
    after = y_out.value_counts(dropna=False).sort_index().to_dict()
    after_serialized = {str(k): int(v) for k, v in after.items()}
    sampler_parameters = {
        "random_state": random_state,
        "k_neighbors": effective_k_neighbors,
    }
    if method == "smotenc":
        sampler_parameters["categorical_feature_indices"] = list(categorical_feature_indices or [])
    print(
        "[preprocessing][smote] "
        f"enabled=true method={method} "
        f"class_counts_before={before_serialized} "
        f"class_counts_after={after_serialized}"
    )

    metadata = {
        "enabled": True,
        "method": method,
        "skipped": False,
        "random_state": random_state,
        "requested_k_neighbors": requested_k_neighbors,
        "k_neighbors": effective_k_neighbors,
        "sampler_parameters": sampler_parameters,
        "class_counts_before_smote": before_serialized,
        "class_counts_after_smote": after_serialized,
        "class_distribution_before": before_serialized,
        "class_distribution_after": after_serialized,
    }
    return X_out, y_out, metadata
