"""Train-only Isolation Forest outlier filtering utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest


def apply_outlier_filter(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Filter outliers from training data only, returning metadata for auditability."""
    enabled = bool(config.get("enabled", False))
    method = str(config.get("method", "isolation_forest")).lower()
    if not enabled:
        return X_train, y_train, {"enabled": False, "method": None}
    if method != "isolation_forest":
        raise ValueError(f"Unsupported outlier method: '{method}'")

    contamination = float(config.get("contamination", 0.1))
    if not 0 < contamination < 0.5:
        raise ValueError("IsolationForest contamination must be in (0, 0.5).")

    random_state = int(config.get("random_state", 42))
    n_estimators = int(config.get("n_estimators", 200))
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    preds = iso.fit_predict(X_train)
    keep_mask = preds == 1

    X_f = X_train.loc[keep_mask].reset_index(drop=True)
    y_f = y_train.loc[keep_mask].reset_index(drop=True)
    metadata = {
        "enabled": True,
        "method": "isolation_forest",
        "contamination": contamination,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "n_original": int(len(X_train)),
        "n_removed": int((~keep_mask).sum()),
        "n_remaining": int(keep_mask.sum()),
    }
    return X_f, y_f, metadata
