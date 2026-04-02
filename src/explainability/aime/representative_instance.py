"""Representative-instance selection for AIME local interpretation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler


def select_representative_instances(X: pd.DataFrame, y: pd.Series, config: dict[str, Any]) -> pd.DataFrame:
    """Select representative instances nearest to class centroids."""
    if X.empty:
        return pd.DataFrame(columns=["index", "class", "distance_to_centroid"])

    per_class = bool(config.get("per_class", True))
    top_n = int(config.get("top_n", 1))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

    rows: list[dict[str, Any]] = []
    if per_class:
        for class_value in sorted(y.unique().tolist()):
            idx = y[y == class_value].index
            if len(idx) == 0:
                continue
            subset = X_scaled_df.loc[idx]
            centroid = subset.mean(axis=0).to_numpy().reshape(1, -1)
            dist = euclidean_distances(subset.to_numpy(), centroid).ravel()
            ranking = np.argsort(dist)[:top_n]
            for r in ranking:
                rows.append(
                    {
                        "index": str(subset.index[r]),
                        "class": str(class_value),
                        "distance_to_centroid": float(dist[r]),
                    }
                )
    else:
        centroid = X_scaled_df.mean(axis=0).to_numpy().reshape(1, -1)
        dist = euclidean_distances(X_scaled_df.to_numpy(), centroid).ravel()
        ranking = np.argsort(dist)[:top_n]
        for r in ranking:
            rows.append(
                {
                    "index": str(X_scaled_df.index[r]),
                    "class": "all",
                    "distance_to_centroid": float(dist[r]),
                }
            )
    return pd.DataFrame(rows)
