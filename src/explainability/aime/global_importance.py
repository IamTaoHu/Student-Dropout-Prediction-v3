"""AIME global feature importance from inverse-operator coefficients."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_aime_global_importance(aime_state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Return global feature rankings based on operator magnitude."""
    feature_names = aime_state["feature_names"]
    operator = np.asarray(aime_state["operator"], dtype=float)
    classes = aime_state.get("classes", [])

    score = np.linalg.norm(operator, ord=2, axis=1)
    per_class_abs = np.abs(operator)

    global_df = pd.DataFrame({"feature": feature_names, "importance": score}).sort_values(
        "importance", ascending=False
    )
    per_class_df = pd.DataFrame(per_class_abs, columns=[f"class_{c}" for c in classes])
    per_class_df.insert(0, "feature", feature_names)

    top_k = int(config.get("top_k", 20))
    return {
        "global_importance": global_df.head(top_k).reset_index(drop=True),
        "per_class_importance": per_class_df,
    }
