"""Unified orchestration for SHAP, LIME, and AIME explainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.explainability.aime.global_importance import compute_aime_global_importance
from src.explainability.aime.inverse_operator import compute_inverse_operator
from src.explainability.aime.local_importance import compute_aime_local_importance
from src.explainability.aime.representative_instance import select_representative_instances
from src.explainability.aime.similarity_plot import save_similarity_plot
from src.explainability.lime_explainer import run_lime_explanations
from src.explainability.shap_explainer import run_shap_explanations


def run_explainability_suite(
    model: Any,
    X_train: pd.DataFrame,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    config: dict[str, Any],
    y_reference: pd.Series | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run configured explainers and return a unified artifact dictionary."""
    out: dict[str, Any] = {}

    if config.get("shap", {}).get("enabled", True):
        out["shap"] = run_shap_explanations(model, X_background, X_explain, config.get("shap", {}))

    if config.get("lime", {}).get("enabled", True):
        out["lime"] = run_lime_explanations(model, X_train, X_explain, config.get("lime", {}))

    if config.get("aime", {}).get("enabled", True):
        aime_cfg = config.get("aime", {})
        aime_state = compute_inverse_operator(model, X_reference=X_background, config=aime_cfg)
        global_info = compute_aime_global_importance(aime_state, aime_cfg)
        local_info = compute_aime_local_importance(aime_state, X_instances=X_explain, config=aime_cfg)

        representative = (
            select_representative_instances(X_background, y_reference, aime_cfg)
            if y_reference is not None
            else pd.DataFrame()
        )
        similarity_path = None
        if output_dir is not None:
            similarity_path = save_similarity_plot(
                aime_state,
                output_path=output_dir / "aime_similarity.png",
                config=aime_cfg,
            )
        out["aime"] = {
            "state_meta": {"assumptions": aime_state.get("assumptions", [])},
            "global_importance": global_info["global_importance"],
            "per_class_importance": global_info["per_class_importance"],
            "local_importance": local_info["local_importance"],
            "representative_instances": representative,
            "similarity_plot_path": None if similarity_path is None else str(similarity_path),
        }

    return out
