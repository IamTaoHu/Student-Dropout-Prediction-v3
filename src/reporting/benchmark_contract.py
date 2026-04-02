"""Benchmark summary contract constants and validation helpers."""

from __future__ import annotations

from typing import Any

BENCHMARK_SUMMARY_VERSION = "1.1"
REQUIRED_EXPLAINABILITY_ARTIFACT_KEYS = (
    "best_model",
    "X_train_preprocessed",
    "X_test_preprocessed",
    "y_train",
)


def validate_benchmark_summary_for_explainability(summary: dict[str, Any]) -> dict[str, str]:
    """Validate benchmark summary artifact contract needed by explainability runner."""
    if not isinstance(summary, dict):
        raise ValueError("Benchmark summary must be a JSON object.")

    artifact_paths = summary.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        raise ValueError(
            "Benchmark summary contract is invalid: missing top-level 'artifact_paths' object. "
            "Please rerun benchmark generation."
        )

    missing = [key for key in REQUIRED_EXPLAINABILITY_ARTIFACT_KEYS if not artifact_paths.get(key)]
    if missing:
        raise ValueError(
            "Benchmark summary contract is invalid for explainability. "
            f"Missing artifact_paths keys: {missing}. "
            "Required keys: "
            f"{list(REQUIRED_EXPLAINABILITY_ARTIFACT_KEYS)}. "
            "Please rerun benchmark with the current runner."
        )

    best_model_name = summary.get("best_model")
    if not best_model_name:
        raise ValueError("Benchmark summary is missing 'best_model'; cannot run explainability.")

    normalized = {k: str(artifact_paths[k]) for k in REQUIRED_EXPLAINABILITY_ARTIFACT_KEYS}
    normalized["best_model_name"] = str(best_model_name)
    normalized["benchmark_summary_version"] = str(
        summary.get("benchmark_summary_version") or summary.get("schema_version") or "legacy"
    )
    return normalized

