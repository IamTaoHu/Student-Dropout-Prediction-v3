from __future__ import annotations

from pathlib import Path
from typing import Any

from src.reporting.artifact_manifest import update_artifact_manifest
from src.reporting.runtime_persistence import (
    _ensure_explainability_compatible_artifact_paths,
    _status_from_path,
)


def artifact_status_from_path(
    path: str | Path,
    missing_reason: str = "missing_expected_output",
) -> dict[str, str]:
    return _status_from_path(path, missing_reason=missing_reason)


def ensure_explainability_contract(summary: dict[str, Any]) -> None:
    _ensure_explainability_compatible_artifact_paths(summary)


def update_benchmark_artifact_manifest(
    *,
    output_dir: Path,
    mandatory_updates: dict[str, dict[str, Any]] | None = None,
    optional_updates: dict[str, dict[str, Any]] | None = None,
    metadata_updates: dict[str, Any] | None = None,
) -> Path:
    return update_artifact_manifest(
        output_dir=output_dir,
        mandatory_updates=mandatory_updates,
        optional_updates=optional_updates,
        metadata_updates=metadata_updates,
    )
