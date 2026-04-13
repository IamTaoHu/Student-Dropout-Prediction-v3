from __future__ import annotations

from pathlib import Path
from typing import Any


def finalize_threshold_tuning_run(
    *,
    result: dict[str, Any],
    exp_cfg: dict[str, Any],
    experiment_config_path: Path,
) -> dict[str, Any]:
    return result


def finalize_error_audit_run(
    *,
    result: dict[str, Any],
    exp_cfg: dict[str, Any],
    experiment_config_path: Path,
) -> dict[str, Any]:
    return result
