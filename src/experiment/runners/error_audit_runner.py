from __future__ import annotations

from pathlib import Path
from typing import Any

from src.reporting.error_audit import run_uct_3class_error_audit


def run_error_audit_mode(
    *,
    exp_cfg: dict[str, Any],
    experiment_config_path: Path,
) -> dict[str, Any]:
    return run_uct_3class_error_audit(
        exp_cfg=exp_cfg,
        experiment_config_path=experiment_config_path,
    )
