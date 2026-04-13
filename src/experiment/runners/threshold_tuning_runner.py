from __future__ import annotations

from pathlib import Path
from typing import Any

from src.reporting.threshold_tuning import run_threshold_tuning_experiment


def run_threshold_tuning_mode(
    *,
    exp_cfg: dict[str, Any],
    experiment_config_path: Path,
) -> dict[str, Any]:
    return run_threshold_tuning_experiment(
        exp_cfg=exp_cfg,
        experiment_config_path=experiment_config_path,
    )
