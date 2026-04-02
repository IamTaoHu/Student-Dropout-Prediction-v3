"""Shared output-layout helpers for benchmark-style experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_standard_output_layout(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = {
        "root": output_dir,
        "explainability": output_dir / "explainability",
        "figures": output_dir / "figures",
        "model": output_dir / "model",
        "runtime_artifacts": output_dir / "runtime_artifacts",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def resolve_results_dir(exp_cfg: dict[str, Any], experiment_id: str) -> Path:
    """Resolve results directory with backward-compatible output key aliases."""
    outputs_cfg = exp_cfg.get("outputs", {}) or {}
    output_cfg = exp_cfg.get("output", {}) or {}
    candidate = (
        outputs_cfg.get("results_dir")
        or outputs_cfg.get("dir")
        or output_cfg.get("results_dir")
        or output_cfg.get("dir")
    )
    return Path(str(candidate or f"results/{experiment_id}"))


def write_skipped_explainability_report(
    output_dir: Path,
    reason: str,
    details: str | None = None,
) -> tuple[Path, Path]:
    explainability_dir = ensure_standard_output_layout(output_dir)["explainability"]
    json_path = explainability_dir / "explanation_report.json"
    md_path = explainability_dir / "explanation_report.md"

    payload: dict[str, Any] = {
        "shap": {"status": "skipped", "reason": reason, "artifacts": {}},
        "lime": {"status": "skipped", "reason": reason, "artifacts": {}},
        "aime": {"status": "skipped", "reason": reason, "artifacts": {}},
    }
    if details:
        payload["details"] = details
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Explainability Report",
        "",
        "- Status: `skipped`",
        f"- Reason: `{reason}`",
    ]
    if details:
        lines.append(f"- Details: {details}")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def infer_source_experiment_name(source_path: str | Path) -> str:
    path = Path(source_path)
    if path.name == "benchmark_summary.json":
        return path.parent.name
    return path.name
