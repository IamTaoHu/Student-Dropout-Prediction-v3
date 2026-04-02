"""Persistence helpers for explainability outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _serialize(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_list()
    return str(value)


def _save_tabular_if_present(path: Path, payload: Any) -> Path | None:
    df: pd.DataFrame | None = None
    if isinstance(payload, pd.DataFrame):
        df = payload
    elif isinstance(payload, list) and payload:
        if all(isinstance(row, dict) for row in payload):
            df = pd.json_normalize(payload)
        else:
            df = pd.DataFrame({"value": payload})
    elif isinstance(payload, dict) and payload:
        df = pd.json_normalize([payload])

    if df is not None and not df.empty:
        df.to_csv(path, index=False)
        return path
    return None


def _normalize_method_status(artifacts: dict[str, Any], method: str) -> None:
    payload = artifacts.get(method)
    if not isinstance(payload, dict):
        artifacts[method] = {
            "status": "skipped",
            "reason": "method_output_missing",
            "error_message": None,
            "artifacts": {},
            "details": {},
        }
        return

    raw_status = str(payload.get("status", "ok")).strip().lower()
    if raw_status in {"ok", "generated", "success"}:
        status = "generated"
    elif raw_status in {"skipped", "unavailable"}:
        status = "skipped"
    else:
        status = "failed"

    reason = payload.get("reason")
    if reason is not None:
        reason = str(reason)

    error_message = payload.get("error_message") or payload.get("error")
    if error_message is not None:
        error_message = str(error_message)

    artifacts_payload = payload.get("artifacts")
    if not isinstance(artifacts_payload, dict):
        artifacts_payload = {}

    details: dict[str, Any] = {
        k: v
        for k, v in payload.items()
        if k not in {"status", "reason", "error_message", "error", "artifacts"}
    }
    normalized = {
        "status": status,
        "reason": reason,
        "error_message": error_message,
        "artifacts": artifacts_payload,
        "details": details,
        **details,
    }
    artifacts[method] = normalized


def _append_saved_file(method_payload: dict[str, Any], path: Path) -> None:
    method_artifacts = method_payload.setdefault("artifacts", {})
    saved = method_artifacts.get("saved_files")
    if not isinstance(saved, list):
        saved = []
        method_artifacts["saved_files"] = saved
    saved.append(str(path))


def save_explanation_report(artifacts: dict[str, Any], output_dir: Path) -> Path:
    """Save explainability artifacts as JSON, CSV indexes, and markdown summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _normalize_method_status(artifacts, "shap")
    _normalize_method_status(artifacts, "lime")
    _normalize_method_status(artifacts, "aime")

    csv_paths: list[Path] = []
    if "shap" in artifacts:
        csv = _save_tabular_if_present(output_dir / "shap_global_importance.csv", artifacts["shap"].get("global_importance"))
        if csv:
            csv_paths.append(csv)
            _append_saved_file(artifacts["shap"], csv)
        shap_local = artifacts["shap"].get("local_importance", artifacts["shap"].get("local_explanations"))
        csv = _save_tabular_if_present(output_dir / "shap_local_importance.csv", shap_local)
        if csv:
            csv_paths.append(csv)
            _append_saved_file(artifacts["shap"], csv)
    if "lime" in artifacts:
        lime_local = artifacts["lime"].get("local_importance", artifacts["lime"].get("results"))
        csv = _save_tabular_if_present(output_dir / "lime_local_importance.csv", lime_local)
        if csv:
            csv_paths.append(csv)
            _append_saved_file(artifacts["lime"], csv)
    if "aime" in artifacts:
        for name in ("global_importance", "per_class_importance", "local_importance", "representative_instances"):
            csv = _save_tabular_if_present(output_dir / f"aime_{name}.csv", artifacts["aime"].get(name))
            if csv:
                csv_paths.append(csv)
                _append_saved_file(artifacts["aime"], csv)

    md_lines = ["# Explainability Report", "", "## Saved Files", ""]
    if csv_paths:
        md_lines.extend([f"- `{p.name}`" for p in csv_paths])
    else:
        md_lines.append("- No tabular explainability outputs were generated.")

    if artifacts.get("aime", {}).get("similarity_plot_path"):
        md_lines.append(f"- `aime_similarity.png` (path: {artifacts['aime']['similarity_plot_path']})")
        _append_saved_file(artifacts["aime"], Path(artifacts["aime"]["similarity_plot_path"]))

    (output_dir / "explanation_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    json_path = output_dir / "explanation_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2, default=_serialize)
    return json_path
