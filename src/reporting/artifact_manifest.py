"""Machine-readable artifact contract manifest helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "artifact_manifest.json"
_VALID_STATUSES = {"generated", "created", "inherited", "skipped", "unavailable", "failed"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    status = str(entry.get("status", "unavailable")).strip().lower()
    if status not in _VALID_STATUSES:
        status = "failed"
    normalized: dict[str, Any] = {"status": status}

    for key in ("path", "paths", "reason", "details", "source_experiment", "source_path"):
        if key in entry and entry[key] is not None:
            normalized[key] = _to_jsonable(entry[key])
    return normalized


def _initialize_manifest(output_dir: Path) -> dict[str, Any]:
    return {
        "contract_version": "1.0",
        "manifest_path": str(output_dir / MANIFEST_FILENAME),
        "updated_at": _utc_now_iso(),
        "metadata": {},
        "mandatory": {},
        "optional": {},
    }


def load_or_initialize_manifest(output_dir: Path) -> tuple[dict[str, Any], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = _initialize_manifest(output_dir)
        except Exception:
            payload = _initialize_manifest(output_dir)
    else:
        payload = _initialize_manifest(output_dir)

    payload.setdefault("metadata", {})
    payload.setdefault("mandatory", {})
    payload.setdefault("optional", {})
    payload["manifest_path"] = str(manifest_path)
    payload["updated_at"] = _utc_now_iso()
    return payload, manifest_path


def update_artifact_manifest(
    output_dir: str | Path,
    mandatory_updates: dict[str, dict[str, Any]] | None = None,
    optional_updates: dict[str, dict[str, Any]] | None = None,
    metadata_updates: dict[str, Any] | None = None,
) -> Path:
    root = Path(output_dir)
    manifest, manifest_path = load_or_initialize_manifest(root)

    if metadata_updates:
        manifest["metadata"].update(_to_jsonable(metadata_updates))

    if mandatory_updates:
        for name, entry in mandatory_updates.items():
            manifest["mandatory"][name] = _normalize_entry(entry)

    if optional_updates:
        for name, entry in optional_updates.items():
            manifest["optional"][name] = _normalize_entry(entry)

    manifest["updated_at"] = _utc_now_iso()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
