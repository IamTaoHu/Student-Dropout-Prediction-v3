"""Run explainability suite from saved experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explainability.orchestrator import run_explainability_suite
from src.reporting.explanation_report import save_explanation_report
from src.reporting.artifact_manifest import update_artifact_manifest
from src.reporting.benchmark_contract import validate_benchmark_summary_for_explainability


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load explainability configs. Install with `pip install pyyaml`.") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required explainability artifact not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _method_status_to_manifest(method_payload: dict[str, Any] | None, fallback_reason: str) -> dict[str, str]:
    if not isinstance(method_payload, dict):
        return {"status": "skipped", "reason": fallback_reason}
    raw_status = str(method_payload.get("status", "ok")).strip().lower()
    if raw_status in {"ok", "generated", "success"}:
        return {"status": "generated"}
    reason = str(method_payload.get("reason", fallback_reason))
    if raw_status == "skipped":
        return {"status": "skipped", "reason": reason}
    return {"status": "failed", "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=False)
    parser.add_argument("--benchmark-summary", type=Path, required=True)
    parser.add_argument("--features-file", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=False)
    args = parser.parse_args()

    with args.benchmark_summary.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    exp_cfg = load_yaml(args.experiment_config) if args.experiment_config else {}

    contract = validate_benchmark_summary_for_explainability(summary)
    best_model_name = contract["best_model_name"]
    model = joblib.load(contract["best_model"])

    X_train_path = Path(contract["X_train_preprocessed"])
    X_test_path = Path(contract["X_test_preprocessed"])
    y_train_path = Path(contract["y_train"])
    X_train = _load_table(X_train_path)
    X_test = _load_table(X_test_path)
    y_train_df = _load_table(y_train_path)
    y_reference = y_train_df.iloc[:, 0] if not y_train_df.empty else None

    if args.features_file:
        features = _load_table(args.features_file)
        _ = features  # optional input retained for backwards compatibility

    explainability_cfg = exp_cfg.get("explainability", {})
    if not explainability_cfg:
        explainability_cfg = {
            "shap": {"enabled": True, "top_k": 15, "instance_indices": [0]},
            "lime": {"enabled": True, "top_k": 10, "instance_indices": [0]},
            "aime": {"enabled": True, "top_k": 15, "suppress_zero_onehot": True},
        }

    default_output_dir = Path(summary.get("output_dir", Path(args.benchmark_summary).parent)) / "explainability"
    output_dir = args.output_dir or default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    explain_results = run_explainability_suite(
        model=model,
        X_train=X_train,
        X_background=X_train,
        X_explain=X_test,
        config=explainability_cfg,
        y_reference=y_reference,
        output_dir=output_dir,
    )
    report_path = save_explanation_report(explain_results, output_dir=output_dir)

    shap_files = [
        output_dir / "shap_global_importance.csv",
        output_dir / "shap_local_importance.csv",
    ]
    lime_files = [output_dir / "lime_local_importance.csv"]
    aime_files = [
        output_dir / "aime_global_importance.csv",
        output_dir / "aime_per_class_importance.csv",
        output_dir / "aime_local_importance.csv",
        output_dir / "aime_representative_instances.csv",
        output_dir / "aime_similarity.png",
    ]

    def _aggregate_status(files: list[Path], fallback: dict[str, str]) -> dict[str, Any]:
        existing = [str(p) for p in files if p.exists()]
        if existing:
            return {"status": "generated", "paths": existing}
        if fallback.get("status") == "generated":
            return {"status": "failed", "reason": "generated_but_expected_files_missing"}
        return fallback

    shap_status = _method_status_to_manifest(explain_results.get("shap"), "shap_not_run_or_unavailable")
    lime_status = _method_status_to_manifest(explain_results.get("lime"), "lime_not_run_or_unavailable")
    aime_status = _method_status_to_manifest(explain_results.get("aime"), "aime_not_run_or_unavailable")

    update_artifact_manifest(
        output_dir=Path(summary.get("output_dir", Path(args.benchmark_summary).parent)),
        optional_updates={
            "explainability_dir": {"status": "generated", "path": str(output_dir)},
            "explainability_report_json": {"status": "generated", "path": str(report_path)},
            "shap_outputs": _aggregate_status(shap_files, shap_status),
            "lime_outputs": _aggregate_status(lime_files, lime_status),
            "aime_outputs": _aggregate_status(aime_files, aime_status),
        },
        metadata_updates={
            "explainability_last_model": best_model_name,
            "benchmark_summary_version": contract.get("benchmark_summary_version"),
            "manifest_scope": "benchmark+explainability",
        },
    )
    print(f"Explainability completed for model: {best_model_name}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
