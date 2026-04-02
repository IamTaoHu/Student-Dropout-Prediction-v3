"""Validate raw UCT label vocabulary against configured target mappings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiment import load_yaml


def _normalize(value: object) -> str:
    return str(value).strip().lower()


def _collect_mapping_labels(dataset_cfg: dict) -> set[str]:
    target_mappings = dataset_cfg.get("target_mappings", {}) or {}
    labels: set[str] = set()
    for formulation in ("binary", "three_class", "four_class"):
        mapping = target_mappings.get(formulation, {}) or {}
        labels.update(_normalize(k) for k in mapping.keys())
    return labels


def validate_uct_label_mapping(dataset_config_path: Path) -> int:
    dataset_cfg = load_yaml(dataset_config_path)
    raw_file = Path(dataset_cfg.get("paths", {}).get("raw_file", ""))
    source_cfg = dataset_cfg.get("source", {})
    outcome_column = dataset_cfg.get("schema", {}).get("outcome_column", "Target")
    delimiter = source_cfg.get("delimiter", ";")
    encoding = source_cfg.get("encoding", "utf-8")

    if not raw_file.exists():
        print(f"[PENDING] Raw UCT file not found: {raw_file}")
        print("Provide the raw file and rerun this validator.")
        return 2

    raw_df = pd.read_csv(raw_file, delimiter=delimiter, encoding=encoding)
    if outcome_column not in raw_df.columns:
        print(f"[ERROR] Outcome column '{outcome_column}' not found in raw file.")
        return 1

    raw_labels = sorted({_normalize(v) for v in raw_df[outcome_column].dropna().unique().tolist()})
    mapped_labels = sorted(_collect_mapping_labels(dataset_cfg))
    unmapped_raw = sorted(set(raw_labels).difference(mapped_labels))
    extra_mapped = sorted(set(mapped_labels).difference(raw_labels))

    print("UCT label mapping validation")
    print(f"- Dataset config: {dataset_config_path}")
    print(f"- Raw file: {raw_file}")
    print(f"- Outcome column: {outcome_column}")
    print(f"- Raw unique labels: {raw_labels}")
    print(f"- Config mapped labels (all formulations): {mapped_labels}")
    print(f"- Unmapped raw labels: {unmapped_raw}")
    print(f"- Config-only labels not present in raw file: {extra_mapped}")

    if unmapped_raw:
        print("[FAIL] Some raw labels are not covered by configured mappings.")
        return 1

    print("[OK] All observed raw UCT labels are covered by configured mappings.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/datasets/uct_student.yaml"),
        help="Path to UCT dataset config file.",
    )
    args = parser.parse_args()
    raise SystemExit(validate_uct_label_mapping(args.dataset_config))


if __name__ == "__main__":
    main()

