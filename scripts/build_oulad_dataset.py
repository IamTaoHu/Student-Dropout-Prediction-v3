"""Build and persist a processed OULAD feature dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.adapters.oulad_adapter import adapt_oulad_schema
from src.data.feature_builders.oulad_paper_features import build_oulad_paper_features
from src.data.loaders.oulad_loader import load_oulad_tables


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load dataset configs. Install with `pip install pyyaml`.") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_features(df: Any, dataset_cfg: dict[str, Any]) -> Path:
    output_root = Path(dataset_cfg.get("paths", {}).get("processed_root", "data/processed/oulad"))
    output_root.mkdir(parents=True, exist_ok=True)
    output_filename = dataset_cfg.get("outputs", {}).get("features_filename", "oulad_features.parquet")
    out_path = output_root / output_filename
    if str(out_path).lower().endswith(".csv"):
        df.to_csv(out_path, index=False)
    else:
        try:
            df.to_parquet(out_path, index=False)
        except Exception:
            fallback = out_path.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            out_path = fallback
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/datasets/oulad.yaml"))
    args = parser.parse_args()

    dataset_cfg = load_yaml(args.dataset_config)
    tables = load_oulad_tables(dataset_cfg)
    adapted = adapt_oulad_schema(tables, dataset_cfg.get("schema", {}))
    features = build_oulad_paper_features(adapted, dataset_cfg.get("features", {}))
    out_path = save_features(features, dataset_cfg)
    print(f"Saved processed OULAD features to: {out_path}")
    print(f"Shape: {features.shape}")


if __name__ == "__main__":
    main()
