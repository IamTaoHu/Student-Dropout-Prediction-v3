"""Build and persist a processed UCT Student dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.adapters.uct_student_adapter import adapt_uct_student_schema
from src.data.feature_builders.uct_student_features import build_uct_student_features
from src.data.loaders.uct_student_loader import load_uct_student_dataframe


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load dataset configs. Install with `pip install pyyaml`.") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(path_value: str | Path, base_dir: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    return (ROOT / path).resolve()


def _resolve_input_path(dataset_cfg: dict[str, Any], base_dir: Path | None) -> Path:
    paths_cfg = dataset_cfg.get("paths", {})
    raw_file = paths_cfg.get("raw_file")
    if raw_file:
        return _resolve_path(raw_file, base_dir)
    raw_root = paths_cfg.get("raw_root")
    filename = dataset_cfg.get("source", {}).get("filename", "uct_student.csv")
    if raw_root:
        return _resolve_path(Path(raw_root) / filename, base_dir)
    return _resolve_path(filename, base_dir)


def _derive_new_columns(before_columns: Iterable[str], after_columns: Iterable[str]) -> list[str]:
    before_set = set(before_columns)
    return sorted([col for col in after_columns if col not in before_set])


def save_features(df: Any, dataset_cfg: dict[str, Any], base_dir: Path | None = None) -> Path:
    output_root = _resolve_path(
        dataset_cfg.get("paths", {}).get("processed_root", "data/processed/uct_student"),
        base_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_filename = dataset_cfg.get("outputs", {}).get("features_filename", "uct_student_features.parquet")
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
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/datasets/uct_student.yaml"))
    args = parser.parse_args()

    cfg_path = args.dataset_config.resolve()
    path_base_dir = ROOT
    dataset_cfg = load_yaml(cfg_path)
    resolved_input_path = _resolve_input_path(dataset_cfg, base_dir=path_base_dir)

    raw_df = load_uct_student_dataframe(dataset_cfg, base_dir=path_base_dir)
    adapted = adapt_uct_student_schema(raw_df, dataset_cfg.get("schema", {}))
    original_columns = list(adapted.get("data", raw_df).columns)
    features = build_uct_student_features(adapted, dataset_cfg.get("features", {}))
    out_path = save_features(features, dataset_cfg, base_dir=path_base_dir).resolve()
    derived_columns = _derive_new_columns(original_columns, features.columns.tolist())

    print(f"Input path: {resolved_input_path}")
    print(f"Output path: {out_path}")
    print(f"Rows: {len(features)}")
    print(f"Columns: {features.shape[1]}")
    print(f"Derived columns ({len(derived_columns)}): {', '.join(derived_columns) if derived_columns else 'none'}")


if __name__ == "__main__":
    main()
