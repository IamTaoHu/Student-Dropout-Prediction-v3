"""Config and model registry consistency tests."""

from __future__ import annotations

from pathlib import Path
import unittest

from src.models.registry import list_available_models


class ConfigAndRegistryTests(unittest.TestCase):
    def test_registry_contains_required_paper_models(self) -> None:
        available = set(list_available_models())
        for model_name in ("svm", "gradient_boosting", "lightgbm", "catboost", "xgboost"):
            self.assertIn(model_name, available)

    def test_uct_binary_config_parses(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")
        exact = Path("configs/experiments/exp_bm_uct_binary_paper_style.yaml")
        if exact.exists():
            path = exact
        else:
            candidates = sorted(Path("configs/experiments").glob("exp_bm_uct*binary*.yaml"))
            if not candidates:
                self.skipTest("No UCT binary experiment config found in this checkout.")
            path = candidates[0]
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertIn("experiment", data)
        self.assertEqual(data["experiment"]["target_formulation"], "binary")

    def test_experiment_model_names_match_registry(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")
        available = set(list_available_models())
        alias_map = {
            "xgboost_optuna": "xgboost",
            "lightgbm_optuna": "lightgbm",
            "catboost_optuna": "catboost",
        }
        exp_dir = Path("configs/experiments")
        for path in exp_dir.glob("*.yaml"):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            raw_models = payload.get("models", {})
            if isinstance(raw_models, dict):
                models = list(raw_models.get("candidates", []))
            elif isinstance(raw_models, list):
                models = list(raw_models)
            else:
                models = []
            for model_name in models:
                normalized = alias_map.get(str(model_name), str(model_name).removesuffix("_optuna"))
                self.assertIn(normalized, available, msg=f"{path.name} uses unknown model '{model_name}'")


if __name__ == "__main__":
    unittest.main()
