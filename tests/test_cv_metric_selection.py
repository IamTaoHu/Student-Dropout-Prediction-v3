from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_experiment import _resolve_model_selection_config, _sort_leaderboard_with_tiebreak


class CvMetricSelectionTests(unittest.TestCase):
    def test_primary_selection_metric_can_switch_to_cv_macro_f1(self) -> None:
        resolved = _resolve_model_selection_config(
            {
                "evaluation": {"primary_selection_metric": "cv_macro_f1"},
                "selection": {"tie_breakers": ["cv_macro_precision", "cv_macro_recall"]},
            }
        )
        self.assertEqual(resolved["primary_selection_metric"], "cv_macro_f1")
        self.assertListEqual(
            resolved["ranking_metrics"],
            ["cv_macro_f1", "cv_macro_precision", "cv_macro_recall"],
        )

    def test_default_selection_for_unrelated_experiments_is_unchanged(self) -> None:
        resolved = _resolve_model_selection_config({"selection": {"primary": "macro_f1"}})
        self.assertEqual(resolved["primary_selection_metric"], "macro_f1")
        self.assertListEqual(
            resolved["ranking_metrics"],
            ["macro_f1", "balanced_accuracy", "accuracy"],
        )

    def test_leaderboard_sort_can_rank_by_cv_metric_columns(self) -> None:
        leaderboard = pd.DataFrame(
            [
                {"model": "svm", "cv_macro_f1": 0.72, "cv_macro_precision": 0.70, "cv_macro_recall": 0.71},
                {"model": "xgboost", "cv_macro_f1": 0.72, "cv_macro_precision": 0.71, "cv_macro_recall": 0.70},
            ]
        )
        ranked, best_model, ranking_columns = _sort_leaderboard_with_tiebreak(
            leaderboard_df=leaderboard,
            selection_cfg={"ranking_metrics": ["cv_macro_f1", "cv_macro_precision", "cv_macro_recall"]},
            source="test",
        )
        self.assertEqual(best_model, "xgboost")
        self.assertListEqual(ranking_columns, ["cv_macro_f1", "cv_macro_precision", "cv_macro_recall"])
        self.assertEqual(str(ranked.iloc[0]["model"]), "xgboost")


if __name__ == "__main__":
    unittest.main()
