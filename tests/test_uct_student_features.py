from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.build_uct_student_dataset import main as build_uct_main
from src.data.feature_builders.uct_student_features import build_uct_student_features


class UctStudentFeatureBuilderTests(unittest.TestCase):
    def test_semester_progression_features_are_derived_from_normalized_columns(self) -> None:
        df = pd.DataFrame(
            {
                "__row_id__": [0, 1],
                "target": ["Dropout", "Graduate"],
                "curricular_units_1st_sem_approved": [5, 10],
                "curricular_units_2nd_sem_approved": [3, 12],
                "curricular_units_1st_sem_enrolled": [10, 10],
                "curricular_units_2nd_sem_enrolled": [8, 12],
                "curricular_units_1st_sem_evaluations": [12, 10],
                "curricular_units_2nd_sem_evaluations": [9, 12],
                "curricular_units_1st_sem_without_evaluations": [2, 0],
                "curricular_units_2nd_sem_without_evaluations": [1, 0],
                "curricular_units_1st_sem_grade": [12.0, 15.0],
                "curricular_units_2nd_sem_grade": [11.0, 16.0],
            }
        )
        adapted = {"data": df, "id_column": "__row_id__", "target_column": "target"}
        out = build_uct_student_features(adapted, {"derive_safe_features": True})

        expected_cols = {
            "approval_rate_1",
            "approval_rate_2",
            "eval_success_rate_1",
            "eval_success_rate_2",
            "eval_gap_1",
            "eval_gap_2",
            "no_eval_ratio_1",
            "no_eval_ratio_2",
            "grade_delta_2_minus_1",
            "approved_delta_2_minus_1",
            "enrolled_delta_2_minus_1",
            "approval_rate_delta",
            "grade_mean_12",
            "grade_abs_delta",
            "approved_consistency",
            "enrolled_consistency",
            "approved_to_enrolled_total",
            "approved_to_evaluated_total",
        }
        self.assertTrue(expected_cols.issubset(set(out.columns)))
        self.assertAlmostEqual(float(out.loc[0, "approval_rate_1"]), 0.5)
        self.assertAlmostEqual(float(out.loc[0, "approval_rate_2"]), 0.375)
        self.assertAlmostEqual(float(out.loc[0, "eval_gap_1"]), 7.0)
        self.assertAlmostEqual(float(out.loc[0, "grade_delta_2_minus_1"]), -1.0)
        self.assertAlmostEqual(float(out.loc[0, "grade_mean_12"]), 11.5)
        self.assertAlmostEqual(float(out.loc[0, "approved_to_enrolled_total"]), 0.4444444444, places=6)

    def test_safe_ratio_defaults_to_zero_for_zero_or_negative_denominator(self) -> None:
        df = pd.DataFrame(
            {
                "__row_id__": [0, 1, 2],
                "target": ["Dropout", "Enrolled", "Graduate"],
                "approved_1st_sem": [5, 5, 5],
                "enrolled_1st_sem": [0, -1, 10],
                "without_evaluations_1st_sem": [1, 1, 1],
            }
        )
        adapted = {"data": df, "id_column": "__row_id__", "target_column": "target"}
        out = build_uct_student_features(adapted, {"derive_safe_features": True})

        self.assertListEqual(out["approval_rate_1"].round(6).tolist(), [0.0, 0.0, 0.5])
        self.assertListEqual(out["no_eval_ratio_1"].round(6).tolist(), [0.0, 0.0, 0.1])

    def test_enrolled_focus_features_are_optional_and_schema_safe(self) -> None:
        df = pd.DataFrame(
            {
                "__row_id__": [0, 1],
                "target": ["Enrolled", "Graduate"],
                "approved_1st_sem": [4, 8],
                "approved_2nd_sem": [5, 9],
                "enrolled_1st_sem": [6, 10],
                "enrolled_2nd_sem": [6, 10],
                "evaluations_1st_sem": [7, 10],
                "evaluations_2nd_sem": [8, 10],
                "grade_1st_sem": [12.0, 14.0],
                "grade_2nd_sem": [13.0, 15.0],
            }
        )
        adapted = {"data": df, "id_column": "__row_id__", "target_column": "target"}
        out = build_uct_student_features(
            adapted,
            {"derive_safe_features": True, "derive_enrolled_focus_features": True},
        )

        expected_cols = {
            "semester_approved_ratio_1st_sem",
            "semester_approved_ratio_2nd_sem",
            "semester_grade_delta",
            "semester_approved_delta",
            "evaluation_to_approved_gap",
            "overall_progress_consistency",
            "semester_load_retention_ratio",
            "semester_approval_retention_ratio",
            "stability_gap_ratio",
            "approved_ratio_sem1",
            "approved_ratio_sem2",
            "grade_delta_sem2_minus_sem1",
            "approved_delta_sem2_minus_sem1",
            "enrolled_delta_sem2_minus_sem1",
            "evaluation_gap_total",
            "progress_consistency_ratio",
            "sem1_to_sem2_grade_stability",
            "sem1_to_sem2_approval_stability",
            "low_progress_but_active",
            "academic_momentum",
        }
        self.assertTrue(expected_cols.issubset(set(out.columns)))
        self.assertAlmostEqual(float(out.loc[0, "semester_approved_ratio_1st_sem"]), 4.0 / 6.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "approved_ratio_sem1"]), 4.0 / 6.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "semester_grade_delta"]), 1.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "grade_delta_sem2_minus_sem1"]), 1.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "semester_approved_delta"]), 1.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "approved_delta_sem2_minus_sem1"]), 1.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "evaluation_to_approved_gap"]), 6.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "evaluation_gap_total"]), 6.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "progress_consistency_ratio"]), 9.0 / 12.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "sem1_to_sem2_grade_stability"]), 0.5, places=6)
        self.assertAlmostEqual(float(out.loc[0, "sem1_to_sem2_approval_stability"]), 4.0 / 5.0, places=6)
        self.assertEqual(float(out.loc[0, "academic_momentum"]), 1.0)

    def test_enrolled_feature_groups_build_compact_pack_and_skip_missing_optional_dependencies(self) -> None:
        df = pd.DataFrame(
            {
                "__row_id__": [0, 1],
                "target": ["Enrolled", "Graduate"],
                "approved_1st_sem": [4, 8],
                "approved_2nd_sem": [5, 9],
                "enrolled_1st_sem": [6, 10],
                "enrolled_2nd_sem": [7, 10],
                "grade_1st_sem": [12.0, 14.0],
                "grade_2nd_sem": [13.0, 15.0],
            }
        )
        adapted = {"data": df, "id_column": "__row_id__", "target_column": "target"}
        out = build_uct_student_features(
            adapted,
            {
                "derive_safe_features": True,
                "enrolled_feature_groups": {
                    "enabled": True,
                    "groups": ["efficiency", "gap", "trend", "consistency", "near_graduate"],
                },
            },
        )

        expected_cols = {
            "sem1_approval_rate",
            "sem2_approval_rate",
            "overall_approval_rate",
            "sem1_grade_efficiency",
            "sem2_grade_efficiency",
            "overall_grade_efficiency",
            "sem1_gap",
            "sem2_gap",
            "persistence_gap",
            "sem1_unfinished_ratio",
            "sem2_unfinished_ratio",
            "persistence_gap_ratio",
            "approval_rate_delta",
            "grade_delta",
            "grade_efficiency_delta",
            "load_delta",
            "completion_delta",
            "gap_delta",
            "approval_consistency",
            "grade_consistency",
            "workload_balance",
            "completion_balance",
            "gap_balance",
            "sem2_completion_strength",
            "overall_completion_strength",
            "completion_strength_delta",
            "near_graduate_gap_signal",
        }
        self.assertTrue(expected_cols.issubset(set(out.columns)))
        self.assertNotIn("evaluation_to_approval_rate", out.columns)
        self.assertNotIn("evaluation_gap", out.columns)
        self.assertAlmostEqual(float(out.loc[0, "sem1_approval_rate"]), 4.0 / 6.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "sem2_approval_rate"]), 5.0 / 7.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "sem1_gap"]), 2.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "persistence_gap_ratio"]), 4.0 / 13.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "grade_efficiency_delta"]), (13.0 / 5.0) - (12.0 / 4.0), places=6)
        self.assertAlmostEqual(float(out.loc[0, "completion_balance"]), 4.0 / 5.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "near_graduate_gap_signal"]), (9.0 / 13.0) * (25.0 / 9.0) - (4.0 / 13.0), places=6)

    def test_derived_columns_are_not_target_dependent(self) -> None:
        base = pd.DataFrame(
            {
                "__row_id__": [0, 1],
                "approved_1st_sem": [3, 4],
                "approved_2nd_sem": [4, 5],
                "enrolled_1st_sem": [5, 6],
                "enrolled_2nd_sem": [5, 6],
                "evaluations_1st_sem": [5, 6],
                "evaluations_2nd_sem": [5, 6],
                "without_evaluations_1st_sem": [0, 0],
                "without_evaluations_2nd_sem": [0, 0],
                "grade_1st_sem": [12.0, 13.0],
                "grade_2nd_sem": [13.0, 13.5],
            }
        )
        df_a = base.copy()
        df_a["target"] = ["Dropout", "Graduate"]
        df_b = base.copy()
        df_b["target"] = ["Graduate", "Dropout"]

        out_a = build_uct_student_features({"data": df_a, "id_column": "__row_id__", "target_column": "target"}, {})
        out_b = build_uct_student_features({"data": df_b, "id_column": "__row_id__", "target_column": "target"}, {})

        derived_cols = [col for col in out_a.columns if col not in df_a.columns]
        self.assertTrue(derived_cols)
        pd.testing.assert_frame_equal(
            out_a[derived_cols].reset_index(drop=True),
            out_b[derived_cols].reset_index(drop=True),
        )


class BuildUctStudentScriptTests(unittest.TestCase):
    def test_build_script_uses_csv_flow_and_reports_summary(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_dir = tmp_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_csv = raw_dir / "uct_student.csv"

            pd.DataFrame(
                {
                    "Target": ["Dropout", "Graduate"],
                    "Curricular units 1st sem (approved)": [5, 7],
                    "Curricular units 2nd sem (approved)": [4, 8],
                    "Curricular units 1st sem (enrolled)": [10, 10],
                    "Curricular units 2nd sem (enrolled)": [10, 10],
                }
            ).to_csv(raw_csv, sep=";", index=False)

            dataset_cfg = {
                "dataset": {"name": "uct_student", "source_format": "csv"},
                "paths": {
                    "raw_file": str(raw_csv),
                    "processed_root": str(tmp_dir / "processed"),
                },
                "source": {"delimiter": ";", "encoding": "utf-8"},
                "schema": {"entity_id": None, "term_column": None, "outcome_column": "Target"},
                "features": {"derive_safe_features": True},
                "outputs": {"features_filename": "uct_student_features.csv"},
            }
            dataset_cfg_path = tmp_dir / "uct_dataset.yaml"
            dataset_cfg_path.write_text(yaml.safe_dump(dataset_cfg), encoding="utf-8")

            stdout = StringIO()
            with patch("sys.argv", ["build_uct_student_dataset.py", "--dataset-config", str(dataset_cfg_path)]):
                with redirect_stdout(stdout):
                    build_uct_main()

            out_text = stdout.getvalue()
            expected_output_file = tmp_dir / "processed" / "uct_student_features.csv"
            self.assertTrue(expected_output_file.exists())
            self.assertIn("Input path:", out_text)
            self.assertIn("Output path:", out_text)
            self.assertIn("Rows:", out_text)
            self.assertIn("Columns:", out_text)
            self.assertIn("Derived columns (", out_text)


if __name__ == "__main__":
    unittest.main()
