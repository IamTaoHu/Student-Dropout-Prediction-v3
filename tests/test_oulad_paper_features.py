"""Regression tests for OULAD feature leakage safeguards."""

from __future__ import annotations

import unittest

import pandas as pd

from src.data.feature_builders.oulad_paper_features import build_oulad_paper_features


class OuladPaperFeatureLeakageTests(unittest.TestCase):
    def test_builder_output_keeps_final_result_for_target_mapping(self) -> None:
        adapted = {
            "tables": {
                "studentinfo": pd.DataFrame(
                    {
                        "id_student": [1],
                        "code_module": ["AAA"],
                        "code_presentation": ["2013J"],
                        "gender": ["M"],
                        "region": ["East"],
                        "highest_education": ["A Level"],
                        "imd_band": ["20-30%"],
                        "age_band": ["35-55"],
                        "num_of_prev_attempts": [0],
                        "studied_credits": [60],
                        "disability": ["N"],
                        "final_result": ["Pass"],
                    }
                ),
                "studentregistration": pd.DataFrame(
                    {
                        "id_student": [1],
                        "code_module": ["AAA"],
                        "code_presentation": ["2013J"],
                        "date_registration": [-5],
                        "date_unregistration": [None],
                    }
                ),
                "studentassessment": pd.DataFrame(
                    {
                        "id_student": [1],
                        "id_assessment": [10],
                        "score": [78],
                        "date_submitted": [18],
                    }
                ),
                "assessments": pd.DataFrame(
                    {
                        "id_assessment": [10],
                        "code_module": ["AAA"],
                        "code_presentation": ["2013J"],
                        "assessment_type": ["TMA"],
                        "weight": [100],
                        "date": [15],
                    }
                ),
                "studentvle": pd.DataFrame(
                    {
                        "id_student": [1],
                        "code_module": ["AAA"],
                        "code_presentation": ["2013J"],
                        "id_site": [1001],
                        "date": [10],
                        "sum_click": [20],
                    }
                ),
                "vle": pd.DataFrame(
                    {
                        "id_site": [1001],
                        "code_module": ["AAA"],
                        "code_presentation": ["2013J"],
                        "activity_type": ["resource"],
                        "week_from": [0],
                        "week_to": [30],
                    }
                ),
            }
        }

        features = build_oulad_paper_features(adapted, feature_config={"cutoff_day": 30})

        self.assertIn("final_result", features.columns)
        self.assertEqual(features.loc[0, "final_result"], "Pass")
        self.assertIn("id_student", features.columns)
        self.assertIn("code_module", features.columns)
        self.assertIn("code_presentation", features.columns)


if __name__ == "__main__":
    unittest.main()
