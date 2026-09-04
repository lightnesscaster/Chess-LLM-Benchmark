import json
from pathlib import Path
import unittest

import numpy as np

from scripts.analyze_supplemental_predictors import (
    inner_family_alpha,
    model_family,
    nested_lofo_residual_predictions,
)


class SupplementalPredictorTests(unittest.TestCase):
    def test_gpt56_base_lines_are_distinct_families(self) -> None:
        self.assertEqual(model_family("gpt-5.6-luna (low)"), "gpt-5.6-luna")
        self.assertEqual(model_family("gpt-5.6-terra (xhigh)"), "gpt-5.6-terra")
        self.assertEqual(model_family("gpt-5.6-sol (medium)"), "gpt-5.6-sol")
        self.assertEqual(model_family("gpt-5.5 (high)"), "gpt-5.5")

    def test_inner_alpha_defaults_to_strong_shrinkage_with_two_families(self) -> None:
        alpha = inner_family_alpha(
            np.asarray([[0.0], [1.0]]),
            np.asarray([0.0, 1.0]),
            ["a", "b"],
            [0.1, 1.0, 10.0, 100.0],
        )
        self.assertEqual(alpha, 100.0)

    def test_outer_predictions_cover_every_held_out_family(self) -> None:
        rows = [
            {
                "player_id": f"p{index}",
                "family": family,
                "actual": float(index * 10),
                "production_prediction": 0.0,
                "feature": float(index),
            }
            for index, family in enumerate(["a", "a", "b", "b", "c", "c"])
        ]
        predictions, selected = nested_lofo_residual_predictions(
            rows,
            ["feature"],
            [0.1, 1.0, 10.0],
        )
        self.assertEqual(len(predictions), len(rows))
        self.assertEqual(set(selected), {"a", "b", "c"})
        self.assertTrue(all(np.isfinite(predictions)))

    def test_frozen_analysis_cannot_promote_current_candidate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        analysis = json.loads(
            (
                root
                / "position_benchmark/validation/2026-07-17-supplemental-predictor-analysis.json"
            ).read_text()
        )
        cohort = analysis["cohorts"]["all_panels"]

        self.assertEqual(cohort["configurations"], 12)
        self.assertEqual(cohort["family_count"], 3)
        self.assertFalse(analysis["recommendation_gate"]["passed"])
        self.assertEqual(
            analysis["recommendation_gate"]["decision"],
            "retain-current-production-predictor",
        )


if __name__ == "__main__":
    unittest.main()
