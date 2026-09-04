import unittest

from rating.reasoning_prediction import (
    CurvePrior,
    RatingObservation,
    ReasoningCurve,
    build_reasoning_curves,
    effort_from_player_id,
    fit_curve_prior,
    normalize_effort,
    predict_final_rating,
)


def observation(player_id: str, effort: str, rating: float) -> RatingObservation:
    return RatingObservation(player_id, effort, rating, 60.0, 20)


class ReasoningPredictionTests(unittest.TestCase):
    def test_normalizes_max_and_rejects_generic_thinking(self) -> None:
        self.assertEqual(normalize_effort("max"), "xhigh")
        self.assertEqual(effort_from_player_id("model (medium)"), "medium")
        self.assertIsNone(effort_from_player_id("model (thinking)"))

    def test_builds_curves_from_final_ratings(self) -> None:
        ratings = {
            "model (low)": {
                "rating": 700,
                "rating_deviation": 80,
                "games_played": 12,
            },
            "model (high)": {
                "rating": 900,
                "rating_deviation": 90,
                "games_played": 10,
            },
        }
        metadata = {
            player_id: {"model_id": "lab/model"} for player_id in ratings
        }
        curves = build_reasoning_curves(ratings, metadata)
        self.assertEqual(set(curves), {"lab/model"})
        self.assertEqual(curves["lab/model"].observations["high"].rating, 900)

    def test_release_holdout_excludes_all_gpt56_siblings(self) -> None:
        curves = {}
        for name in ("sol", "terra", "luna"):
            model_id = f"openai/gpt-5.6-{name}"
            curves[model_id] = ReasoningCurve(
                model_id=model_id,
                lab="openai",
                release_cohort="openai/gpt-5.6",
                observations={
                    "low": observation(f"{name} (low)", "low", 600),
                    "high": observation(f"{name} (high)", "high", 900),
                },
            )
        curves["openai/gpt-5.5"] = ReasoningCurve(
            model_id="openai/gpt-5.5",
            lab="openai",
            release_cohort="openai/gpt-5.5",
            observations={
                "low": observation("5.5 (low)", "low", 500),
                "high": observation("5.5 (high)", "high", 700),
            },
        )
        prior = fit_curve_prior(
            curves,
            target_model_id="openai/gpt-5.6-sol",
            exclude_release_cohort=True,
        )
        self.assertEqual(prior.lab_training_lines, ("openai/gpt-5.5",))

    def test_predicts_from_latest_or_mean_anchor(self) -> None:
        curve = ReasoningCurve(
            model_id="openai/example",
            lab="openai",
            release_cohort="openai/example",
            observations={
                "low": observation("example (low)", "low", 600),
                "medium": observation("example (medium)", "medium", 650),
            },
        )
        prior = CurvePrior(
            increments=(0.0, 0.0, 100.0, 150.0, 200.0),
            global_increments=(0.0, 0.0, 100.0, 150.0, 200.0),
            training_lines=(),
            lab_training_lines=(),
        )
        latest = predict_final_rating(
            curve,
            anchor_efforts=("low", "medium"),
            target_effort="xhigh",
            prior=prior,
            combination="latest",
        )
        mean = predict_final_rating(
            curve,
            anchor_efforts=("low", "medium"),
            target_effort="xhigh",
            prior=prior,
            combination="mean",
        )
        self.assertEqual(latest.rating, 1000)
        self.assertEqual(mean.rating, 1025)


if __name__ == "__main__":
    unittest.main()
