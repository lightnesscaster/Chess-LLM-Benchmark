from types import SimpleNamespace
import unittest

from position_benchmark.acquisition import load_acquisition_policy
from scripts.plan_benchmark_acquisition import (
    build_acquisition_plan,
    plan_summary,
    reasoning_player_ids,
    selected_llm_configs,
)


class FakeRatingStore:
    def __init__(self, ratings):
        self.ratings = ratings

    def get(self, player_id):
        return self.ratings[player_id]


class FakeFreezeChecker:
    def __init__(self, frozen=()):
        self.frozen = set(frozen)
        self.stats_collector = SimpleNamespace(get_player_stats=lambda: {})

    def get_player_cost(self, _player_id):
        return 0.0

    def is_frozen(self, player_id, _rd, _stats):
        return player_id in self.frozen


class FakeCostCalculator:
    def estimate_position_benchmark_cost(self, _player_id, **kwargs):
        return kwargs["num_positions"] / 100.0


class AcquisitionPreflightTests(unittest.TestCase):
    def test_codex_preflight_selects_only_codex_configs(self) -> None:
        config = {
            "llms": [
                {
                    "player_id": "codex-model",
                    "model_name": "gpt-test",
                    "api": "codex",
                    "reasoning_effort": "high",
                },
                {
                    "player_id": "unavailable-codex-model",
                    "model_name": "gpt-unavailable",
                    "api": "codex",
                    "unavailable": True,
                },
                {"player_id": "other-model", "model_name": "other/test"},
            ]
        }

        selected = selected_llm_configs(config, "codex")

        self.assertEqual(set(selected), {"codex-model (high)"})
        self.assertEqual(reasoning_player_ids(selected), {"codex-model (high)"})

    def test_backend_preflight_keeps_direct_gemini_provider_isolated(self) -> None:
        config = {
            "llms": [
                {
                    "player_id": "direct-gemini",
                    "model_name": "google/gemini-test",
                    "api": "gemini",
                    "reasoning_effort": "medium",
                },
                {
                    "player_id": "openrouter-gemini",
                    "model_name": "google/gemini-test",
                },
                {
                    "player_id": "codex-model",
                    "model_name": "openai/gpt-test",
                    "api": "codex",
                },
            ]
        }

        self.assertEqual(
            set(selected_llm_configs(config, "gemini")),
            {"direct-gemini (medium)"},
        )
        self.assertEqual(
            set(selected_llm_configs(config, "openrouter")),
            {"openrouter-gemini"},
        )

    def test_budget_plan_is_ordered_resumable_and_zero_call(self) -> None:
        policy = load_acquisition_policy()
        configs = {
            "partial": {"model_name": "test/partial"},
            "complete": {"model_name": "test/complete"},
            "frozen": {"model_name": "test/frozen"},
        }
        ratings = {
            "partial": SimpleNamespace(rating=1000, rating_deviation=300),
            "complete": SimpleNamespace(rating=1000, rating_deviation=200),
            "frozen": SimpleNamespace(rating=1000, rating_deviation=100),
        }
        state = {
            "partial": set(),
            "complete": {panel.name for panel in policy.panels},
            "frozen": {"core"},
        }

        rows = build_acquisition_plan(
            configs,
            policy=policy,
            acquisition_state=state,
            rating_store=FakeRatingStore(ratings),
            freeze_checker=FakeFreezeChecker({"frozen"}),
            cost_calculator=FakeCostCalculator(),
            reasoning_ids=set(),
            max_cost=0.75,
        )
        by_player = {row.player_id: row for row in rows}

        self.assertEqual(by_player["partial"].status, "queued-partial")
        self.assertEqual(by_player["partial"].scheduled_panels, ("core",))
        self.assertEqual(
            by_player["partial"].deferred_panels,
            ("game_like", "continuation_stability"),
        )
        self.assertEqual(by_player["partial"].planned_first_attempt_calls, 50)
        self.assertEqual(by_player["frozen"].status, "frozen")
        self.assertEqual(by_player["complete"].status, "complete")

        summary = plan_summary(rows, policy)
        self.assertTrue(summary["zero_model_calls"])
        self.assertEqual(summary["queued_models"], 1)
        self.assertEqual(summary["budget_deferred_models"], 1)
        self.assertEqual(summary["fully_budget_deferred_models"], 0)


if __name__ == "__main__":
    unittest.main()
