from types import SimpleNamespace
import unittest

from position_benchmark.acquisition import AcquisitionState, load_acquisition_policy
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

    def test_policy_deferred_supplements_consume_no_budget_and_name_source(self) -> None:
        target = "example (high)"
        state = AcquisitionState()
        state[target] = {"core"}
        state.deferred_sources[target] = "example (low)"
        policy = load_acquisition_policy()
        rows = build_acquisition_plan(
            {target: {}, "new": {}},
            policy=policy,
            acquisition_state=state,
            rating_store=FakeRatingStore({
                target: SimpleNamespace(rating=1000, rating_deviation=300),
                "new": SimpleNamespace(rating=1000, rating_deviation=200),
            }),
            freeze_checker=FakeFreezeChecker(),
            cost_calculator=FakeCostCalculator(),
            reasoning_ids=set(),
            max_cost=0.55,
        )
        target_row = next(row for row in rows if row.player_id == target)
        self.assertEqual(target_row.status, "policy-deferred")
        self.assertEqual(target_row.measured_panels, ("core",))
        self.assertEqual(target_row.policy_deferred_panels, ("game_like", "continuation_stability"))
        self.assertEqual(target_row.source_player_id, "example (low)")
        summary = plan_summary(rows, policy)
        self.assertEqual(summary["policy_deferred_models"], 1)
        self.assertEqual(summary["complete_models"], 0)
        self.assertEqual(summary["budget_deferred_models"], 1)
        self.assertEqual(summary["planned_first_attempt_calls"], 50)
        self.assertEqual(summary["estimated_base_cost"], 0.5)

    def test_ready_local_panels_only_need_publication(self) -> None:
        player = "saved (high)"
        policy = load_acquisition_policy()
        state = AcquisitionState()
        state.local_ready_panels = {player: {"core", "game_like"}}
        rows = build_acquisition_plan(
            {player: {}}, policy=policy, acquisition_state=state,
            rating_store=FakeRatingStore({player: SimpleNamespace(rating=1000, rating_deviation=300)}),
            freeze_checker=FakeFreezeChecker(), cost_calculator=FakeCostCalculator(),
            reasoning_ids=set(), max_cost=0.33,
        )
        self.assertEqual(rows[0].scheduled_panels, ("continuation_stability",))
        self.assertEqual(rows[0].publish_saved_panels, ("core", "game_like"))
        self.assertEqual(rows[0].planned_first_attempt_calls, 32)
        self.assertAlmostEqual(rows[0].estimated_base_cost, 0.32)

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

        self.assertEqual(
            by_player["partial"].conditional_panels, ("game_like", "continuation_stability")
        )
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
