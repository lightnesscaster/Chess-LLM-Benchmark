from pathlib import Path
import unittest

import yaml

from cli import create_llm_players
from game.freeze_checker import FreezeChecker
from position_benchmark.stability_cap_shadow import model_family
from rating.cost_calculator import CostCalculator


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
PLAYER_IDS = {
    "deepseek-v4-flash-0731 (no thinking)",
    "deepseek-v4-flash-0731 (high)",
    "deepseek-v4-flash-0731 (max)",
}
GEMINI_38_MODEL_ID = "google/gemini-3.8-flash"
GEMINI_38_PLAYER_ID = "gemini-3.8-flash (medium)"


class DeepSeekV4Flash0731RegistrationTests(unittest.TestCase):
    def test_benchmark_factory_registers_all_supported_reasoning_modes(self) -> None:
        with open(ROOT / "config" / "benchmark.yaml") as config_file:
            config = yaml.safe_load(config_file)

        players, reasoning_ids = create_llm_players(
            config,
            api_key="registration-test",
        )

        self.assertLessEqual(PLAYER_IDS, set(players))
        self.assertEqual(
            {player_id for player_id in players if player_id.startswith("deepseek-v4-flash-0731")},
            PLAYER_IDS,
        )
        self.assertTrue(all(players[player_id].model_name == MODEL_ID for player_id in PLAYER_IDS))
        self.assertFalse(players["deepseek-v4-flash-0731 (no thinking)"].reasoning)
        self.assertEqual(players["deepseek-v4-flash-0731 (high)"].reasoning_effort, "high")
        self.assertEqual(players["deepseek-v4-flash-0731 (max)"].reasoning_effort, "xhigh")
        self.assertEqual(
            reasoning_ids & PLAYER_IDS,
            PLAYER_IDS - {"deepseek-v4-flash-0731 (no thinking)"},
        )

    def test_cost_calculator_can_price_every_variant(self) -> None:
        calculator = CostCalculator()

        self.assertIn(MODEL_ID, calculator.pricing)
        self.assertTrue(all(calculator.get_model_for_player(player_id) == MODEL_ID for player_id in PLAYER_IDS))

    def test_freeze_checker_loads_release_metadata_for_every_variant(self) -> None:
        checker = FreezeChecker.__new__(FreezeChecker)
        checker._publish_dates = {}
        checker._player_providers = {}
        checker._models_by_provider = {}
        checker._player_model_ids = {}
        checker._models_by_model_id = {}

        checker._load_publish_dates()

        self.assertTrue(all(checker._publish_dates[player_id] == 1785456000 for player_id in PLAYER_IDS))
        self.assertEqual(set(checker._models_by_model_id[MODEL_ID]), PLAYER_IDS)


class Gemini38FlashRegistrationTests(unittest.TestCase):
    def test_benchmark_factory_registers_direct_medium_thinking(self) -> None:
        with open(ROOT / "config" / "benchmark.yaml") as config_file:
            config = yaml.safe_load(config_file)

        players, reasoning_ids = create_llm_players(
            config,
            api_key="registration-test",
            api_backend="gemini",
        )

        self.assertIn(GEMINI_38_PLAYER_ID, players)
        self.assertEqual(
            {
                player_id
                for player_id in players
                if player_id.startswith("gemini-3.8-flash")
            },
            {GEMINI_38_PLAYER_ID},
        )
        player = players[GEMINI_38_PLAYER_ID]
        self.assertEqual(player.model_name, "gemini-3.8-flash")
        self.assertEqual(player.reasoning_effort, "medium")
        self.assertIn(GEMINI_38_PLAYER_ID, reasoning_ids)

    def test_cost_calculator_applies_introductory_pricing(self) -> None:
        calculator = CostCalculator()

        self.assertEqual(
            calculator.calculate_game_cost(
                {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
                GEMINI_38_MODEL_ID,
            ),
            4.5,
        )
        self.assertEqual(
            calculator.get_model_for_player(GEMINI_38_PLAYER_ID),
            GEMINI_38_MODEL_ID,
        )

    def test_freeze_checker_loads_release_metadata(self) -> None:
        checker = FreezeChecker.__new__(FreezeChecker)
        checker._publish_dates = {}
        checker._player_providers = {}
        checker._models_by_provider = {}
        checker._player_model_ids = {}
        checker._models_by_model_id = {}

        checker._load_publish_dates()

        self.assertEqual(
            checker._publish_dates.get(GEMINI_38_PLAYER_ID),
            1788307200,
        )
        self.assertEqual(
            checker._models_by_model_id.get(GEMINI_38_MODEL_ID),
            [GEMINI_38_PLAYER_ID],
        )

    def test_stability_analysis_uses_the_38_model_line(self) -> None:
        self.assertEqual(model_family(GEMINI_38_PLAYER_ID), "gemini-3.8")


if __name__ == "__main__":
    unittest.main()
