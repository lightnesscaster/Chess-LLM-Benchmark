from pathlib import Path
import unittest

import yaml

from cli import create_llm_players
from game.freeze_checker import FreezeChecker
from rating.cost_calculator import CostCalculator


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
PLAYER_IDS = {
    "deepseek-v4-flash-0731 (no thinking)",
    "deepseek-v4-flash-0731 (high)",
    "deepseek-v4-flash-0731 (max)",
}


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


if __name__ == "__main__":
    unittest.main()
