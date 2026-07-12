from types import SimpleNamespace
import unittest

from scripts.run_fixed_matchups import (
    build_missing_games,
    normalize_matchups,
    selected_player_configs,
)


class FixedMatchupTests(unittest.TestCase):
    def test_build_missing_games_balances_colors_and_resumes(self) -> None:
        existing = [SimpleNamespace(white_id="new", black_id="anchor")]

        games = build_missing_games([("new", "anchor")], existing, games_per_color=2)

        self.assertEqual(
            [(game.white_id, game.black_id) for game in games],
            [
                ("new", "anchor"),
                ("anchor", "new"),
                ("anchor", "new"),
            ],
        )

    def test_normalize_matchups_rejects_reversed_duplicate(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate matchup"):
            normalize_matchups([["a", "b"], ["b", "a"]])

    def test_selected_player_configs_accepts_codex_llm_and_engine(self) -> None:
        source = {
            "llms": [{"player_id": "model (high)", "model_name": "m", "api": "codex"}],
            "engines": [{"player_id": "random-bot", "type": "random", "rating": 400}],
        }

        llms, engines = selected_player_configs(source, {"model (high)", "random-bot"})

        self.assertEqual([config["player_id"] for config in llms], ["model (high)"])
        self.assertEqual([config["player_id"] for config in engines], ["random-bot"])


if __name__ == "__main__":
    unittest.main()
