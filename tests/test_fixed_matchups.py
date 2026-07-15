import asyncio
from types import SimpleNamespace
import unittest

from scripts.run_fixed_matchups import (
    build_missing_games,
    close_game_players,
    create_game_players,
    index_player_configs,
    MatchupPlan,
    normalize_matchups,
    PlannedGame,
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

    def test_per_matchup_game_targets_share_one_resumable_plan(self) -> None:
        matchups = normalize_matchups(
            [
                ["new", "peer"],
                {"players": ["new", "anchor"], "games_per_color": 2},
            ],
            default_games_per_color=1,
        )
        existing = [
            SimpleNamespace(white_id="new", black_id="anchor"),
            SimpleNamespace(white_id="anchor", black_id="new"),
        ]

        games = build_missing_games(matchups, existing)

        self.assertEqual(
            matchups,
            [
                MatchupPlan("new", "peer", 1),
                MatchupPlan("new", "anchor", 2),
            ],
        )
        self.assertEqual(
            [(game.white_id, game.black_id) for game in games],
            [
                ("new", "peer"),
                ("peer", "new"),
                ("new", "anchor"),
                ("anchor", "new"),
            ],
        )

    def test_selected_player_configs_accepts_codex_llm_and_engine(self) -> None:
        source = {
            "llms": [{"player_id": "model (high)", "model_name": "m", "api": "codex"}],
            "engines": [{"player_id": "random-bot", "type": "random", "rating": 400}],
        }

        llms, engines = selected_player_configs(source, {"model (high)", "random-bot"})

        self.assertEqual([config["player_id"] for config in llms], ["model (high)"])
        self.assertEqual([config["player_id"] for config in engines], ["random-bot"])

    def test_create_game_players_returns_isolated_instances(self) -> None:
        llm_configs, engine_configs = index_player_configs(
            [
                {
                    "player_id": "model (high)",
                    "model_name": "openai/gpt-test",
                    "api": "codex",
                    "reasoning_effort": "high",
                }
            ],
            [{"player_id": "random-bot", "type": "random", "rating": 400}],
        )
        game = PlannedGame("model (high)", "random-bot")

        first = create_game_players(game, llm_configs, engine_configs)
        second = create_game_players(game, llm_configs, engine_configs)

        self.assertIsNot(first[0], second[0])
        self.assertIsNot(first[1], second[1])
        self.assertEqual((first[0].player_id, first[1].player_id), ("model (high)", "random-bot"))
        self.assertEqual((second[0].player_id, second[1].player_id), ("model (high)", "random-bot"))

        asyncio.run(close_game_players(*first))
        asyncio.run(close_game_players(*second))


if __name__ == "__main__":
    unittest.main()
