import json
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import yaml

from cli import recalculate_ratings
from game.models import GameResult


THINKING_PLAYER = "claude-sonnet-4 (thinking)"
LOWER_EFFORT_PLAYER = "claude-sonnet-4 (no thinking)"


def _loss(game_num: int, player_id: str, anchor_id: str) -> GameResult:
    return GameResult(
        game_id=f"game-{game_num}",
        white_id=player_id,
        black_id=anchor_id,
        winner="black",
        termination="forfeit_illegal_move",
        moves=20,
        illegal_moves_white=1,
        illegal_moves_black=0,
        total_moves_white=10,
        total_moves_black=10,
        pgn_path=f"game-{game_num}.pgn",
        created_at=f"2026-01-01T00:00:{game_num:02d}+00:00",
    )


class RatingRecalculationSeedTests(IsolatedAsyncioTestCase):
    async def test_later_passes_preserve_legality_seed(self) -> None:
        anchors = {
            "random-bot": 400,
            "maia-1100": 1628,
            "maia-1900": 1816,
        }
        results = [
            _loss(1, LOWER_EFFORT_PLAYER, "random-bot"),
            _loss(2, THINKING_PLAYER, "random-bot"),
            _loss(3, THINKING_PLAYER, "random-bot"),
            _loss(4, THINKING_PLAYER, "maia-1100"),
            _loss(5, THINKING_PLAYER, "maia-1100"),
            _loss(6, THINKING_PLAYER, "maia-1900"),
            _loss(7, THINKING_PLAYER, "maia-1900"),
        ]

        self.enterContext(patch("cli.PGNLogger.load_all_results", return_value=results))

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "benchmark.yaml"
            output_path = temp_path / "ratings.json"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "engines": [
                            {
                                "player_id": player_id,
                                "type": "random",
                                "rating": rating,
                            }
                            for player_id, rating in anchors.items()
                        ],
                        "llms": [
                            {
                                "player_id": THINKING_PLAYER,
                                "model_name": "anthropic/claude-sonnet-4",
                                "reasoning": True,
                            },
                            {
                                "player_id": LOWER_EFFORT_PLAYER,
                                "model_name": "anthropic/claude-sonnet-4",
                            },
                        ],
                    }
                )
            )
            args = SimpleNamespace(
                config=str(config_path),
                verbose=False,
                validation_output=output_path,
                validation_seed_rd=166.0,
                validation_disable_benchmark_seeds=True,
            )

            exit_code = await recalculate_ratings(args)
            ratings = json.loads(output_path.read_text())

        thinking = ratings[THINKING_PLAYER]
        self.assertEqual(exit_code, 0)
        self.assertEqual(thinking["losses"], 6)
        self.assertGreater(thinking["rating"], 150)
        self.assertLess(thinking["rating"], 170)
        self.assertGreater(thinking["rating_deviation"], 200)
        self.assertLess(thinking["rating_deviation"], 220)
