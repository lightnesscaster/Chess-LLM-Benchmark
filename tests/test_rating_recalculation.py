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


def _human_challenge(
    human_rd: float,
    *,
    game_id: str = "human-challenge-1",
    created_at: str = "2026-09-02T12:05:00+00:00",
    human_rating: float = 1500,
    winner: str = "black",
) -> GameResult:
    return GameResult(
        game_id=game_id,
        white_id="lichess:Some_Player",
        black_id="test-model",
        winner=winner,
        termination="checkmate",
        moves=40,
        illegal_moves_white=0,
        illegal_moves_black=0,
        total_moves_white=20,
        total_moves_black=20,
        pgn_path=f"{game_id}.pgn",
        created_at=created_at,
        game_type="human_challenge",
        human_email="player@example.com",
        human_lichess_username="Some_Player",
        human_rating=human_rating,
        human_rating_deviation=human_rd,
        human_rating_provisional=human_rd >= 110,
        human_rating_pool="classical",
    )


class RatingRecalculationSeedTests(IsolatedAsyncioTestCase):
    async def test_unrelated_new_games_do_not_reorder_existing_rating_history(self):
        from tempfile import TemporaryDirectory

        games = [_human_challenge(60, game_id=f"historic-{i}",
                 created_at=f"2026-09-02T12:0{i}:00+00:00",
                 human_rating=1000 + 200 * i,
                 winner="white" if i % 2 else "black") for i in range(6)]
        extra = _human_challenge(60, game_id="new-unrelated", created_at="2026-09-03T12:00:00Z")
        extra.black_id = "unrelated-model"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({"engines": [], "llms": []}))
            snapshots = []
            for index, rows in enumerate((games, games + [extra])):
                output = root / f"ratings-{index}.json"
                args = SimpleNamespace(config=str(config), verbose=False,
                    validation_output=output, validation_seed_rd=166.0,
                    validation_disable_benchmark_seeds=True)
                with patch("cli.PGNLogger.load_all_results", return_value=rows):
                    self.assertEqual(await recalculate_ratings(args), 0)
                snapshots.append(json.loads(output.read_text())["test-model"])
            for field in ("rating", "rating_deviation", "games_rd", "games_played", "wins", "losses"):
                self.assertEqual(snapshots[0][field], snapshots[1][field], field)

    async def test_invalid_anchor_history_refuses_replay_before_touching_ratings(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            output = root / "ratings.json"
            original = '{"sentinel": {"player_id": "sentinel", "rating": 1750}}'
            output.write_text(original)
            config_path.write_text(yaml.safe_dump({"engines": [{
                "player_id": "eubos", "rating": 2346,
                "rating_history": [{"before": "invalid", "rating": 2211}],
            }], "llms": []}))
            args = SimpleNamespace(config=str(config_path), verbose=False,
                validation_output=output, validation_seed_rd=166.0,
                validation_disable_benchmark_seeds=True)
            with patch("cli.PGNLogger.load_all_results", return_value=[_loss(1, "test-model", "eubos")]):
                self.assertEqual(await recalculate_ratings(args), 1)
            self.assertEqual(output.read_text(), original)

    async def test_anchor_rating_change_preserves_past_games_in_both_colors(self) -> None:
        from tempfile import TemporaryDirectory

        async def replay(root, rating, history, timestamp, anchor_white):
            game = _loss(1, "test-model", "eubos")
            game.created_at = timestamp
            if anchor_white:
                game.white_id, game.black_id = game.black_id, game.white_id
                game.winner = "white"
            config_path = root / "config.yaml"
            output = root / "ratings.json"
            config_path.write_text(yaml.safe_dump({
                "engines": [{"player_id": "eubos", "rating": rating,
                             "rating_history": history}],
                "llms": [{"player_id": "test-model", "model_name": "test/model"}],
            }))
            args = SimpleNamespace(config=str(config_path), verbose=False,
                validation_output=output, validation_seed_rd=166.0,
                validation_disable_benchmark_seeds=True)
            with patch("cli.PGNLogger.load_all_results", return_value=[game]):
                self.assertEqual(await recalculate_ratings(args), 0)
            return json.loads(output.read_text())

        cutoff = "2026-09-08T13:02:05+00:00"
        history = [{"before": cutoff, "rating": 2211}]
        with TemporaryDirectory() as directory:
            for anchor_white in (False, True):
                with self.subTest(anchor_white=anchor_white):
                    root = Path(directory)
                    old = await replay(root, 2211, [], "2026-09-07T12:00:00Z", anchor_white)
                    preserved = await replay(root, 2346, history, "2026-09-07T12:00:00Z", anchor_white)
                    current = await replay(root, 2346, history, cutoff, anchor_white)
                    for field in ("rating", "rating_deviation", "games_rd", "losses"):
                        self.assertEqual(old["test-model"][field], preserved["test-model"][field])
                    self.assertGreater(current["test-model"]["rating"], preserved["test-model"]["rating"])
                    self.assertEqual(preserved["eubos"]["rating"], 2346)
                    self.assertEqual(preserved["eubos"]["games_played"], 1)

    async def test_human_replay_is_independent_of_storage_stream_order(self) -> None:
        from tempfile import TemporaryDirectory

        games = [
            _human_challenge(
                60,
                game_id="human-a",
                created_at="2026-09-02T12:00:00+00:00",
                human_rating=1200,
                winner="black",
            ),
            _human_challenge(
                90,
                game_id="human-b",
                created_at="2026-09-02T13:00:00+00:00",
                human_rating=1900,
                winner="white",
            ),
        ]

        async def replay(temp_path: Path, ordered_games: list[GameResult], suffix: str) -> dict:
            config_path = temp_path / f"benchmark-{suffix}.yaml"
            output_path = temp_path / f"ratings-{suffix}.json"
            config_path.write_text(yaml.safe_dump({
                "engines": [{"player_id": "anchor", "type": "random", "rating": 1200}],
                "llms": [{"player_id": "test-model", "model_name": "provider/test-model"}],
            }))
            args = SimpleNamespace(
                config=str(config_path),
                verbose=False,
                validation_output=output_path,
                validation_seed_rd=166.0,
                validation_disable_benchmark_seeds=True,
            )
            with patch("cli.PGNLogger.load_all_results", return_value=ordered_games):
                self.assertEqual(await recalculate_ratings(args), 0)
            return json.loads(output_path.read_text())["test-model"]

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            forward = await replay(temp_path, games, "forward")
            reversed_result = await replay(temp_path, list(reversed(games)), "reversed")

        self.assertEqual(forward, reversed_result)

    async def test_human_snapshot_updates_only_model_and_uses_stored_rd(self) -> None:
        from tempfile import TemporaryDirectory

        async def recalculate_with_rd(temp_path: Path, human_rd: float) -> dict:
            config_path = temp_path / f"benchmark-{human_rd}.yaml"
            output_path = temp_path / f"ratings-{human_rd}.json"
            config_path.write_text(yaml.safe_dump({
                "engines": [{"player_id": "anchor", "type": "random", "rating": 1200}],
                "llms": [{"player_id": "test-model", "model_name": "provider/test-model"}],
            }))
            args = SimpleNamespace(
                config=str(config_path),
                verbose=False,
                validation_output=output_path,
                validation_seed_rd=166.0,
                validation_disable_benchmark_seeds=True,
            )
            with patch("cli.PGNLogger.load_all_results", return_value=[_human_challenge(human_rd)]):
                exit_code = await recalculate_ratings(args)
            self.assertEqual(exit_code, 0)
            return json.loads(output_path.read_text())

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            certain = await recalculate_with_rd(temp_path, 45)
            uncertain = await recalculate_with_rd(temp_path, 300)

        self.assertNotIn("lichess:Some_Player", certain)
        self.assertEqual(certain["test-model"]["games_played"], 1)
        self.assertEqual(certain["test-model"]["wins"], 1)
        self.assertGreater(certain["test-model"]["rating"], uncertain["test-model"]["rating"])

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
