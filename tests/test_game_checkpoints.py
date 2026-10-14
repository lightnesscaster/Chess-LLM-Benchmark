import asyncio
import json
import tempfile
import unittest
import threading
from unittest.mock import patch
from pathlib import Path

from game.game_runner import GameRunner
from llm.base_llm import BaseLLMPlayer
from engines.uci_engine import UCIEngine
from engines.maia_engine import MaiaEngine
import chess
import chess.engine


class CheckpointPlayer(BaseLLMPlayer):
    def __init__(self, player_id, responses):
        super().__init__(player_id, "checkpoint-test")
        self.responses = iter(responses)
        self.calls = []

    async def select_move(self, board, is_retry=False, last_move_illegal=None):
        self.calls.append((board.fen(), is_retry, last_move_illegal))
        action = next(self.responses)
        if isinstance(action, BaseException):
            raise action
        if getattr(self, "debug_payload", False):
            self.last_prompt = "private prompt"
            self.last_raw_response = "private response"
        self.prompt_tokens += 10
        self.total_tokens += 10
        self.move_times.append(2.0)
        self.total_move_time += 2.0
        return action

    async def close(self):
        pass


class RunnerCheckpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_changed_maia_binary_rejects_resume_without_engine_start(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(asyncio.CancelledError):
                await GameRunner(
                    CheckpointPlayer("white", [asyncio.CancelledError()]),
                    MaiaEngine("maia", 1900, lc0_path="original", weights_path="weights"),
                    checkpoint_dir=directory).play_game()
            path = next(Path(directory).glob("*.json"))
            changed = MaiaEngine("maia", 1900, lc0_path="changed", weights_path="weights")
            with self.assertRaisesRegex(ValueError, "configuration does not match"):
                await GameRunner(CheckpointPlayer("white", []), changed,
                                 resume_checkpoint=path).play_game()
            self.assertIsNone(changed._engine)

    async def test_completed_checkpoint_and_replayed_result_exclude_raw_debug_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            white = CheckpointPlayer("white", ["a1a1", "e2e4"])
            white.debug_payload = True
            result, _ = await GameRunner(white, CheckpointPlayer("black", []),
                                        max_moves=1, checkpoint_dir=directory).play_game()
            path = next(Path(directory).glob("*.json"))
            self.assertNotIn("private prompt", path.read_text())
            self.assertNotIn("private response", path.read_text())
            replayed, _ = await GameRunner(CheckpointPlayer("white", []),
                                          CheckpointPlayer("black", []),
                                          resume_checkpoint=path).play_game()
            self.assertEqual(replayed, result)
            self.assertEqual(result.illegal_move_details[0]["parsed_move"], "a1a1")
            self.assertNotIn("prompt", result.illegal_move_details[0])

    async def test_engine_search_heartbeat_keeps_spent_time_without_increment(self):
        class SlowSearch:
            def __init__(self):
                self.release = threading.Event()

            def play(self, board, limit):
                self.release.wait(2)
                return chess.engine.PlayResult(next(iter(board.legal_moves)), None)

            def quit(self):
                self.release.set()

        with tempfile.TemporaryDirectory() as directory:
            engine = UCIEngine("clock", 2000, "test-engine", initial_time=100,
                               increment=3, time_scale=0.5)
            search = SlowSearch()
            engine._engine = search
            runner = GameRunner(engine, CheckpointPlayer("black", []), checkpoint_dir=directory)
            runner._checkpoint_interval = 0.02
            task = asyncio.create_task(runner.play_game())
            await asyncio.sleep(0.075)
            path = next(Path(directory).glob("*.json"))
            state = json.loads(path.read_text())
            clock = state["players"]["white"]["clock"]
            self.assertEqual(state["moves"], [])
            self.assertLess(clock["white_clock"], 49.98)
            self.assertEqual(clock["black_clock"], 50.0)
            self.assertEqual(clock["increment"], 1.5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            restored = UCIEngine("clock", 2000, "test-engine", initial_time=100,
                                 increment=3, time_scale=0.5)
            # Resume to a one-ply limit using the checkpoint's saved move limit.
            state["max_moves"] = 1
            path.write_text(json.dumps(state))
            class ImmediateSearch:
                def play(self, board, limit):
                    self.limit = limit
                    return chess.engine.PlayResult(chess.Move.from_uci("e2e4"), None)
            immediate = ImmediateSearch()
            restored._engine = immediate
            result, _ = await GameRunner(restored, CheckpointPlayer("black", []),
                                        resume_checkpoint=path).play_game()
            self.assertEqual(result.moves, 1)
            self.assertAlmostEqual(immediate.limit.white_clock, clock["white_clock"])
            self.assertEqual(immediate.limit.white_inc, 1.5)

    async def test_checkpoint_write_failure_stops_without_forfeit(self):
        from game.checkpoints import CheckpointStore, CheckpointError

        with tempfile.TemporaryDirectory() as directory:
            white = CheckpointPlayer("white", ["e2e4"])
            with patch.object(CheckpointStore, "write", side_effect=CheckpointError("disk full")):
                with self.assertRaises(CheckpointError):
                    await GameRunner(white, CheckpointPlayer("black", []),
                                     checkpoint_dir=directory).play_game()
            self.assertEqual(white.calls, [])

    async def test_resume_preserves_game_moves_accounting_draw_and_ratings(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = GameRunner(
                CheckpointPlayer("white", ["e2e4 offer_draw"]),
                CheckpointPlayer("black", [asyncio.CancelledError()]),
                checkpoint_dir=directory, rating_snapshots={"white": {"rating": 1900}},
            )
            with self.assertRaises(asyncio.CancelledError):
                await runner.play_game()
            path = next(Path(directory).glob("*.json"))
            state = json.loads(path.read_text())
            black = CheckpointPlayer("black", ["accept_draw"])
            resumed = GameRunner(CheckpointPlayer("white", []), black,
                                 resume_checkpoint=path)
            result, pgn = await resumed.play_game()
            self.assertEqual(result.game_id, state["game_id"])
            self.assertEqual(result.moves, 1)
            self.assertEqual(result.tokens_white["total_tokens"], 10)
            self.assertEqual(result.accounting_status_black, "partial")
            self.assertEqual(result.termination, "draw_agreement")
            self.assertEqual(resumed.rating_snapshots, {"white": {"rating": 1900}})
            self.assertIn("[%draw_offer white]", pgn)
            self.assertIn("[%draw_accept black]", pgn)
            # Finished checkpoints return the same result without another request.
            again, same_pgn = await GameRunner(
                CheckpointPlayer("white", []), CheckpointPlayer("black", []),
                resume_checkpoint=path).play_game()
            self.assertEqual(again, result)
            self.assertEqual(same_pgn, pgn)

    async def test_interrupted_retry_keeps_strikes_and_retry_context(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(asyncio.CancelledError):
                await GameRunner(CheckpointPlayer("white", ["a1a1", asyncio.CancelledError()]),
                                 CheckpointPlayer("black", []), checkpoint_dir=directory).play_game()
            path = next(Path(directory).glob("*.json"))
            white = CheckpointPlayer("white", ["a1a1"])
            result, _ = await GameRunner(white, CheckpointPlayer("black", []),
                                        resume_checkpoint=path).play_game()
            self.assertTrue(white.calls[0][1])
            self.assertEqual(white.calls[0][2], "a1a1")
            self.assertEqual(result.illegal_moves_white, 2)
            self.assertEqual(result.retry_attempts_white, 1)
            self.assertEqual(result.retry_failures_white, 1)

    async def test_mismatched_identity_or_history_fails_before_request(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(asyncio.CancelledError):
                await GameRunner(CheckpointPlayer("white", [asyncio.CancelledError()]),
                                 CheckpointPlayer("black", []), checkpoint_dir=directory).play_game()
            path = next(Path(directory).glob("*.json"))
            white = CheckpointPlayer("other", [])
            with self.assertRaises(ValueError):
                await GameRunner(white, CheckpointPlayer("black", []), resume_checkpoint=path).play_game()
            self.assertEqual(white.calls, [])
            state = json.loads(path.read_text())
            state["fen"] = "incorrect"
            path.write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                await GameRunner(CheckpointPlayer("white", []), CheckpointPlayer("black", []),
                                 resume_checkpoint=path).play_game()


class CheckpointStorageTests(unittest.TestCase):
    def test_failed_write_leaves_previous_checkpoint_readable(self):
        from game.checkpoints import CheckpointStore, CheckpointError

        with tempfile.TemporaryDirectory() as directory:
            with CheckpointStore(Path(directory) / "game.json") as store:
                store.write({"value": "previous"})
                with patch("game.checkpoints.os.replace", side_effect=OSError("disk full")):
                    with self.assertRaises(CheckpointError):
                        store.write({"value": "new"})
                self.assertEqual(store.read(), {"value": "previous"})
                self.assertEqual(list(Path(directory).glob(".checkpoint-*")), [])

    def test_atomic_private_write_and_exclusive_lock(self):
        from game.checkpoints import CheckpointStore

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.json"
            with CheckpointStore(path) as store:
                store.write({"version": 1, "value": "durable"})
                self.assertEqual(store.read()["value"], "durable")
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                with self.assertRaises(RuntimeError):
                    with CheckpointStore(path):
                        pass
            with CheckpointStore(path) as store:
                self.assertEqual(store.read()["value"], "durable")
