import asyncio

import chess
import pytest

from llm.base_llm import BaseLLMPlayer, request_llm_move
from llm.prompts import build_chess_prompt


CONTEXT = {
    "self": {"rating": 1303, "rating_deviation": 90, "source": "ChessBench", "provisional": True},
    "opponent": {"rating": 2329, "rating_deviation": 141, "source": "Lichess classical", "provisional": False},
}


@pytest.mark.parametrize("retry", [False, True])
def test_ratings_are_labeled_in_normal_and_retry_prompts(retry):
    prompt = build_chess_prompt(chess.Board(), is_retry=retry, illegal_move="a1a1",
                                allow_resignation=True, rating_context=CONTEXT)
    assert "Your rating: 1303" in prompt
    assert "Opponent rating: 2329" in prompt
    assert "ChessBench" in prompt and "Lichess classical" in prompt
    assert "RD 90" in prompt and "provisional" in prompt
    assert "Do not resign merely because" in prompt
    assert "not directly interchangeable" in prompt


def test_rating_context_does_not_change_position_probe_prompt():
    assert build_chess_prompt(chess.Board(), rating_context=CONTEXT) == build_chess_prompt(chess.Board())


def test_unknown_ratings_are_not_fabricated():
    prompt = build_chess_prompt(chess.Board(), allow_resignation=True,
                                rating_context={"self": None, "opponent": None})
    assert "Your rating: unknown" in prompt and "Opponent rating: unknown" in prompt


def test_request_context_is_scoped_even_after_provider_failure():
    class Player(BaseLLMPlayer):
        async def select_move(self, board, **kwargs):
            self.last_prompt = build_chess_prompt(board, allow_resignation=self.allow_resignation,
                                                  rating_context=self.rating_context)
            raise RuntimeError("provider failed")
        async def close(self):
            pass
    player = Player("test", "test")
    with pytest.raises(RuntimeError):
        asyncio.run(request_llm_move(player, chess.Board(), is_retry=False,
                                     last_move_illegal=None, allow_resignation=True,
                                     rating_context=CONTEXT))
    assert "Opponent rating: 2329" in player.last_prompt
    assert player.rating_context is None


def test_completion_prompt_uses_real_context_not_fictional_elos():
    from llm.openrouter_completion_client import OpenRouterCompletionPlayer
    player = OpenRouterCompletionPlayer("test", "test", api_key="test")
    player.rating_context = CONTEXT
    board = chess.Board()
    for with_history in (False, True):
        if with_history:
            board.push_uci("e2e4")
        prompt = player._build_prompt(board, allow_resignation=True)
        assert "Your rating: 1303" in prompt
        assert "Opponent rating: 2329" in prompt
        assert "2885" not in prompt and "2812" not in prompt


def test_runner_assigns_ratings_from_each_players_perspective():
    from game.game_runner import GameRunner
    class Player(BaseLLMPlayer):
        async def select_move(self, board, **kwargs):
            self.last_prompt = build_chess_prompt(board, allow_resignation=self.allow_resignation,
                                                  rating_context=self.rating_context)
            return "e2e4" if board.turn else "e7e5"
        async def close(self):
            pass
    white, black = Player("white", "test"), Player("black", "test")
    runner = GameRunner(white, black, max_moves=2,
                        rating_snapshots={"white": CONTEXT["self"], "black": CONTEXT["opponent"]})
    asyncio.run(runner.play_game())
    assert "Your rating: 1303" in white.last_prompt
    assert "Opponent rating: 2329" in white.last_prompt
    assert "Your rating: 2329" in black.last_prompt
    assert "Opponent rating: 1303" in black.last_prompt


def test_snapshot_does_not_create_unknown_player_and_labels_seed(tmp_path):
    from rating.rating_store import RatingStore
    from rating.glicko2 import PlayerRating
    from rating.prompt_context import rating_snapshot
    store = RatingStore(str(tmp_path / "ratings.json"), use_firestore=False,
                        use_benchmark_predictions=False)
    assert rating_snapshot(store, "unknown") is None
    assert not store.has_player("unknown")
    store.set(PlayerRating("model", rating=1303, rating_deviation=166), auto_save=False)
    snapshot = rating_snapshot(store, "model")
    assert snapshot["rating"] == 1303
    assert snapshot["provisional"] is True


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_cli_provider_prompt_includes_rating_context(backend):
    from llm.codex_subagent_client import CodexSubagentPlayer
    from llm.claude_code_client import ClaudeCodePlayer
    cls = CodexSubagentPlayer if backend == "codex" else ClaudeCodePlayer
    player = cls("test", "test", reasoning_effort="low")
    player.rating_context = CONTEXT
    assert "Opponent rating: 2329" in player._build_prompt(chess.Board(), False, None, allow_resignation=True)
