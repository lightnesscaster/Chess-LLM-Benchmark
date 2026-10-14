import asyncio
import io

import chess
import chess.pgn
import pytest

from engines.random_engine import RandomEngine
from game.game_runner import GameRunner
from game.draw_offers import annotate_draw_events
from game.publication import validate_result
from llm.base_llm import BaseLLMPlayer


class DrawPlayer(BaseLLMPlayer):
    def __init__(self, name, moves):
        super().__init__(name, "test/model")
        self.moves = iter(moves)
        self.contexts = []

    async def select_move(self, board, is_retry=False, last_move_illegal=None):
        self.contexts.append((getattr(self, "allow_draw_offers", False),
                              getattr(self, "opponent_draw_offer", False), is_retry))
        return next(self.moves)

    async def close(self):
        pass


def play(white_moves, black_moves, **kwargs):
    white, black = DrawPlayer("white", white_moves), DrawPlayer("black", black_moves)
    result, pgn = asyncio.run(GameRunner(white, black, **kwargs).play_game())
    return result, pgn, white, black


def test_offer_and_accept_is_a_rated_draw_without_an_extra_move():
    result, pgn, white, black = play(["e2e4 offer_draw"], ["accept_draw"])
    assert (result.winner, result.termination, result.moves) == ("draw", "draw_agreement", 1)
    assert (result.total_moves_white, result.total_moves_black) == (1, 0)
    assert result.illegal_moves_white == result.illegal_moves_black == 0
    assert white.contexts == [(True, False, False)]
    assert black.contexts == [(True, True, False)]
    assert "[%draw_offer white]" in pgn and "[%draw_accept black]" in pgn
    validate_result(result, pgn)


def test_unilateral_acceptance_is_invalid_and_uses_existing_retry_policy():
    result, pgn, white, _ = play(["accept_draw", "e2e4", "resign"], ["e7e5"])
    assert result.termination == "resignation"
    assert result.illegal_moves_white == 1
    assert result.retry_recoveries_white == 1
    assert "draw_accept" not in pgn


def test_legal_reply_declines_and_offer_cannot_be_accepted_by_its_sender():
    result, pgn, white, _ = play(["e2e4 offer_draw", "accept_draw", "resign"], ["e7e5"])
    assert result.termination == "resignation"
    assert result.illegal_moves_white == 1
    assert white.contexts[1][1] is False
    assert "[%draw_decline black]" in pgn


def test_counteroffer_declines_previous_offer_then_can_be_accepted():
    result, pgn, _, _ = play(["e2e4 offer_draw", "accept_draw"], ["e7e5 offer_draw"])
    assert (result.termination, result.moves) == ("draw_agreement", 2)
    assert "[%draw_decline black]" in pgn
    assert "[%draw_offer black]" in pgn
    assert "[%draw_accept white]" in pgn
    validate_result(result, pgn)


def test_illegal_retry_preserves_pending_offer_and_can_recover_by_accepting():
    result, _, _, black = play(["e2e4 offer_draw"], ["e2e4 offer_draw", "accept_draw"])
    assert result.termination == "draw_agreement"
    assert result.illegal_moves_black == 1
    assert result.retry_recoveries_black == 1
    assert black.contexts == [(True, True, False), (True, True, True)]


def test_offer_is_disabled_against_engine_without_negotiation_support():
    white = DrawPlayer("white", ["e2e4 offer_draw", "e2e4"])
    engine = RandomEngine(player_id="engine", rating=400, seed=1)
    result, pgn = asyncio.run(GameRunner(white, engine, max_moves=1).play_game())
    assert result.termination == "max_moves"
    assert result.illegal_moves_white == 1
    assert all(not capability for capability, _, _ in white.contexts)
    assert "draw_offer" not in pgn


def test_checkmate_takes_precedence_over_attached_offer():
    result, pgn, _, _ = play([], ["d8h4 offer_draw"], pre_moves=["f2f3", "e7e5", "g2g4"])
    assert (result.winner, result.termination) == ("black", "checkmate")
    assert "draw_offer" not in pgn


@pytest.mark.parametrize("events", [
    [{"action": "accept", "by": "black", "ply": 1}],
    [{"action": "offer", "by": "black", "ply": 1}],
    [{"action": "offer", "by": "white", "ply": 0}],
    [{"action": "offer", "by": "white", "ply": True}],
    [{"action": "offer", "by": "white", "ply": 2}],
    [{"action": "offer", "by": "white", "ply": 1},
     {"action": "accept", "by": "white", "ply": 1}],
    [{"action": "offer", "by": "white", "ply": 1},
     {"action": "decline", "by": "black", "ply": 1},
     {"action": "offer", "by": "white", "ply": 1}],
    [{"action": [], "by": "white", "ply": 1}],
])
def test_pgn_annotation_rejects_invalid_or_unilateral_events(events):
    game = chess.pgn.Game()
    game.add_variation(chess.Move.from_uci("e2e4"))
    with pytest.raises(ValueError):
        annotate_draw_events(game, events)


def test_publication_rejects_agreement_without_matching_opposing_final_tags():
    result, pgn, _, _ = play(["e2e4 offer_draw"], ["accept_draw"])
    for altered in (pgn.replace("[%draw_accept black]", ""),
                    pgn.replace("[%draw_offer white]", "[%draw_offer black]"),
                    pgn.replace("[%draw_accept black]", "[%draw_accept white]")):
        with pytest.raises(ValueError, match="agreement"):
            validate_result(result, altered)


def test_publication_rejects_agreement_at_naturally_finished_position():
    result, pgn, _, _ = play([], ["d8h4 offer_draw"], pre_moves=["f2f3", "e7e5", "g2g4"])
    result.winner, result.termination = "draw", "draw_agreement"
    game = chess.pgn.read_game(io.StringIO(pgn))
    game.headers["Result"] = "1/2-1/2"
    game.headers["Termination"] = "draw_agreement"
    game.end().comment = "[%draw_offer black] [%draw_accept white]"
    with pytest.raises(ValueError, match="agreement"):
        validate_result(result, str(game))


def test_agreed_draw_publication_updates_both_players_only_once():
    from tests.test_publication import Database, Transaction
    from game.publication import commit_result

    result, pgn, _, _ = play(["e2e4 offer_draw"], ["accept_draw"])
    db = Database()
    transaction = Transaction(db)
    first = commit_result(transaction, db, result, pgn, set(), set())
    second = commit_result(transaction, db, result, pgn, set(), set())
    assert not first["already_published"] and second["already_published"]
    for side in ("white", "black"):
        rating = db.docs[("ratings", side)]
        assert rating["games_played"] == rating["draws"] == 1
        assert rating["wins"] == rating["losses"] == 0


def test_live_snapshot_exposes_only_a_valid_pending_offering_side():
    from game.live_games import build_snapshot, sanitize_snapshot

    board = chess.Board()
    board.push_uci("e2e4")
    data = build_snapshot("game", "white", "black", board, started_at=0, draw_offer="white")
    assert data["draw_offer"] == "white"
    assert sanitize_snapshot({**data, "draw_offer": None})["draw_offer"] is None
    for value in ("black", "accept_draw", {}, True):
        with pytest.raises(ValueError, match="draw offer"):
            sanitize_snapshot({**data, "draw_offer": value})
    ended = build_snapshot("game", "white", "black", board, started_at=0,
                           draw_offer="white", status="finished", winner="draw", termination="draw_agreement")
    assert ended["draw_offer"] is None
