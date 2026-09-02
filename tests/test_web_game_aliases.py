from types import SimpleNamespace

import pytest

import web.app as web_app


class FakePGNLogger:
    def __init__(self, result):
        self.result = result

    def load_all_results(self):
        return [self.result]

    def load_result(self, game_id):
        return self.result if game_id == self.result.game_id else None

    def load_pgn(self, game_id):
        if game_id != self.result.game_id:
            return None
        return (
            f'[White "{self.result.white_id}"]\n'
            f'[Black "{self.result.black_id}"]\n\n1. e4 e5 *'
        )


def _human_game(human_color="white"):
    model_id = "claude-fable-5.1"
    human_id = "lichess:ladyjustice"
    return SimpleNamespace(
        game_id="human-game-1",
        white_id=human_id if human_color == "white" else model_id,
        black_id=human_id if human_color == "black" else model_id,
        winner="black",
        termination="resignation",
        moves=18,
        illegal_moves_white=0,
        illegal_moves_black=0,
        created_at="2026-09-02T07:00:00+00:00",
        game_type="human_challenge",
        human_email="johnstondaniel4@gmail.com",
        human_lichess_username="ladyjustice",
    )


def test_game_library_uses_configured_human_display_alias(monkeypatch):
    result = _human_game()
    monkeypatch.setenv(
        "HUMAN_GAME_DISPLAY_ALIASES",
        '{"johnstondaniel4@gmail.com":"the39clues"}',
    )
    monkeypatch.setattr(web_app, "PGNLogger", lambda: FakePGNLogger(result))
    monkeypatch.setattr(web_app, "_games_cache", [])
    monkeypatch.setattr(web_app, "_games_cache_time", 0)
    monkeypatch.setattr(web_app, "_games_refreshing", False)

    games = web_app.get_all_games()

    assert games[0]["white"] == "the39clues"
    assert result.white_id == "lichess:ladyjustice"


@pytest.mark.parametrize("human_color", ["white", "black"])
def test_game_detail_uses_same_human_display_alias(monkeypatch, human_color):
    result = _human_game(human_color)
    monkeypatch.setenv(
        "HUMAN_GAME_DISPLAY_ALIASES",
        '{"johnstondaniel4@gmail.com":"the39clues"}',
    )
    monkeypatch.setattr(web_app, "PGNLogger", lambda: FakePGNLogger(result))

    game = web_app.get_game(result.game_id)

    human_side = "white" if human_color == "white" else "black"
    model_side = "black" if human_color == "white" else "white"
    assert game[human_side] == "the39clues"
    assert game[model_side] == "claude-fable-5.1"
    assert "lichess:ladyjustice" in game["pgn"]
