from types import SimpleNamespace
import unittest
import pytest

from rating.leaderboard import Leaderboard
from game.models import GameResult


class FakeRatingStore:
    def get_sorted_ratings(self, min_games: int = 1):
        return [
            SimpleNamespace(
                player_id="gemini-3.6-flash (medium)",
                rating=2062.8,
                rating_deviation=123.7,
                games_played=15,
                is_frozen=False,
                wins=14,
                losses=0,
                draws=1,
            )
        ]

    def is_anchor(self, _player_id: str) -> bool:
        return False


def test_retired_model_keeps_history_and_is_labelled_on_leaderboard(monkeypatch):
    store = FakeRatingStore()
    rating = store.get_sorted_ratings()[0]
    rating.player_id = "grok-4.1-fast"
    store.get_sorted_ratings = lambda **_: [rating]
    entry = Leaderboard(store).get_leaderboard()[0]
    assert entry["retired"] is True
    assert entry["games_played"] == 15
    import web.app as web_app
    monkeypatch.setattr(web_app, "get_leaderboard_data", lambda **_: [entry])
    with web_app.app.test_client() as client:
        html = client.get("/leaderboard").get_data(as_text=True)
    assert "Retired</span>" in html
    assert "grok-4.1-fast" in html


@pytest.mark.parametrize("effort", ["medium", "high"])
def test_gemini38_variants_have_release_dates_and_medium_has_labelled_estimate(effort, monkeypatch):
    store = FakeRatingStore()
    rating = store.get_sorted_ratings()[0]
    rating.player_id = f"gemini-3.8-flash ({effort})"
    store.get_sorted_ratings = lambda **_: [rating]
    entry = Leaderboard(store).get_leaderboard()[0]
    assert entry["publish_date"] == "09/26"
    assert entry["publish_timestamp"] == 1788307200
    if effort == "medium":
        assert entry["avg_cost_per_game"] > 0
        assert entry["cost_estimated"] is True
        assert "benchmark" in entry["cost_basis"]
    import web.app as web_app
    monkeypatch.setattr(web_app, "get_leaderboard_data", lambda **_: [entry])
    with web_app.app.test_client() as client:
        response = client.get("/leaderboard")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'class="released">09/26</td>' in html
    if effort == "medium":
        assert "~$0.0868</td>" in html
        assert 'title="Estimate, not recorded spend:' in html


@pytest.mark.parametrize("status", ["complete", "partial"])
def test_recorded_usage_replaces_estimate_but_partial_usage_is_labelled(status, monkeypatch):
    store = FakeRatingStore()
    rating = store.get_sorted_ratings()[0]
    rating.player_id = "gemini-3.8-flash (medium)"
    store.get_sorted_ratings = lambda **_: [rating]
    store.get = lambda _: rating
    result = GameResult(
        game_id="cost-test", white_id=rating.player_id, black_id="human",
        winner="black", termination="resignation", moves=2,
        illegal_moves_white=0, illegal_moves_black=0,
        total_moves_white=1, total_moves_black=1, pgn_path="", created_at="2026-09-08",
        tokens_white={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
        accounting_status_white=status,
    )
    board = Leaderboard(store, results=[result])
    entry = board.get_leaderboard()[0]
    assert entry["avg_cost_per_game"] == pytest.approx(0.000525)
    assert not entry.get("cost_estimated")
    assert bool(entry.get("cost_lower_bound")) is (status == "partial")
    if status == "partial":
        assert "missing" in entry["cost_basis"]
        assert "≥$0.0005" in board.format_table()
        import web.app as web_app
        monkeypatch.setattr(web_app, "get_leaderboard_data", lambda **_: [entry])
        with web_app.app.test_client() as client:
            response = client.get("/leaderboard")
        assert response.status_code == 200
        assert "≥$0.0005</td>" in response.get_data(as_text=True)


class LeaderboardMetadataTests(unittest.TestCase):
    def test_exposes_resignation_rate_separately_from_forfeits(self) -> None:
        stats = SimpleNamespace(results=[], get_player_stats=lambda: {
            "gemini-3.6-flash (medium)": {
                "legal_move_rate": 1.0,
                "forfeit_rate": 0.1,
                "resignation_rate": 0.25,
            }
        })

        entry = Leaderboard(FakeRatingStore(), stats=stats).get_leaderboard()[0]

        self.assertEqual(entry["forfeit_rate"], 0.1)
        self.assertEqual(entry["resignation_rate"], 0.25)

    def test_estimated_cost_and_release_date_fill_missing_game_accounting(
        self,
    ) -> None:
        entry = Leaderboard(FakeRatingStore()).get_leaderboard()[0]

        self.assertEqual(entry["publish_date"], "07/26")
        self.assertEqual(entry["avg_cost_per_game"], 0.174)
        self.assertTrue(entry["cost_estimated"])
        self.assertIn("31 retained full-token", entry["cost_basis"])


if __name__ == "__main__":
    unittest.main()
