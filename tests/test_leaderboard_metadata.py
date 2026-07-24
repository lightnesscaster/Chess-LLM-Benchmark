from types import SimpleNamespace
import unittest

from rating.leaderboard import Leaderboard


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


class LeaderboardMetadataTests(unittest.TestCase):
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
