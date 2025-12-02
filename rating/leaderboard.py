"""
Leaderboard display and formatting.
"""

from typing import Dict, Any, List, Optional

from .glicko2 import PlayerRating, Glicko2System
from .rating_store import RatingStore
from .fide_estimate import estimate_fide
from game.stats_collector import StatsCollector


class Leaderboard:
    """
    Generates and formats leaderboard data.
    """

    def __init__(self, rating_store: RatingStore, stats: Optional[StatsCollector] = None):
        """
        Initialize leaderboard.

        Args:
            rating_store: The rating store
            stats: Optional stats collector for additional metrics
        """
        self.rating_store = rating_store
        self.stats = stats

    def get_leaderboard(self, min_games: int = 1) -> List[Dict[str, Any]]:
        """
        Get leaderboard data.

        Args:
            min_games: Minimum games to include

        Returns:
            List of leaderboard entries
        """
        ratings = self.rating_store.get_sorted_ratings(min_games=min_games)
        player_stats = self.stats.get_player_stats() if self.stats else {}

        leaderboard = []
        for i, rating in enumerate(ratings, 1):
            # 95% confidence interval (clamped to rating floor)
            ci = 1.96 * rating.rating_deviation
            ci_low = max(rating.rating - ci, Glicko2System.RATING_FLOOR)

            # Use unclamped rating for sorting; fall back to clamped if not available
            unclamped = rating.unclamped_rating if rating.unclamped_rating is not None else rating.rating

            entry = {
                "rank": i,
                "player_id": rating.player_id,
                "rating": round(rating.rating),
                "unclamped_rating": round(unclamped),
                "fide_estimate": estimate_fide(rating.rating),
                "rating_deviation": round(rating.rating_deviation),
                "confidence_low": round(ci_low),
                "confidence_high": round(rating.rating + ci),
                "games_played": rating.games_played,
                "is_anchor": self.rating_store.is_anchor(rating.player_id),
                # W-L-D from rating store (single source of truth)
                "wins": rating.wins,
                "losses": rating.losses,
                "draws": rating.draws,
            }

            # Add additional stats if available (legal move rate, forfeit rate, etc.)
            if rating.player_id in player_stats:
                pstats = player_stats[rating.player_id]
                entry.update({
                    "win_rate": pstats.get("win_rate", 0),
                    "legal_move_rate": pstats.get("legal_move_rate", 1.0),
                    "forfeit_rate": pstats.get("forfeit_rate", 0),
                })

            leaderboard.append(entry)

        # Sort by unclamped rating (desc) - this naturally handles ties at the floor
        leaderboard.sort(key=lambda e: -e["unclamped_rating"])

        # Re-number ranks after sorting
        for i, entry in enumerate(leaderboard, 1):
            entry["rank"] = i

        return leaderboard

    def format_table(self, min_games: int = 1) -> str:
        """
        Format leaderboard as ASCII table.

        Args:
            min_games: Minimum games to include

        Returns:
            Formatted table string
        """
        lb = self.get_leaderboard(min_games=min_games)
        if not lb:
            return "No players with enough games."

        # Header
        lines = [
            "=" * 85,
            f"{'Rank':<5} {'Player':<25} {'Rating':<8} {'RD':<5} {'Games':<6} {'W-L-D':<12} {'Legal%':<8}",
            "=" * 85,
        ]

        for entry in lb:
            player = entry["player_id"]
            if entry.get("is_anchor"):
                player += " *"

            wld = f"{entry.get('wins', '-')}-{entry.get('losses', '-')}-{entry.get('draws', '-')}"
            legal_pct = entry.get("legal_move_rate", 1.0) * 100

            # Show N/A for RD on anchor models (fixed ratings)
            rd_str = "N/A" if entry.get("is_anchor") else str(entry['rating_deviation'])

            lines.append(
                f"{entry['rank']:<5} {player:<25} {entry['rating']:<8} "
                f"{rd_str:<5} {entry['games_played']:<6} "
                f"{wld:<12} {legal_pct:>6.1f}%"
            )

        lines.append("=" * 85)
        lines.append("* = Anchor (fixed rating)")

        return "\n".join(lines)
