"""
Leaderboard display and formatting.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from .glicko2 import PlayerRating
from .rating_store import RatingStore
from .fide_estimate import estimate_fide
from .cost_calculator import CostCalculator, filter_results_by_rating_diff
from game.stats_collector import StatsCollector
from game.models import GameResult

# Load model publish dates
_PUBLISH_DATES_PATH = Path(__file__).parent.parent / "data" / "model_publish_dates.json"
_publish_dates: Dict[str, Dict[str, Any]] = {}
try:
    with open(_PUBLISH_DATES_PATH) as f:
        _publish_dates = json.load(f)
except FileNotFoundError:
    logging.warning(f"Model publish dates file not found: {_PUBLISH_DATES_PATH}")
except json.JSONDecodeError as e:
    logging.error(f"Invalid JSON in model publish dates file: {e}")


class Leaderboard:
    """
    Generates and formats leaderboard data.
    """

    def __init__(
        self,
        rating_store: RatingStore,
        stats: Optional[StatsCollector] = None,
        results: Optional[List[GameResult]] = None,
    ):
        """
        Initialize leaderboard.

        Args:
            rating_store: The rating store
            stats: Optional stats collector for additional metrics
            results: Optional list of game results for cost calculation
        """
        self.rating_store = rating_store
        self.stats = stats
        self.results = results or (stats.results if stats else [])
        self._cost_data: Optional[Dict[str, Dict[str, Any]]] = None

    @property
    def cost_data(self) -> Dict[str, Dict[str, Any]]:
        """Lazily calculate cost data (only for games against similarly-rated opponents)."""
        if self._cost_data is None:
            if self.results:
                # Filter to only include games where opponents are within rating threshold
                filtered_results = filter_results_by_rating_diff(
                    self.results, self.rating_store
                )
                calculator = CostCalculator()
                self._cost_data = calculator.calculate_player_costs(filtered_results)
            else:
                self._cost_data = {}
        return self._cost_data

    def get_leaderboard(self, min_games: int = 1, sort_by: str = "rating") -> List[Dict[str, Any]]:
        """
        Get leaderboard data.

        Args:
            min_games: Minimum games to include
            sort_by: Sort field - "rating", "legal", or "cost"

        Returns:
            List of leaderboard entries
        """
        ratings = self.rating_store.get_sorted_ratings(min_games=min_games)
        player_stats = self.stats.get_player_stats() if self.stats else {}

        leaderboard = []
        for i, rating in enumerate(ratings, 1):
            # 95% confidence interval
            ci = 1.96 * rating.rating_deviation

            entry = {
                "rank": i,
                "player_id": rating.player_id,
                "rating": round(rating.rating),
                "fide_estimate": estimate_fide(rating.rating),
                "rating_deviation": round(rating.rating_deviation),
                "confidence_low": round(rating.rating - ci),
                "confidence_high": round(rating.rating + ci),
                "games_played": rating.games_played,
                "is_anchor": self.rating_store.is_anchor(rating.player_id),
                # W-L-D from rating store (single source of truth)
                "wins": rating.wins,
                "losses": rating.losses,
                "draws": rating.draws,
            }

            # Add publish date if available
            if rating.player_id in _publish_dates:
                date_info = _publish_dates[rating.player_id]
                # Format as MM/YY from YYYY-MM-DD
                date_str = date_info.get("created_date", "")
                if date_str and len(date_str) == 10:  # YYYY-MM-DD is always 10 chars
                    parts = date_str.split("-")
                    if len(parts) == 3 and all(p.isdigit() for p in parts):
                        entry["publish_date"] = f"{parts[1]}/{parts[0][-2:]}"  # MM/YY
                        entry["publish_timestamp"] = date_info.get("created_timestamp", 0)

            # Add additional stats if available (legal move rate, forfeit rate, etc.)
            if rating.player_id in player_stats:
                pstats = player_stats[rating.player_id]
                entry.update({
                    "win_rate": pstats.get("win_rate", 0),
                    "legal_move_rate": pstats.get("legal_move_rate", 1.0),
                    "forfeit_rate": pstats.get("forfeit_rate", 0),
                })

            # Add cost data if available
            if rating.player_id in self.cost_data:
                cost_stats = self.cost_data[rating.player_id]
                entry.update({
                    "total_cost": cost_stats.get("total_cost", 0.0),
                    "avg_cost_per_game": cost_stats.get("avg_cost_per_game", 0.0),
                    "games_with_cost": cost_stats.get("games_with_cost", 0),
                })

            leaderboard.append(entry)

        # Sort based on sort_by parameter
        if sort_by == "legal":
            # Sort by legal move rate (desc), then rating (desc) for ties
            # Default to 1.0 (100%) for engines/anchors that don't have stats
            leaderboard.sort(key=lambda e: (-e.get("legal_move_rate", 1.0), -e["rating"]))
        elif sort_by == "cost":
            # Sort by cost (desc), None values last, then rating (desc) for ties
            def cost_sort_key(e):
                cost = e.get("avg_cost_per_game")
                # None values go last (True > False in tuple comparison)
                # Negate cost for descending order (most expensive first)
                return (cost is None, -(cost if cost is not None else 0), -e["rating"])
            leaderboard.sort(key=cost_sort_key)
        else:
            # Default: sort by rating (desc), then by legal move rate (desc) for ties
            leaderboard.sort(key=lambda e: (-e["rating"], -e.get("legal_move_rate", 0)))

        # Re-number ranks after sorting
        for i, entry in enumerate(leaderboard, 1):
            entry["rank"] = i

        return leaderboard

    def format_table(self, min_games: int = 1, sort_by: str = "rating") -> str:
        """
        Format leaderboard as ASCII table.

        Args:
            min_games: Minimum games to include
            sort_by: Sort field - "rating", "legal", or "cost"

        Returns:
            Formatted table string
        """
        lb = self.get_leaderboard(min_games=min_games, sort_by=sort_by)
        if not lb:
            return "No players with enough games."

        # Header
        lines = [
            "=" * 97,
            f"{'Rank':<5} {'Player':<25} {'Rating':<8} {'RD':<5} {'Games':<6} {'W-L-D':<12} {'Legal%':<8} {'$/Game':<10}",
            "=" * 97,
        ]

        for entry in lb:
            player = entry["player_id"]
            if entry.get("is_anchor"):
                player += " *"

            wld = f"{entry.get('wins', '-')}-{entry.get('losses', '-')}-{entry.get('draws', '-')}"
            legal_pct = entry.get("legal_move_rate", 1.0) * 100

            # Show N/A for RD on anchor models (fixed ratings)
            rd_str = "N/A" if entry.get("is_anchor") else str(entry['rating_deviation'])

            # Format cost per game
            avg_cost = entry.get("avg_cost_per_game")
            if avg_cost is not None:
                cost_str = f"${avg_cost:.4f}"
            else:
                cost_str = "-"

            lines.append(
                f"{entry['rank']:<5} {player:<25} {entry['rating']:<8} "
                f"{rd_str:<5} {entry['games_played']:<6} "
                f"{wld:<12} {legal_pct:>6.1f}% {cost_str:>10}"
            )

        lines.append("=" * 97)
        lines.append("* = Anchor (fixed rating)")

        return "\n".join(lines)
