"""
Statistics collection and aggregation.
"""

from collections import defaultdict
from typing import Dict, List, Any

from .models import GameResult


class StatsCollector:
    """
    Collects and aggregates statistics from game results.
    """

    def __init__(self):
        self.results: List[GameResult] = []

    def add_result(self, result: GameResult) -> None:
        """Add a game result."""
        self.results.append(result)

    def add_results(self, results: List[GameResult]) -> None:
        """Add multiple game results."""
        self.results.extend(results)

    def get_player_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate per-player statistics.

        Returns:
            Dict mapping player_id to stats dict
        """
        stats = defaultdict(lambda: {
            "games_played": 0,
            "games_as_white": 0,
            "games_as_black": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "wins_as_white": 0,
            "wins_as_black": 0,
            "forfeits": 0,
            "illegal_moves": 0,
            "total_moves": 0,
            "total_game_moves": 0,  # Sum of game lengths
            "opponents": set(),
        })

        for result in self.results:
            white_id = result.white_id
            black_id = result.black_id

            # Update white player stats
            stats[white_id]["games_played"] += 1
            stats[white_id]["games_as_white"] += 1
            stats[white_id]["illegal_moves"] += result.illegal_moves_white
            stats[white_id]["total_moves"] += result.total_moves_white
            stats[white_id]["total_game_moves"] += result.moves
            stats[white_id]["opponents"].add(black_id)

            # Update black player stats
            stats[black_id]["games_played"] += 1
            stats[black_id]["games_as_black"] += 1
            stats[black_id]["illegal_moves"] += result.illegal_moves_black
            stats[black_id]["total_moves"] += result.total_moves_black
            stats[black_id]["total_game_moves"] += result.moves
            stats[black_id]["opponents"].add(white_id)

            # Determine winner/loser
            if result.winner == "white":
                stats[white_id]["wins"] += 1
                stats[white_id]["wins_as_white"] += 1
                stats[black_id]["losses"] += 1
            elif result.winner == "black":
                stats[black_id]["wins"] += 1
                stats[black_id]["wins_as_black"] += 1
                stats[white_id]["losses"] += 1
            else:  # draw
                stats[white_id]["draws"] += 1
                stats[black_id]["draws"] += 1

            # Track forfeits
            if "forfeit" in result.termination:
                if result.winner == "black":
                    stats[white_id]["forfeits"] += 1
                elif result.winner == "white":
                    stats[black_id]["forfeits"] += 1

        # Calculate derived stats
        for player_id, player_stats in stats.items():
            games = player_stats["games_played"]
            if games > 0:
                player_stats["win_rate"] = player_stats["wins"] / games
                player_stats["loss_rate"] = player_stats["losses"] / games
                player_stats["draw_rate"] = player_stats["draws"] / games
                player_stats["forfeit_rate"] = player_stats["forfeits"] / games
                player_stats["avg_game_length"] = player_stats["total_game_moves"] / games

            total_moves = player_stats["total_moves"]
            illegal = player_stats["illegal_moves"]
            if total_moves > 0:
                player_stats["legal_move_rate"] = 1 - (illegal / total_moves)
            else:
                player_stats["legal_move_rate"] = 1.0

            # Convert set to list for JSON serialization
            player_stats["opponents"] = list(player_stats["opponents"])
            player_stats["num_opponents"] = len(player_stats["opponents"])

        return dict(stats)

    def get_head_to_head(self, player_a: str, player_b: str) -> Dict[str, Any]:
        """
        Get head-to-head stats between two players.

        Returns:
            Dict with h2h statistics
        """
        h2h_games = [
            r for r in self.results
            if (r.white_id == player_a and r.black_id == player_b) or
               (r.white_id == player_b and r.black_id == player_a)
        ]

        if not h2h_games:
            return {
                "games": 0,
                "player_a_wins": 0,
                "player_b_wins": 0,
                "draws": 0,
            }

        a_wins = 0
        b_wins = 0
        draws = 0

        for game in h2h_games:
            if game.winner == "white":
                if game.white_id == player_a:
                    a_wins += 1
                else:
                    b_wins += 1
            elif game.winner == "black":
                if game.black_id == player_a:
                    a_wins += 1
                else:
                    b_wins += 1
            else:
                draws += 1

        return {
            "games": len(h2h_games),
            "player_a_id": player_a,
            "player_b_id": player_b,
            "player_a_wins": a_wins,
            "player_b_wins": b_wins,
            "draws": draws,
            "player_a_score": a_wins + 0.5 * draws,
            "player_b_score": b_wins + 0.5 * draws,
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get overall benchmark summary.

        Returns:
            Summary statistics dict
        """
        total_games = len(self.results)

        terminations = defaultdict(int)
        for result in self.results:
            terminations[result.termination] += 1

        return {
            "total_games": total_games,
            "terminations": dict(terminations),
            "unique_players": len(self.get_player_stats()),
        }
