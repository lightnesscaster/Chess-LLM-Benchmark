"""
PGN and game result logging.
"""

import os
import json
from pathlib import Path
from typing import Optional

from .models import GameResult


class PGNLogger:
    """
    Handles saving PGN files and game result JSON files.
    """

    def __init__(
        self,
        games_dir: str = "data/games",
        results_dir: str = "data/results",
    ):
        """
        Initialize the logger.

        Args:
            games_dir: Directory to save PGN files
            results_dir: Directory to save result JSON files
        """
        self.games_dir = Path(games_dir)
        self.results_dir = Path(results_dir)

        # Ensure directories exist
        self.games_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_game(self, result: GameResult, pgn_str: str) -> GameResult:
        """
        Save a game's PGN and result.

        Args:
            result: The GameResult object
            pgn_str: The PGN string

        Returns:
            Updated GameResult with pgn_path set
        """
        game_id = result.game_id

        # Save PGN
        pgn_path = self.games_dir / f"{game_id}.pgn"
        with open(pgn_path, "w") as f:
            f.write(pgn_str)

        # Update result with path
        result.pgn_path = str(pgn_path)

        # Save result JSON
        result_path = self.results_dir / f"{game_id}.json"
        with open(result_path, "w") as f:
            json.dump(result.to_json(), f, indent=2)

        return result

    def load_result(self, game_id: str) -> Optional[GameResult]:
        """
        Load a game result by ID.

        Args:
            game_id: The game ID

        Returns:
            GameResult or None if not found
        """
        result_path = self.results_dir / f"{game_id}.json"
        if not result_path.exists():
            return None

        with open(result_path) as f:
            data = json.load(f)

        return GameResult.from_json(data)

    def load_all_results(self, verbose: bool = False) -> list[GameResult]:
        """
        Load all game results.

        Args:
            verbose: If True, print warnings for corrupted files

        Returns:
            List of GameResult objects (skips corrupted files)
        """
        results = []
        for path in self.results_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                results.append(GameResult.from_json(data))
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Warning: Skipping corrupted JSON file {path.name}: {e}")
            except (KeyError, TypeError, ValueError) as e:
                if verbose:
                    print(f"Warning: Skipping invalid result file {path.name}: {e}")

        return results

    def load_pgn(self, game_id: str) -> Optional[str]:
        """
        Load a PGN by game ID.

        Args:
            game_id: The game ID

        Returns:
            PGN string or None if not found
        """
        pgn_path = self.games_dir / f"{game_id}.pgn"
        if not pgn_path.exists():
            return None

        with open(pgn_path) as f:
            return f.read()
