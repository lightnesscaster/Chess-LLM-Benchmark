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

    Supports both local file and Firestore backends.
    """

    def __init__(
        self,
        games_dir: str = "data/games",
        results_dir: str = "data/results",
        use_firestore: bool = None,
    ):
        """
        Initialize the logger.

        Args:
            games_dir: Directory to save PGN files (used for local storage)
            results_dir: Directory to save result JSON files (used for local storage)
            use_firestore: If True, use Firestore. If None, auto-detect.
        """
        self.games_dir = Path(games_dir)
        self.results_dir = Path(results_dir)

        # Determine storage backend
        if use_firestore is None:
            use_firestore = self._should_use_firestore()
        self._use_firestore = use_firestore

        if self._use_firestore:
            self._init_firestore()
        else:
            # Ensure directories exist for local storage
            self.games_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)

    def _should_use_firestore(self) -> bool:
        """Check if we should use Firestore."""
        # Check env var
        if os.environ.get("FIREBASE_ENABLED", "").lower() in ("1", "true", "yes"):
            return True

        # Check for credentials file
        possible_paths = [
            Path(__file__).parent.parent / "firebase-key.json",
            Path.cwd() / "firebase-key.json",
        ]
        for p in possible_paths:
            if p.exists():
                return True

        return False

    def _init_firestore(self) -> None:
        """Initialize Firestore connection."""
        from firebase_client import get_firestore_client, GAMES_COLLECTION, RESULTS_COLLECTION
        self._db = get_firestore_client()
        self._games_collection = GAMES_COLLECTION
        self._results_collection = RESULTS_COLLECTION

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

        if self._use_firestore:
            # Save PGN to Firestore
            self._db.collection(self._games_collection).document(game_id).set({
                "game_id": game_id,
                "pgn": pgn_str,
            })

            # Update result with path reference
            result.pgn_path = f"firestore://{self._games_collection}/{game_id}"

            # Save result to Firestore
            self._db.collection(self._results_collection).document(game_id).set(
                result.to_json()
            )
        else:
            # Save PGN locally
            pgn_path = self.games_dir / f"{game_id}.pgn"
            with open(pgn_path, "w") as f:
                f.write(pgn_str)

            # Update result with path
            result.pgn_path = str(pgn_path)

            # Save result JSON locally
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
        if self._use_firestore:
            doc = self._db.collection(self._results_collection).document(game_id).get()
            if not doc.exists:
                return None
            return GameResult.from_json(doc.to_dict())
        else:
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

        if self._use_firestore:
            docs = self._db.collection(self._results_collection).stream()
            for doc in docs:
                try:
                    data = doc.to_dict()
                    results.append(GameResult.from_json(data))
                except (KeyError, TypeError, ValueError) as e:
                    if verbose:
                        print(f"Warning: Skipping invalid result doc {doc.id}: {e}")
        else:
            for path in self.results_dir.glob("*.json"):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    results.append(GameResult.from_json(data))
                except json.JSONDecodeError as e:
                    if verbose:
                        print(f"Warning: Skipping corrupted JSON file {path.name}: {e}")
                except (FileNotFoundError, OSError) as e:
                    if verbose:
                        print(f"Warning: Skipping unreadable file {path.name}: {e}")
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
        if self._use_firestore:
            doc = self._db.collection(self._games_collection).document(game_id).get()
            if not doc.exists:
                return None
            data = doc.to_dict()
            return data.get("pgn")
        else:
            pgn_path = self.games_dir / f"{game_id}.pgn"
            if not pgn_path.exists():
                return None

            with open(pgn_path) as f:
                return f.read()
