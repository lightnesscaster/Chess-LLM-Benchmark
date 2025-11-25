"""
Data models for the chess LLM benchmark.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel
from datetime import datetime


class PlayerType(str, Enum):
    """Type of player: LLM or chess engine."""
    LLM = "llm"
    ENGINE = "engine"


class PlayerConfig(BaseModel):
    """Configuration for a player (LLM or engine)."""
    player_id: str              # e.g. "llama-4-maverick", "maia-1100"
    player_type: PlayerType
    display_name: str
    is_anchor: bool = False     # True for engine anchors with fixed ratings
    anchor_rating: Optional[int] = None  # Fixed rating for anchors

    # Engine-specific
    engine_path: Optional[str] = None
    engine_options: Optional[dict] = None

    # LLM-specific
    model_name: Optional[str] = None  # OpenRouter model identifier


class GameResult(BaseModel):
    """Result of a single game."""
    game_id: str
    white_id: str
    black_id: str
    winner: str                 # "white", "black", "draw"
    termination: str            # "checkmate", "stalemate", "insufficient_material",
                                # "fifty_moves", "threefold_repetition",
                                # "forfeit_illegal_move", etc.
    moves: int                  # Total half-moves (plies)
    illegal_moves_white: int
    illegal_moves_black: int
    total_moves_white: int      # Moves attempted by white
    total_moves_black: int      # Moves attempted by black
    pgn_path: str
    created_at: str             # ISO timestamp

    def to_json(self) -> dict:
        """Convert to JSON-serializable dict."""
        return self.model_dump()

    @classmethod
    def from_json(cls, data: dict) -> "GameResult":
        """Create from JSON dict."""
        return cls(**data)


class MatchConfig(BaseModel):
    """Configuration for a match (series of games between two players)."""
    player_a_id: str
    player_b_id: str
    games_per_color: int = 1    # Games where A is white, then games where A is black
    max_moves: int = 200        # Max half-moves per game before draw


class BenchmarkConfig(BaseModel):
    """Overall benchmark configuration."""
    games_vs_anchor_per_color: int = 10
    games_vs_other_llm_per_color: int = 5
    max_moves: int = 200
    max_concurrent_games: int = 4
    openrouter_api_key: Optional[str] = None
