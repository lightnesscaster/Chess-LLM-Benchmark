"""
Base class for LLM player wrappers.
"""

import abc
import chess


class BaseLLMPlayer(abc.ABC):
    """Abstract base class for LLM chess players."""

    def __init__(self, player_id: str, model_name: str):
        """
        Initialize the LLM player.

        Args:
            player_id: Unique identifier for this player
            model_name: Model identifier for the LLM API
        """
        self.player_id = player_id
        self.model_name = model_name
        # Token usage tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        # Last request/response for debugging illegal moves
        self.last_prompt: str = ""
        self.last_raw_response: str = ""
        # Last successful response (for context in next move's prompt)
        self.last_successful_response: str = ""

    def reset_token_usage(self) -> None:
        """Reset token counters and debug state (call at start of each game)."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.last_prompt = ""
        self.last_raw_response = ""
        self.last_successful_response = ""
        # Reset provider tracking for OpenRouterPlayer subclass
        if hasattr(self, 'last_provider'):
            self.last_provider = None

    def mark_move_successful(self) -> None:
        """Mark the last response as successful (called after a legal move)."""
        self.last_successful_response = self.last_raw_response

    def get_token_usage(self) -> dict:
        """Get current token usage stats."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @abc.abstractmethod
    async def select_move(self, board: chess.Board, is_retry: bool = False,
                          last_move_illegal: str = None) -> str:
        """
        Select a move given the current board position.

        Args:
            board: python-chess Board object
            is_retry: Whether this is a retry after an illegal move
            last_move_illegal: The illegal move that was attempted (if retry)

        Returns:
            A move in UCI format (e.g., "e2e4", "g1f3", "e7e8q")
        """
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
