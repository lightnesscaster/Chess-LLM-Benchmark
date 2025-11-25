"""
OpenRouter API client for LLM chess players.
"""

import os
import re
import aiohttp
import chess
from typing import Optional
from .base_llm import BaseLLMPlayer
from .prompts import build_chess_prompt


class OpenRouterPlayer(BaseLLMPlayer):
    """
    LLM player using OpenRouter API.

    Supports any model available on OpenRouter.
    """

    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        player_id: str,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 10,
    ):
        """
        Initialize OpenRouter player.

        Args:
            player_id: Unique identifier for this player
            model_name: OpenRouter model identifier (e.g., "meta-llama/llama-4-maverick")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response
        """
        super().__init__(player_id, model_name)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazily create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _parse_move(self, response_text: str) -> Optional[str]:
        """
        Parse UCI move from LLM response.

        Tries to extract a valid UCI move pattern from potentially noisy output.
        Handles both pure UCI (e2e4) and SAN-style with piece prefix (Nb1c3).

        Args:
            response_text: Raw response from LLM

        Returns:
            UCI move string or None if no valid pattern found
        """
        if not response_text:
            return None

        # Clean up the response
        text = response_text.strip()

        # Try to find a UCI move pattern
        # Standard moves: e2e4, a1h8
        # Promotions: e7e8q, a2a1r
        # Pattern: [a-h][1-8][a-h][1-8][qrbn]?
        uci_pattern = r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b'

        matches = re.findall(uci_pattern, text.lower())
        if matches:
            return matches[0]

        # Try to find SAN-style moves with piece prefix (e.g., Nb1c3, Qd1d3)
        # Pattern: optional piece letter + UCI move
        san_uci_pattern = r'\b[KQRBN]?([a-h][1-8][a-h][1-8][qrbn]?)\b'
        matches = re.findall(san_uci_pattern, text, re.IGNORECASE)
        if matches:
            return matches[0].lower()

        # If no UCI pattern found, try taking first word/token
        tokens = text.split()
        if tokens:
            first_token = tokens[0].lower().strip(".,;:!?")
            # Check if it looks like a UCI move
            if re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', first_token):
                return first_token
            # Check if it's a piece prefix + UCI move
            if re.match(r'^[kqrbn]?([a-h][1-8][a-h][1-8][qrbn]?)$', first_token):
                match = re.match(r'^[kqrbn]?([a-h][1-8][a-h][1-8][qrbn]?)$', first_token)
                return match.group(1)

        return None

    async def select_move(self, board: chess.Board, is_retry: bool = False,
                          last_move_illegal: str = None) -> str:
        """
        Select a move by querying OpenRouter API.

        Args:
            board: python-chess Board object
            is_retry: Whether this is a retry after an illegal move
            last_move_illegal: The illegal move that was attempted (if retry)

        Returns:
            A move in UCI format
        """
        session = await self._ensure_session()

        prompt = build_chess_prompt(board, is_retry, last_move_illegal)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/chess-llm-benchmark",
            "X-Title": "Chess LLM Benchmark",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        timeout = aiohttp.ClientTimeout(total=120)  # 2 minute timeout per move
        async with session.post(
            self.OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"OpenRouter API error {response.status}: {error_text}")

            data = await response.json()

        # Extract response text
        try:
            response_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected API response format: {data}") from e

        # Parse and return the move
        move = self._parse_move(response_text)
        if move is None:
            # Return raw response for logging, will be marked as illegal
            return response_text.strip()[:20] if response_text else ""

        return move

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
