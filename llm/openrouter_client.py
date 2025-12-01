"""
OpenRouter API client for LLM chess players.
"""

import json
import os
import re
import asyncio
import random
import aiohttp
import chess
from typing import Optional
from .base_llm import BaseLLMPlayer
from .prompts import build_chess_prompt


class TransientAPIError(Exception):
    """Raised when API call fails due to transient network issues after retries."""
    pass


class OpenRouterPlayer(BaseLLMPlayer):
    """
    LLM player using OpenRouter API.

    Supports any model available on OpenRouter.
    """

    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    VALID_REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}

    def __init__(
        self,
        player_id: str,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 10,
        reasoning: bool = False,
        reasoning_effort: Optional[str] = None,
    ):
        """
        Initialize OpenRouter player.

        Args:
            player_id: Unique identifier for this player
            model_name: OpenRouter model identifier (e.g., "meta-llama/llama-4-maverick")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response
            reasoning: Enable reasoning mode for hybrid models
            reasoning_effort: Reasoning effort level (low, medium, high, xhigh).
                If set, automatically enables reasoning mode.
        """
        super().__init__(player_id, model_name)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required")
        if reasoning_effort is not None and reasoning_effort not in self.VALID_REASONING_EFFORTS:
            raise ValueError(
                f"Invalid reasoning_effort: '{reasoning_effort}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_REASONING_EFFORTS))}"
            )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazily create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _parse_move(self, response_text: str, board: chess.Board = None) -> Optional[str]:
        """
        Parse UCI move from LLM response.

        Tries to extract a valid UCI move pattern from potentially noisy output.
        Also attempts to parse SAN notation (e.g., Nf6, Bc4) if UCI not found.

        Args:
            response_text: Raw response from LLM
            board: Current board position (needed for SAN parsing)

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
            # Return last match - models often explain failed moves before giving correct one
            return matches[-1]

        # Try to find SAN-style moves with piece prefix (e.g., Nb1c3, Qd1d3)
        # Pattern: optional piece letter + UCI move
        san_uci_pattern = r'\b[KQRBN]?([a-h][1-8][a-h][1-8][qrbn]?)\b'
        matches = re.findall(san_uci_pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].lower()

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

        # Try to parse as SAN notation (e.g., Nf6, Bc4, O-O) if board provided
        if board is not None:
            # Try first token as SAN
            for token in tokens[:3]:  # Check first few tokens
                clean_token = token.strip(".,;:!?")
                # Try variations: as-is, with first letter capitalized (for piece prefix)
                variations = [clean_token]
                if len(clean_token) > 1 and clean_token[0].lower() in 'kqrbn':
                    # LLM might have used lowercase piece letter (e.g., "nbd7" instead of "Nbd7")
                    capitalized = clean_token[0].upper() + clean_token[1:]
                    if capitalized != clean_token:
                        variations.append(capitalized)
                for variant in variations:
                    try:
                        move = board.parse_san(variant)
                        return move.uci()
                    except (chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
                        continue

            # Try to find SAN pattern anywhere in text
            # Match piece moves: Nf3, Bb5, Qd1, Kf1, Rh8
            # Match pawn moves: e4, d5, exd5, fxg6
            # Match castling: O-O, O-O-O
            san_patterns = [
                r'\b([KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b',  # Piece moves
                r'\b([a-h]x?[a-h]?[1-8](?:=[QRBN])?[+#]?)\b',  # Pawn moves
                r'\b(O-O-O|O-O)\b',  # Castling
            ]
            for pattern in san_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    try:
                        move = board.parse_san(match)
                        return move.uci()
                    except (chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
                        continue

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

        Raises:
            TransientAPIError: If the API call fails after retries due to network issues
        """
        session = await self._ensure_session()

        # Pass previous successful response for context
        # (use last_successful_response so retries still have context from last good move)
        prompt = build_chess_prompt(board, is_retry, last_move_illegal, self.last_successful_response)
        self.last_prompt = prompt  # Store for debugging illegal moves
        self.last_raw_response = ""  # Clear stale data before API call

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
        }

        # Only set max_tokens if explicitly specified (non-zero)
        # Reasoning models need unlimited tokens to complete their analysis
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens

        # Enable reasoning mode for hybrid models
        if self.reasoning or self.reasoning_effort is not None:
            reasoning_config = {"enabled": True}
            if self.reasoning_effort is not None:
                reasoning_config["effort"] = self.reasoning_effort
            payload["reasoning"] = reasoning_config

        # Retry logic for transient network and HTTP errors
        max_retries = 3
        retry_delay = 2.0  # seconds
        retryable_http_codes = {429, 500, 502, 503, 504}

        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout per move
                async with session.post(
                    self.OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        # Store error for debugging before raising
                        self.last_raw_response = f"[HTTP {response.status}] {error_text[:500]}"
                        # Retry on transient server errors and rate limits
                        if response.status in retryable_http_codes:
                            raise aiohttp.ClientResponseError(
                                response.request_info,
                                response.history,
                                status=response.status,
                                message=f"HTTP {response.status}: {error_text[:200]}"
                            )
                        # Non-retryable error (4xx client errors except 429)
                        raise RuntimeError(f"OpenRouter API error {response.status}: {error_text}")

                    data = await response.json()

                    # Check for embedded errors (API returns 200 but with error in body)
                    choices = data.get("choices")
                    if (isinstance(choices, list) and len(choices) > 0 and
                        isinstance(choices[0], dict) and "error" in choices[0]):
                        error_info = choices[0]["error"]
                        # Handle error_info being a string or dict
                        if isinstance(error_info, dict):
                            error_msg = error_info.get("message", "Unknown error")
                            error_code = error_info.get("code", 0)
                            # Ensure error_code is int for comparison
                            try:
                                error_code = int(error_code) if error_code else 0
                            except (ValueError, TypeError):
                                error_code = 0
                        else:
                            error_msg = str(error_info)
                            error_code = 0
                        self.last_raw_response = f"[API Error {error_code}] {error_msg}"
                        # Treat as transient (retryable) if it's a network/server error
                        if error_code in retryable_http_codes or "network" in error_msg.lower():
                            raise aiohttp.ClientResponseError(
                                response.request_info,
                                response.history,
                                status=error_code if error_code else 500,
                                message=f"Embedded error {error_code}: {error_msg}"
                            )
                        # Non-retryable embedded error - still an API error, not an illegal move
                        raise TransientAPIError(f"API error in response: {error_code} - {error_msg}")

                # Success - break out of retry loop
                break

            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError,
                    json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                if attempt < max_retries - 1:
                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0, 0.1 * retry_delay)
                    sleep_time = retry_delay + jitter
                    print(f"  [Transient error, retrying in {sleep_time:.1f}s]: {type(e).__name__}: {e}")
                    await asyncio.sleep(sleep_time)
                    retry_delay = min(retry_delay * 2, 60)  # Exponential backoff with cap
                    # Recreate session in case connection is stale
                    await self.close()
                    session = await self._ensure_session()
                else:
                    raise TransientAPIError(
                        f"API call failed after {max_retries} retries: {e}"
                    ) from e

        # Track token usage
        if "usage" in data:
            usage = data["usage"]
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)

        # Extract response text
        try:
            response_text = data["choices"][0]["message"]["content"]
            self.last_raw_response = response_text or ""  # Store for debugging illegal moves
            # Debug: check for empty or suspiciously short responses
            if not response_text or len(response_text.strip()) < 4:
                print(f"  [DEBUG] Short/empty response. Full API data: {data}")
        except (KeyError, IndexError) as e:
            self.last_raw_response = f"[Failed to extract: {data}]"
            raise RuntimeError(f"Unexpected API response format: {data}") from e

        # Parse and return the move
        move = self._parse_move(response_text, board)
        if move is None:
            # Debug: print raw response when parsing fails
            print(f"  [DEBUG] Raw LLM response: {repr(response_text[:200] if response_text else '')}")
            # Return raw response for logging, will be marked as illegal
            return response_text.strip()[:20] if response_text else ""

        return move

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
