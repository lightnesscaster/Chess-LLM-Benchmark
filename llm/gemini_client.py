"""
Direct Google Gemini API client for LLM chess players.

Bypasses OpenRouter to call Gemini models directly via the google-genai SDK.
"""

import os
import re
import asyncio
import random
import time
import chess
from typing import Optional
from .base_llm import BaseLLMPlayer
from .openrouter_client import TransientAPIError
from .prompts import build_chess_prompt


class GeminiPlayer(BaseLLMPlayer):
    """
    LLM player using the Google Gemini API directly.

    Uses the google-genai SDK for direct access to Gemini models.
    """

    def __init__(
        self,
        player_id: str,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        reasoning: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        timeout: int = 300,
    ):
        super().__init__(player_id, model_name)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required (set GEMINI_API_KEY)")
        self.temperature = temperature
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout

        from google import genai
        self._genai = genai
        self._client = self._create_client()

    def _create_client(self):
        """Create a fresh genai client."""
        return self._genai.Client(
            api_key=self.api_key,
            http_options=self._genai.types.HttpOptions(timeout=self.timeout * 1000),
        )

    def _parse_move(self, response_text: str, board: chess.Board = None) -> Optional[str]:
        """
        Parse UCI move from LLM response.

        Same logic as OpenRouterPlayer._parse_move.
        """
        if not response_text:
            return None

        text = response_text.strip()

        # Try to find a UCI move pattern
        uci_pattern = r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b'
        matches = re.findall(uci_pattern, text.lower())
        if matches:
            return matches[-1]

        # Try SAN-style moves with piece prefix
        san_uci_pattern = r'\b[KQRBN]?([a-h][1-8][a-h][1-8][qrbn]?)\b'
        matches = re.findall(san_uci_pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].lower()

        # Try first token
        tokens = text.split()
        if tokens:
            first_token = tokens[0].lower().strip(".,;:!?")
            if re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', first_token):
                return first_token
            if re.match(r'^[kqrbn]?([a-h][1-8][a-h][1-8][qrbn]?)$', first_token):
                match = re.match(r'^[kqrbn]?([a-h][1-8][a-h][1-8][qrbn]?)$', first_token)
                return match.group(1)

        # Try SAN notation if board provided
        if board is not None:
            for token in tokens[:3]:
                clean_token = token.strip(".,;:!?")
                variations = [clean_token]
                if len(clean_token) > 1 and clean_token[0].lower() in 'kqrbn':
                    capitalized = clean_token[0].upper() + clean_token[1:]
                    if capitalized != clean_token:
                        variations.append(capitalized)
                for variant in variations:
                    try:
                        move = board.parse_san(variant)
                        return move.uci()
                    except (chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
                        continue

            san_patterns = [
                r'\b([KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b',
                r'\b([a-h]x?[a-h]?[1-8](?:=[QRBN])?[+#]?)\b',
                r'\b(O-O-O|O-O)\b',
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
        Select a move by querying Gemini API directly.

        Raises:
            TransientAPIError: If the API call fails after retries
        """
        move_start_time = time.time()

        prompt = build_chess_prompt(board, is_retry, last_move_illegal, self.last_successful_response)
        self.last_prompt = prompt
        self.last_raw_response = ""

        # Build config
        config_kwargs = {
            "temperature": self.temperature,
        }

        # Configure thinking/reasoning
        # Gemini 3.x uses thinking_level (string), Gemini 2.5 uses thinking_budget (token count)
        if self.reasoning or self.reasoning_effort:
            is_gemini3 = "gemini-3" in self.model_name
            if is_gemini3:
                thinking_level = {
                    "minimal": "low",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                }.get(self.reasoning_effort, "high")
                config_kwargs["thinking_config"] = self._genai.types.ThinkingConfig(
                    thinking_level=thinking_level,
                )
            else:
                thinking_budget = {
                    "minimal": 1024,
                    "low": 2048,
                    "medium": 16384,
                    "high": 32768,
                }.get(self.reasoning_effort, 16384)
                config_kwargs["thinking_config"] = self._genai.types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                )
            # Gemini API requires temperature=0 when thinking is enabled
            config_kwargs["temperature"] = 0.0

        config = self._genai.types.GenerateContentConfig(**config_kwargs)

        # Retry logic
        max_retries = 7
        retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                # Use the SDK's native async client so cancelled requests do not
                # strand worker threads in asyncio's default executor.
                response = await self._client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )

                # Track token usage
                if response.usage_metadata:
                    self.prompt_tokens += response.usage_metadata.prompt_token_count or 0
                    self.completion_tokens += response.usage_metadata.candidates_token_count or 0
                    self.total_tokens += response.usage_metadata.total_token_count or 0

                # Extract response text (response.text raises ValueError if safety-filtered)
                try:
                    response_text = response.text or ""
                except (ValueError, AttributeError):
                    response_text = ""
                self.last_raw_response = response_text

                # Parse and return the move
                move = self._parse_move(response_text, board)

                if move is None:
                    print(f"  [DEBUG] Raw Gemini response: {repr(response_text[:200] if response_text else '')}")
                    elapsed = time.time() - move_start_time
                    self.move_times.append(elapsed)
                    self.total_move_time += elapsed
                    return response_text.strip()[:20] if response_text else ""

                elapsed = time.time() - move_start_time
                self.move_times.append(elapsed)
                self.total_move_time += elapsed
                return move

            except asyncio.TimeoutError as e:
                if attempt < max_retries - 1:
                    jitter = random.uniform(0, 0.1 * retry_delay)
                    sleep_time = retry_delay + jitter
                    print(f"  [Timeout on {self.model_name}, retrying in {sleep_time:.1f}s]: {e}")
                    await asyncio.sleep(sleep_time)
                    retry_delay = min(retry_delay * 2, 300)
                else:
                    elapsed = time.time() - move_start_time
                    self.move_times.append(elapsed)
                    self.total_move_time += elapsed
                    raise TransientAPIError(
                        f"Gemini API call timed out after {max_retries} retries"
                    ) from e

            except Exception as e:
                error_str = str(e).lower()
                is_transient = any(kw in error_str for kw in [
                    "429", "500", "502", "503", "504",
                    "resource_exhausted", "unavailable", "deadline_exceeded",
                    "internal", "timeout", "rate limit",
                ])
                if is_transient and attempt < max_retries - 1:
                    jitter = random.uniform(0, 0.1 * retry_delay)
                    sleep_time = retry_delay + jitter
                    print(f"  [Transient error on {self.model_name}, retrying in {sleep_time:.1f}s]: {type(e).__name__}: {e}")
                    await asyncio.sleep(sleep_time)
                    retry_delay = min(retry_delay * 2, 300)
                elif not is_transient:
                    elapsed = time.time() - move_start_time
                    self.move_times.append(elapsed)
                    self.total_move_time += elapsed
                    raise TransientAPIError(f"Gemini API error: {e}") from e
                else:
                    elapsed = time.time() - move_start_time
                    self.move_times.append(elapsed)
                    self.total_move_time += elapsed
                    raise TransientAPIError(
                        f"Gemini API call failed after {max_retries} retries: {e}"
                    ) from e

    async def close(self) -> None:
        """Close the Gemini SDK clients."""
        if self._client is None:
            return

        client = self._client
        self._client = None

        try:
            await client.aio.aclose()
        finally:
            client.close()
