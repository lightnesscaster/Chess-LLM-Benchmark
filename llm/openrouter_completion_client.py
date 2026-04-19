"""
OpenRouter /completions client for text-completion models (e.g. gpt-3.5-turbo-instruct).

These models are not chat-tuned. They perform best when prompted with a
PGN continuation and the raw SAN output is parsed back to UCI.
"""

import asyncio
import json
import os
import random
import re
import time
from typing import Optional

import aiohttp
import chess

from .base_llm import BaseLLMPlayer
from .openrouter_client import TransientAPIError


class OpenRouterCompletionPlayer(BaseLLMPlayer):
    """LLM player that uses OpenRouter's /completions endpoint with a PGN prompt."""

    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/completions"

    # Logit bias workarounds (cl100k_base token IDs):
    #   - Digits '0'..'9' (15..24): suppressed so the model can't emit PGN result
    #     markers ("1-0", "0-1", "1/2-1/2") as if they were moves. SAN moves never
    #     start with a digit; mild bias still lets rank digits through mid-move.
    #   - '-' (12) and '/' (14): appear in result markers but only rarely in SAN
    #     (hyphen in O-O castling). Heavy bias used on retry only.
    #   - <|endoftext|> (100257): heavily suppressed. On deep-endgame PGN prompts
    #     the model otherwise wants to emit EOS (treating the game record as
    #     finished), producing an empty response.
    # Escalating sampling configs: (digit_bias, dash_slash_bias, temperature).
    # Tried in order until a legal move is parsed. Most normal positions resolve
    # at step 0. Steps 1+ fire on losing/forced positions where the model wants
    # to emit a PGN result marker instead of a move. Digit bias is kept at -3
    # throughout — stronger bias breaks legitimate moves containing rank digits
    # (e.g. suppresses the "1" in "Ka1"). Temperature ramps up to break ties
    # when the result-marker prior is near-certain; repeats at the high step
    # re-sample for diversity.
    _SAMPLING_LADDER = [
        (-3, 0, 0.0),
        (-3, -100, 0.3),
        (-3, -100, 0.8),
        (-3, -100, 0.8),
        (-3, -100, 0.8),
        (-3, -100, 1.0),
        (-3, -100, 1.0),
    ]

    # PGN headers without [Result] — including a result marker ("1-0", "*", etc.)
    # strongly primes the model to emit that marker instead of continuing with a move
    # at drawish or decided-looking positions. Omitting it avoids this failure mode.
    PGN_HEADERS = (
        '[Event "FIDE World Championship Match 2024"]\n'
        '[Site "Los Angeles, USA"]\n'
        '[Date "2024.12.01"]\n'
        '[Round "5"]\n'
        '[White "Carlsen, Magnus"]\n'
        '[Black "Nepomniachtchi, Ian"]\n'
        '[WhiteElo "2885"]\n'
        '[WhiteTitle "GM"]\n'
        '[WhiteFideId "1503014"]\n'
        '[BlackElo "2812"]\n'
        '[BlackTitle "GM"]\n'
        '[BlackFideId "4168119"]\n'
        '[TimeControl "40/7200:20/3600:900+30"]\n'
        '[UTCDate "2024.11.27"]\n'
        '[UTCTime "09:01:25"]\n'
        '[Variant "Standard"]\n'
    )

    def __init__(
        self,
        player_id: str,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8,
        timeout: int = 300,
        **_unused,
    ):
        super().__init__(player_id, model_name)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        self._session: Optional[aiohttp.ClientSession] = None
        self.last_provider: Optional[str] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _build_prompt(self, board: chess.Board) -> str:
        """Build a PGN-continuation prompt ending with the move-number prefix."""
        if board.move_stack:
            replay = chess.Board()
            tokens = []
            for i, move in enumerate(board.move_stack):
                san = replay.san(move)
                if i % 2 == 0:
                    tokens.append(f"{(i // 2) + 1}. {san}")
                else:
                    tokens.append(san)
                replay.push(move)
            body = " ".join(tokens)
            # Append the next move-number prefix so the model directly continues with
            # a SAN move. Without this suffix, at deep endgames the model sometimes
            # emits the PGN result marker instead of a move.
            next_num = (len(board.move_stack) // 2) + 1
            if board.turn == chess.WHITE:
                body = f"{body} {next_num}."
            else:
                body = f"{body} {next_num}..."
            return f"{self.PGN_HEADERS}\n{body} "

        # No move history — use FEN setup header so the model knows the position.
        fen_headers = (
            f'[FEN "{board.fen()}"]\n'
            f'[SetUp "1"]\n'
        )
        fullmove = board.fullmove_number
        if board.turn == chess.WHITE:
            suffix = f"{fullmove}."
        else:
            suffix = f"{fullmove}... "
        return f"{self.PGN_HEADERS}{fen_headers}\n{suffix} "

    # Move-number prefixes (e.g. "12", "12.", "12..."), game-result markers, and
    # annotations that may appear before the actual SAN move in a PGN continuation.
    _SKIP_TOKEN_RE = re.compile(r"^(?:\d+\.*|\*|1-0|0-1|1/2-1/2|\{.*\}|\(.*\))$")

    def _parse_san(self, completion_text: str, board: chess.Board) -> Optional[str]:
        """Parse the first SAN move from a completion, return UCI or None."""
        if not completion_text:
            return None
        tokens = re.split(r"\s+", completion_text.strip())
        for token in tokens:
            token = token.rstrip(".,;:")
            if not token or self._SKIP_TOKEN_RE.match(token):
                continue
            candidates = [token]
            stripped = token.rstrip("+#!?")
            if stripped != token:
                candidates.append(stripped)
            for cand in candidates:
                try:
                    move = board.parse_san(cand)
                    return move.uci()
                except (chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
                    continue
        return None

    def _track_usage(self, data: dict) -> None:
        usage = data.get("usage") or {}
        self._last_prompt_tokens = usage.get("prompt_tokens", 0)
        self._last_completion_tokens = usage.get("completion_tokens", 0)
        self.prompt_tokens += self._last_prompt_tokens
        self.completion_tokens += self._last_completion_tokens
        self.total_tokens += usage.get("total_tokens", 0)

    async def select_move(
        self,
        board: chess.Board,
        is_retry: bool = False,
        last_move_illegal: str = None,
    ) -> str:
        move_start_time = time.time()
        session = await self._ensure_session()

        prompt = self._build_prompt(board)
        self.last_prompt = prompt
        self.last_raw_response = ""
        self.last_provider = None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/chess-llm-benchmark",
            "X-Title": "Chess LLM Benchmark",
        }
        # Try each sampling config in the ladder until we get a legal move.
        # On the game-runner's retry (is_retry=True), start at step 1 to avoid
        # repeating the config that just produced an illegal move.
        start_step = 1 if is_retry else 0
        last_completion = ""
        move_uci = None
        for digit_bias, dash_slash_bias, temperature in self._SAMPLING_LADDER[start_step:]:
            bias = {str(tok): digit_bias for tok in range(15, 25)}
            bias["100257"] = -100
            if dash_slash_bias:
                bias["12"] = dash_slash_bias
                bias["14"] = dash_slash_bias
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": max(temperature, self.temperature),
                "stop": ["\n"],
                "logit_bias": bias,
            }
            data = await self._post_with_retries(session, headers, payload)
            self._track_usage(data)
            completion_text = ""
            choices = data.get("choices") or []
            if choices:
                first = choices[0] or {}
                completion_text = first.get("text", "") or ""
            last_completion = completion_text
            move_uci = self._parse_san(completion_text, board)
            if move_uci is not None:
                try:
                    if chess.Move.from_uci(move_uci) in board.legal_moves:
                        break
                except (ValueError, chess.InvalidMoveError):
                    pass
                move_uci = None
        self.last_raw_response = last_completion

        elapsed = time.time() - move_start_time
        self.move_times.append(elapsed)
        self.total_move_time += elapsed

        if move_uci is None:
            # Return raw completion so the caller can log the illegal/garbled attempt.
            return last_completion.strip()[:20]
        return move_uci

    async def _post_with_retries(self, session, headers, payload) -> dict:
        """POST to OpenRouter with transient-error retries. Returns parsed JSON."""
        max_retries = 7
        retry_delay = 2.0
        retryable_http_codes = {429, 500, 502, 503, 504}
        for attempt in range(max_retries):
            try:
                request_timeout = aiohttp.ClientTimeout(total=self.timeout, sock_read=120)
                async with session.post(
                    self.OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=request_timeout,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.last_raw_response = f"[HTTP {response.status}] {error_text[:500]}"
                        if response.status in retryable_http_codes:
                            raise aiohttp.ClientResponseError(
                                response.request_info,
                                response.history,
                                status=response.status,
                                message=f"HTTP {response.status}: {error_text[:200]}",
                            )
                        raise TransientAPIError(
                            f"OpenRouter API error {response.status}: {error_text}"
                        )
                    data = await response.json()
                if data is None:
                    raise TransientAPIError("OpenRouter returned null body")
                provider = data.get("provider")
                if isinstance(provider, str):
                    self.last_provider = provider.strip()[:100]
                return data
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                ConnectionError,
                json.JSONDecodeError,
                aiohttp.ContentTypeError,
            ) as e:
                if attempt < max_retries - 1:
                    jitter = random.uniform(0, 0.1 * retry_delay)
                    sleep_time = retry_delay + jitter
                    print(
                        f"  [Transient error on {self.model_name}, retrying in {sleep_time:.1f}s]: {type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(sleep_time)
                    retry_delay = min(retry_delay * 2, 300)
                else:
                    raise TransientAPIError(
                        f"API call failed after {max_retries} retries: {e}"
                    ) from e
        raise TransientAPIError("unreachable")

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
