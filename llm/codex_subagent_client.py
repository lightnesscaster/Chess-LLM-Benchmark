"""
Codex CLI-backed chess player.

This player lets the benchmark ask a fresh non-interactive Codex agent for each
move. It is intentionally implemented as a normal BaseLLMPlayer so existing game
runner retry, PGN, timing, and token accounting paths continue to work.
"""

import asyncio
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import chess

from .base_llm import BaseLLMPlayer
from .openrouter_client import TransientAPIError
from .prompts import build_chess_prompt


class CodexSubagentPlayer(BaseLLMPlayer):
    """Chess player that shells out to `codex exec` for each move."""

    VALID_REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}
    _GLOBAL_SEMAPHORE: Optional[asyncio.Semaphore] = None
    _GLOBAL_LIMIT: Optional[int] = None

    def __init__(
        self,
        player_id: str,
        model_name: str = "gpt-5.5",
        reasoning_effort: str = "medium",
        codex_command: str = "codex",
        timeout: int = 600,
        max_retries: int = 2,
        max_concurrent: Optional[int] = None,
        sandbox: str = "read-only",
        ignore_rules: bool = True,
        ephemeral: bool = True,
        include_legal_moves: bool = False,
        extra_args: Optional[list[str]] = None,
        working_dir: Optional[str] = None,
        **_: object,
    ):
        codex_model_name = self._normalize_model_name(model_name)
        super().__init__(player_id=player_id, model_name=codex_model_name)

        if reasoning_effort not in self.VALID_REASONING_EFFORTS:
            raise ValueError(
                f"Invalid Codex reasoning_effort: {reasoning_effort}. "
                f"Must be one of: {', '.join(sorted(self.VALID_REASONING_EFFORTS))}"
            )

        self.reasoning_effort = reasoning_effort
        self.codex_command = codex_command
        self.timeout = timeout
        self.max_retries = max_retries
        self.sandbox = sandbox
        self.ignore_rules = ignore_rules
        self.ephemeral = ephemeral
        self.include_legal_moves = include_legal_moves
        self.extra_args = extra_args or []
        self.working_dir = working_dir
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0

        if max_concurrent is None:
            max_concurrent = int(os.environ.get("CODEX_SUBAGENT_MAX_CONCURRENT", "6"))
        self._set_global_limit(max(1, max_concurrent))

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """Convert repo/OpenRouter-style model ids to Codex CLI model ids."""
        if model_name.startswith("openai/"):
            return model_name[len("openai/"):]
        return model_name

    @classmethod
    def _set_global_limit(cls, max_concurrent: int) -> None:
        if cls._GLOBAL_SEMAPHORE is None or cls._GLOBAL_LIMIT != max_concurrent:
            cls._GLOBAL_SEMAPHORE = asyncio.Semaphore(max_concurrent)
            cls._GLOBAL_LIMIT = max_concurrent

    def _build_prompt(
        self,
        board: chess.Board,
        is_retry: bool,
        last_move_illegal: Optional[str],
    ) -> str:
        prompt = build_chess_prompt(
            board,
            is_retry=is_retry,
            illegal_move=last_move_illegal,
            previous_response=self.last_successful_response,
        )

        constraints = [
            "Codex subagent constraints:",
            "- Do not use tools, engines, files, web search, or external sources.",
            "- Return exactly one line in this format: MOVE: <uci>",
        ]
        if self.include_legal_moves:
            legal_moves = " ".join(move.uci() for move in board.legal_moves)
            constraints.append(f"- Choose one move from these legal UCI moves: {legal_moves}")

        return f"{prompt}\n\n" + "\n".join(constraints)

    def _command(self, output_path: str, prompt: str) -> list[str]:
        cmd = [
            self.codex_command,
            "-a",
            "never",
            "--sandbox",
            self.sandbox,
            "exec",
            "-m",
            self.model_name,
            "-c",
            f"model_reasoning_effort={self.reasoning_effort}",
            "--json",
        ]
        if self.ignore_rules:
            cmd.append("--ignore-rules")
        if self.ephemeral:
            cmd.append("--ephemeral")
        cmd.extend(["--output-last-message", output_path])
        cmd.extend(self.extra_args)
        cmd.append(prompt)
        return cmd

    async def select_move(
        self,
        board: chess.Board,
        is_retry: bool = False,
        last_move_illegal: str = None,
    ) -> str:
        move_start_time = time.time()
        prompt = self._build_prompt(board, is_retry, last_move_illegal)
        self.last_prompt = prompt
        self.last_raw_response = ""
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0

        assert self._GLOBAL_SEMAPHORE is not None
        async with self._GLOBAL_SEMAPHORE:
            try:
                response_text, usage = await self._run_codex(prompt)
            finally:
                elapsed = time.time() - move_start_time
                self.move_times.append(elapsed)
                self.total_move_time += elapsed

        self.last_raw_response = response_text
        self._track_usage(usage)

        move = self._parse_move(response_text, board)
        if move is None:
            return response_text.strip()[:80] if response_text else ""
        return move

    async def _run_codex(self, prompt: str) -> tuple[str, dict]:
        last_error: Optional[BaseException] = None

        for attempt in range(self.max_retries):
            output_file = tempfile.NamedTemporaryFile(prefix="codex_chess_move_", suffix=".txt", delete=False)
            output_path = output_file.name
            output_file.close()

            try:
                cmd = self._command(output_path, prompt)
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=self.working_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                stdout_bytes, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                usage = self._parse_usage(stdout)
                response_text = self._read_response(output_path, stdout)

                if process.returncode == 0:
                    return response_text, usage

                last_error = RuntimeError(
                    f"codex exec exited {process.returncode}: {stdout[-1000:]}"
                )
            except asyncio.TimeoutError as exc:
                last_error = exc
                if "process" in locals() and process.returncode is None:
                    process.kill()
                    await process.wait()
            finally:
                try:
                    Path(output_path).unlink(missing_ok=True)
                except OSError:
                    pass

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))

        raise TransientAPIError(f"Codex subagent call failed: {last_error}")

    def _read_response(self, output_path: str, stdout: str) -> str:
        path = Path(output_path)
        if path.exists():
            text = path.read_text(errors="replace").strip()
            if text:
                return text

        last_agent_message = ""
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed":
                item = event.get("item") or {}
                if item.get("type") == "agent_message":
                    last_agent_message = item.get("text", "") or last_agent_message
        return last_agent_message.strip()

    def _parse_usage(self, stdout: str) -> dict:
        usage = {}
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn.completed":
                event_usage = event.get("usage") or {}
                usage = {
                    "prompt_tokens": int(event_usage.get("input_tokens", 0) or 0),
                    "completion_tokens": int(event_usage.get("output_tokens", 0) or 0),
                    "cached_input_tokens": int(event_usage.get("cached_input_tokens", 0) or 0),
                }
        return usage

    def _track_usage(self, usage: dict) -> None:
        self._last_prompt_tokens = usage.get("prompt_tokens", 0)
        self._last_completion_tokens = usage.get("completion_tokens", 0)
        self.prompt_tokens += self._last_prompt_tokens
        self.completion_tokens += self._last_completion_tokens
        self.total_tokens += self._last_prompt_tokens + self._last_completion_tokens

    def _parse_move(self, response_text: str, board: chess.Board) -> Optional[str]:
        if not response_text:
            return None

        text = response_text.strip().replace("0-0-0", "O-O-O").replace("0-0", "O-O")
        if text.upper().startswith("MOVE:"):
            text = text.split(":", 1)[1].strip()

        uci_pattern = r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b"
        matches = re.findall(uci_pattern, text.lower())
        if matches:
            return matches[-1]

        san_uci_pattern = r"\b[KQRBN]?([a-h][1-8][a-h][1-8][qrbn]?)[+#]?\b"
        matches = re.findall(san_uci_pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].lower()

        tokens = text.split()
        for token in tokens[:4]:
            for candidate in self._san_candidates(token):
                try:
                    return board.parse_san(candidate).uci()
                except (chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
                    continue

        san_patterns = [
            r"\b([KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?[!?]*)\b",
            r"\b([a-h]x?[a-h]?[1-8](?:=[QRBN])?[+#]?[!?]*)\b",
            r"\b(O-O-O|O-O)\b",
        ]
        for pattern in san_patterns:
            for match in re.findall(pattern, text):
                for candidate in self._san_candidates(match):
                    try:
                        return board.parse_san(candidate).uci()
                    except (chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
                        continue

        return None

    @staticmethod
    def _san_candidates(token: str) -> list[str]:
        clean = token.strip("` \t\r\n.,;:")
        candidates = [clean]
        without_annotations = re.sub(r"[!?]+$", "", clean)
        if without_annotations not in candidates:
            candidates.append(without_annotations)
        without_check = without_annotations.rstrip("+#")
        if without_check not in candidates:
            candidates.append(without_check)

        for item in list(candidates):
            if item and item[0] in "kqrbn":
                capitalized = item[0].upper() + item[1:]
                if capitalized not in candidates:
                    candidates.append(capitalized)

        return [candidate for candidate in candidates if candidate]

    async def close(self) -> None:
        """No persistent resources are held between moves."""
        return None
