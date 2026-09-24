"""Claude Code CLI-backed chess player."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
import time
from typing import Optional

import chess

from .base_llm import BaseLLMPlayer
from .openrouter_client import TransientAPIError
from .protocol import parse_resignation
from .prompts import build_chess_prompt


SECRET_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_-]+|Bearer\s+\S+", re.IGNORECASE)


class ClaudeCodePlayer(BaseLLMPlayer):
    """Chess player that shells out to Claude Code for each move."""

    CHESS_SYSTEM_PROMPT = (
        "You are a chess move selector. Analyze only the position in the "
        "user prompt and return exactly the requested response line."
    )

    def __init__(
        self,
        player_id: str,
        model_name: str,
        reasoning_effort: str = "medium",
        claude_command: str = "claude",
        timeout: int = 600,
        **_: object,
    ) -> None:
        normalized_model = model_name.removeprefix("anthropic/")
        super().__init__(player_id=player_id, model_name=normalized_model)
        self.reasoning_effort = reasoning_effort
        self.claude_command = claude_command
        self.timeout = timeout
        self.last_api_error = ""

    def _build_prompt(
        self,
        board: chess.Board,
        is_retry: bool,
        last_move_illegal: Optional[str],
        allow_resignation: bool = False,
    ) -> str:
        prompt = build_chess_prompt(
            board,
            is_retry=is_retry,
            illegal_move=last_move_illegal,
            previous_response=self.last_successful_response,
            allow_resignation=allow_resignation,
            rating_context=self.rating_context,
        )
        output_constraint = (
            "- Return exactly one line: <uci> or resign"
            if allow_resignation
            else "- Return exactly one line containing a legal UCI move"
        )
        return (
            f"{prompt}\n\n"
            "Claude Code chess constraints:\n"
            "- Do not use tools, files, web search, or external sources.\n"
            f"{output_constraint}"
        )

    def _command(self, prompt: str) -> list[str]:
        return [
            self.claude_command,
            "--print",
            "--output-format",
            "json",
            "--model",
            self.model_name,
            "--effort",
            self.reasoning_effort,
            "--safe-mode",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--no-chrome",
            "--permission-mode",
            "dontAsk",
            "--max-turns",
            "1",
            "--tools",
            "",
            "--disallowedTools",
            "mcp__*",
            "--system-prompt",
            self.CHESS_SYSTEM_PROMPT,
            prompt,
        ]

    @staticmethod
    def _subscription_environment(environ: dict[str, str]) -> dict[str, str]:
        environment = dict(environ)
        environment.pop("ANTHROPIC_API_KEY", None)
        environment.pop("ANTHROPIC_AUTH_TOKEN", None)
        return environment

    async def select_move(
        self,
        board: chess.Board,
        is_retry: bool = False,
        last_move_illegal: Optional[str] = None,
    ) -> str:
        prompt = self._build_prompt(
            board,
            is_retry,
            last_move_illegal,
            allow_resignation=self.allow_resignation,
        )
        self.last_prompt = prompt
        self.last_raw_response = ""
        started = time.time()
        try:
            response_text, usage = await self._run_cli(prompt)
        except Exception as error:
            self.last_api_error = (
                f"Claude Code call failed ({type(error).__name__})."
            )
            raise TransientAPIError(self.last_api_error) from error
        finally:
            elapsed = time.time() - started
            self.move_times.append(elapsed)
            self.total_move_time += elapsed

        self.last_api_error = ""
        self.last_raw_response = response_text
        self.record_chess_usage(usage, self.CHESS_SYSTEM_PROMPT + "\n\n" + prompt)

        resignation = parse_resignation(response_text)
        if resignation:
            return resignation
        match = re.findall(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", response_text.lower())
        if match:
            return match[-1]
        return response_text.strip()[:80] if response_text else ""

    async def _run_cli(self, prompt: str) -> tuple[str, dict]:
        with tempfile.TemporaryDirectory(prefix="claude_chess_workspace_") as workdir:
            process = await asyncio.create_subprocess_exec(
                *self._command(prompt),
                cwd=workdir,
                env=self._subscription_environment(dict(os.environ)),
                stdin=subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout_bytes, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise

        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            raise RuntimeError(
                f"Claude Code exited with status {process.returncode}: "
                f"{self._failure_detail(stdout)}"
            )

        payload = self._parse_payload(stdout)
        if payload.get("is_error") or payload.get("subtype") not in {None, "success"}:
            raise RuntimeError(
                "Claude Code returned an unsuccessful result: "
                f"{self._failure_detail(stdout)}"
            )
        response_text = str(payload.get("result") or "").strip()
        if not response_text:
            raise RuntimeError("Claude Code returned an empty result.")

        raw_usage = payload.get("usage") or {}
        usage = {
            "prompt_tokens": sum(int(raw_usage.get(key, 0) or 0) for key in (
                "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"
            )),
            "completion_tokens": int(raw_usage.get("output_tokens", 0) or 0),
            "cached_input_tokens": int(raw_usage.get("cache_read_input_tokens", 0) or 0),
            "cache_creation_input_tokens": int(raw_usage.get("cache_creation_input_tokens", 0) or 0),
            "cache_accounting_known": all(key in raw_usage for key in (
                "cache_read_input_tokens", "cache_creation_input_tokens"
            )),
        }
        creation = raw_usage.get("cache_creation") or {}
        for ttl in ("5m", "1h"):
            key = f"ephemeral_{ttl}_input_tokens"
            if key in creation:
                usage[f"cache_creation_{ttl}_input_tokens"] = int(creation[key] or 0)
        return response_text, usage

    @classmethod
    def _failure_detail(cls, stdout: str) -> str:
        """Summarize CLI output for server logs, with credentials redacted."""
        try:
            payload = cls._parse_payload(stdout)
            detail = str(payload.get("result") or payload.get("subtype") or "")
        except RuntimeError:
            detail = ""
        if not detail:
            lines = [line for line in stdout.splitlines() if line.strip()]
            detail = lines[-1] if lines else "no output"
        return SECRET_PATTERN.sub("[redacted]", " ".join(detail.split()))[:300]

    @staticmethod
    def _parse_payload(stdout: str) -> dict:
        for line in reversed(stdout.splitlines()):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise RuntimeError("Claude Code returned invalid JSON.")

    async def close(self) -> None:
        return None
