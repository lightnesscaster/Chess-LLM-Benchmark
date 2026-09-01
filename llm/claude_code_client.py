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
from .prompts import build_chess_prompt


class ClaudeCodePlayer(BaseLLMPlayer):
    """Chess player that shells out to Claude Code for each move."""

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
    ) -> str:
        prompt = build_chess_prompt(
            board,
            is_retry=is_retry,
            illegal_move=last_move_illegal,
            previous_response=self.last_successful_response,
        )
        return (
            f"{prompt}\n\n"
            "Claude Code chess constraints:\n"
            "- Do not use tools, files, web search, or external sources.\n"
            "- Return exactly one line in this format: MOVE: <uci>"
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
            (
                "You are a chess move selector. Analyze only the position in the "
                "user prompt and return exactly the requested UCI move line."
            ),
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
        prompt = self._build_prompt(board, is_retry, last_move_illegal)
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
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens

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

        if process.returncode != 0:
            raise RuntimeError(f"Claude Code exited with status {process.returncode}.")

        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        payload = self._parse_payload(stdout)
        if payload.get("is_error") or payload.get("subtype") not in {None, "success"}:
            raise RuntimeError("Claude Code returned an unsuccessful result.")
        response_text = str(payload.get("result") or "").strip()
        if not response_text:
            raise RuntimeError("Claude Code returned an empty result.")

        raw_usage = payload.get("usage") or {}
        usage = {
            "prompt_tokens": int(raw_usage.get("input_tokens", 0) or 0),
            "completion_tokens": int(raw_usage.get("output_tokens", 0) or 0),
        }
        return response_text, usage

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
