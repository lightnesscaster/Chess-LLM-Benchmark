"""Discover Claude Code models available to the configured subscription."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

import yaml

from llm.claude_code_client import ClaudeCodePlayer


DEFAULT_CATALOG_PATH = Path("/tmp/chessbench_claude_models.json")
Probe = Callable[[str], Awaitable[bool]]


def _configured_claude_models(config_path: Path) -> list[str]:
    try:
        config = yaml.safe_load(Path(config_path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(config, dict):
        return []

    configured = list(config.get("web_play_models", []) or []) + list(
        config.get("llms", []) or []
    )
    models = []
    for entry in configured:
        if not isinstance(entry, dict):
            continue
        backend = entry.get("web_api") or entry.get("api") or "openrouter"
        if backend != "claude_code":
            continue
        model_name = str(
            entry.get("web_model_name") or entry.get("model_name") or ""
        ).strip()
        if model_name and model_name not in models:
            models.append(model_name)
    return models


async def _probe_model(model_name: str) -> bool:
    player = ClaudeCodePlayer(
        player_id=f"catalog:{model_name}",
        model_name=model_name,
        reasoning_effort="low",
        timeout=90,
    )
    try:
        await player._run_cli("Reply with exactly: AVAILABLE")
    except Exception:
        return False
    return True


def _write_catalog(output_path: Path, models: list[str]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    temporary_path.write_text(json.dumps({"models": models}))
    temporary_path.replace(output_path)


async def refresh_claude_catalog(
    config_path: Path,
    output_path: Path,
    probe: Probe = _probe_model,
) -> list[str]:
    """Probe configured Claude models concurrently and persist successes."""
    model_names = _configured_claude_models(config_path)

    async def safe_probe(model_name: str) -> bool:
        try:
            return bool(await probe(model_name))
        except Exception:
            return False

    results = await asyncio.gather(*(safe_probe(name) for name in model_names))
    available = [
        name for name, is_available in zip(model_names, results) if is_available
    ]
    _write_catalog(output_path, available)
    return available


def main() -> None:
    config_path = Path(__file__).parent.parent / "config" / "benchmark.yaml"
    output_path = Path(
        os.environ.get("CLAUDE_MODEL_CATALOG_PATH", str(DEFAULT_CATALOG_PATH))
    )
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        _write_catalog(output_path, [])
        return
    available = asyncio.run(refresh_claude_catalog(config_path, output_path))
    print(f"Verified {len(available)} Claude subscription models.")


if __name__ == "__main__":
    main()
