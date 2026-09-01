"""Tests for subscription-backed Claude model discovery."""

import asyncio
import json

import yaml


def test_refresh_catalog_writes_only_models_the_subscription_can_use(tmp_path):
    from web.claude_catalog import refresh_claude_catalog

    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump({
        "web_play_models": [
            {
                "player_id": "fable-5.1",
                "model_name": "claude-fable-5-1",
                "web_api": "claude_code",
            },
            {
                "player_id": "fable-5",
                "model_name": "claude-fable-5",
                "web_api": "claude_code",
            },
        ],
        "llms": [
            {
                "player_id": "latest-opus",
                "model_name": "anthropic/legacy-label",
                "web_api": "claude_code",
                "web_model_name": "opus",
            },
            {
                "player_id": "codex-model",
                "model_name": "openai/gpt-5.6-sol",
                "api": "codex",
            },
        ],
    }))
    output_path = tmp_path / "claude-models.json"

    async def probe(model_name):
        return model_name in {"claude-fable-5-1", "opus"}

    available = asyncio.run(refresh_claude_catalog(
        config_path=config_path,
        output_path=output_path,
        probe=probe,
    ))

    assert available == ["claude-fable-5-1", "opus"]
    assert json.loads(output_path.read_text()) == {
        "models": ["claude-fable-5-1", "opus"],
    }


def test_refresh_catalog_fails_closed_when_probes_error(tmp_path):
    from web.claude_catalog import refresh_claude_catalog

    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump({
        "web_play_models": [
            {
                "player_id": "fable-5.1",
                "model_name": "claude-fable-5-1",
                "web_api": "claude_code",
            },
        ],
    }))
    output_path = tmp_path / "claude-models.json"

    async def failing_probe(_model_name):
        raise RuntimeError("subscription unavailable")

    available = asyncio.run(refresh_claude_catalog(
        config_path=config_path,
        output_path=output_path,
        probe=failing_probe,
    ))

    assert available == []
    assert json.loads(output_path.read_text()) == {"models": []}
