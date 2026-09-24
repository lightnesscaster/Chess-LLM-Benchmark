"""Web-play models come from the live repository config without a redeploy."""

import json

import pytest
import yaml

import web.live_models as live_models
from web import claude_catalog
from web.play_service import list_playable_models


BUNDLED = {"engines": [{"player_id": "random-bot"}], "web_play_models": [], "llms": [
    {"player_id": "bundled-model", "model_name": "vendor/bundled"},
]}
LIVE = {"web_play_models": [], "llms": [
    {"player_id": "bundled-model", "model_name": "vendor/bundled"},
    {"player_id": "new-model (high)", "model_name": "vendor/new", "reasoning_effort": "high"},
]}


@pytest.fixture(autouse=True)
def fresh_cache():
    live_models.clear_cache()
    yield
    live_models.clear_cache()


def test_live_config_is_off_outside_render():
    calls = []
    config = live_models.apply_live_models(BUNDLED, {}, fetch=calls.append)
    assert config is BUNDLED
    assert calls == []


def test_live_model_sections_replace_bundled_ones_and_are_cached():
    calls = []

    def fetch(url):
        calls.append(url)
        return yaml.safe_dump(LIVE)

    for _ in range(2):
        config = live_models.apply_live_models(BUNDLED, {"RENDER": "true"}, fetch=fetch)
        assert config["llms"] == LIVE["llms"]
        assert config["engines"] == BUNDLED["engines"]
    assert calls == [live_models.DEFAULT_LIVE_CONFIG_URL]


def test_unreachable_or_invalid_live_config_keeps_last_good_copy(monkeypatch):
    environ = {"LIVE_MODEL_CONFIG_URL": "https://example.test/benchmark.yaml"}

    def fail(url):
        raise OSError("offline")

    assert live_models.apply_live_models(BUNDLED, environ, fetch=fail) == BUNDLED
    live_models.clear_cache()
    live_models.apply_live_models(BUNDLED, environ, fetch=lambda url: yaml.safe_dump(LIVE))
    monkeypatch.setattr(live_models, "CACHE_SECONDS", 0)
    for broken in (fail, lambda url: "not: [a, model, config"):
        assert live_models.apply_live_models(BUNDLED, environ, fetch=broken)["llms"] == LIVE["llms"]


def test_arena_lists_a_model_pushed_after_deploy(tmp_path, monkeypatch):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(BUNDLED))
    monkeypatch.setattr(live_models, "_fetch", lambda url: yaml.safe_dump(LIVE))
    environ = {"OPENROUTER_API_KEY": "test"}
    assert [m["id"] for m in list_playable_models(config_path, environ)] == ["bundled-model"]
    live = list_playable_models(config_path, {**environ, "RENDER": "true"})
    assert [m["id"] for m in live] == ["bundled-model", "new-model"]


def test_new_claude_models_are_probed_once_in_background(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_catalog, "_failed_at", {})
    catalog = tmp_path / "claude-models.json"
    catalog.write_text(json.dumps({"models": ["claude-opus-5"]}))
    probed = []

    async def probe(name):
        probed.append(name)
        return name == "claude-opus-5-5"

    names = ["claude-opus-5", "claude-opus-5-5", "claude-missing"]
    claude_catalog.probe_new_models_in_background(names, catalog, probe).join()
    assert probed == ["claude-opus-5-5", "claude-missing"]
    assert json.loads(catalog.read_text())["models"] == ["claude-opus-5", "claude-opus-5-5"]
    # Verified and recently failed models are not probed again.
    assert claude_catalog.probe_new_models_in_background(names, catalog, probe) is None
