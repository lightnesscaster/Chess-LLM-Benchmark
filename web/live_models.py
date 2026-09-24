"""Serve web-play model entries from the repository's live config.

Adding a model is a push to ``config/benchmark.yaml`` on ``main``. The deployed
site reads the model sections from GitHub (cached briefly) so new entries are
playable without a redeploy; the bundled file stays the fallback.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping

import requests
import yaml


logger = logging.getLogger(__name__)

DEFAULT_LIVE_CONFIG_URL = (
    "https://raw.githubusercontent.com/lightnesscaster/"
    "Chess-LLM-Benchmark/main/config/benchmark.yaml"
)
MODEL_SECTIONS = ("web_play_models", "llms")
CACHE_SECONDS = 120
FETCH_TIMEOUT_SECONDS = 3

Fetch = Callable[[str], str]

_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, dict | None]] = {}


def live_config_url(environ: Mapping[str, str]) -> str | None:
    """Return the live config URL, enabled by default only on Render."""
    configured = environ.get("LIVE_MODEL_CONFIG_URL")
    if configured is not None:
        return configured.strip() or None
    return DEFAULT_LIVE_CONFIG_URL if environ.get("RENDER") else None


def _fetch(url: str) -> str:
    response = requests.get(url, timeout=FETCH_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def _model_sections(text: str) -> dict | None:
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        return None
    sections = {key: config.get(key) for key in MODEL_SECTIONS}
    if not all(isinstance(value, list) for value in sections.values()):
        return None
    return sections


def _live_sections(url: str, fetch: Fetch) -> dict | None:
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(url)
        if cached and now - cached[0] < CACHE_SECONDS:
            return cached[1]
    try:
        sections = _model_sections(fetch(url))
        if sections is None:
            raise ValueError("live config has no model sections")
    except Exception as error:  # keep serving the last good copy
        logger.warning("Live model config unavailable: %s", error)
        sections = cached[1] if cached else None
    with _cache_lock:
        _cache[url] = (now, sections)
    return sections


def apply_live_models(
    config: dict,
    environ: Mapping[str, str],
    fetch: Fetch | None = None,
) -> dict:
    """Replace the bundled model sections with the live ones when available."""
    url = live_config_url(environ)
    if not url:
        return config
    sections = _live_sections(url, fetch or _fetch)
    return {**config, **sections} if sections else config


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
