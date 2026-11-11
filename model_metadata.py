"""Load current model metadata on Render, with local files as offline fallbacks."""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import date
from pathlib import Path
import threading
from time import monotonic, time

import requests


logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
REPOSITORY_URL = "https://raw.githubusercontent.com/lightnesscaster/Chess-LLM-Benchmark/main"
CACHE_SECONDS = 120
_PATHS = {
    "pricing": ROOT / "config/pricing.json",
    "publish_dates": ROOT / "data/model_publish_dates.json",
}
_SOURCES = {
    "pricing": ("LIVE_PRICING_URL", f"{REPOSITORY_URL}/config/pricing.json"),
    "publish_dates": ("LIVE_MODEL_PUBLISH_DATES_URL", f"{REPOSITORY_URL}/data/model_publish_dates.json"),
}
_lock = threading.Lock()
_cache: dict[tuple[str, str], dict] = {}


def _source(kind: str) -> str | None:
    variable, default = _SOURCES[kind]
    if variable in os.environ:
        return os.environ[variable].strip() or None
    return default if os.environ.get("RENDER") else None


def _fetch(url: str) -> str:
    response = requests.get(url, timeout=3)
    response.raise_for_status()
    return response.text


def _nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value >= 0
    )


def _validate(data: object, kind: str) -> bool:
    if not isinstance(data, dict) or not data:
        return False
    for key, entry in data.items():
        if not isinstance(key, str) or not key or not isinstance(entry, dict):
            return False
        if kind == "pricing":
            if not {"prompt", "completion"} <= entry.keys():
                return False
            # OpenRouter's dynamic routers use this pair for unknown rates.
            if entry == {"prompt": -1.0, "completion": -1.0}:
                continue
            if not all(_nonnegative_number(value) for value in entry.values()):
                return False
        else:
            if not isinstance(entry.get("model_id"), str) or not entry["model_id"]:
                return False
            if not _nonnegative_number(entry.get("created_timestamp")):
                return False
            if not isinstance(entry.get("created_date"), str) or len(entry["created_date"]) != 10:
                return False
            try:
                date.fromisoformat(entry["created_date"])
            except (KeyError, TypeError, ValueError):
                return False
            if "display_cost_per_game" in entry and not _nonnegative_number(entry["display_cost_per_game"]):
                return False
    return True


def _live_snapshot(kind: str) -> dict | None:
    url = _source(kind)
    if not url:
        return None
    key = (kind, url)
    # One caller refreshes; concurrent readers keep serving the previous snapshot.
    if not _lock.acquire(blocking=False):
        return _cache.get(key)
    try:
        now = monotonic()
        cached = _cache.get(key)
        if cached is not None and now - cached["checked_at"] < CACHE_SECONDS:
            return cached
        data = cached["data"] if cached else None
        changed_at = cached["changed_at"] if cached else 0.0
        try:
            fetched = json.loads(_fetch(url))
            if not _validate(fetched, kind):
                raise ValueError(f"invalid {kind} metadata")
            if fetched != data:
                data = fetched
                changed_at = time()
        except Exception as error:
            logger.warning("Live %s unavailable: %s", kind, error)
        snapshot = {"data": data, "checked_at": monotonic(), "changed_at": changed_at}
        _cache[key] = snapshot
        return snapshot
    finally:
        _lock.release()


def _load(kind: str, path: str | Path | None) -> dict:
    selected = Path(path) if path is not None else _PATHS[kind]
    # Explicit alternate files are used by local experiments and backtests.
    if selected.resolve() == _PATHS[kind].resolve():
        snapshot = _live_snapshot(kind)
        if snapshot and snapshot["data"] is not None:
            return snapshot["data"]
    try:
        with selected.open() as source:
            return json.load(source)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_pricing(path: str | Path | None = None) -> dict:
    """Return cached live token rates, or the selected local pricing file."""
    return _load("pricing", path)


def load_publish_dates(path: str | Path | None = None) -> dict:
    """Return cached live release metadata, or the selected local metadata file."""
    return _load("publish_dates", path)


def metadata_changed_since(cache_time: float) -> bool:
    """Refresh expired sources and report whether a rendered page is outdated."""
    snapshots = [_live_snapshot(kind) for kind in _SOURCES]
    return any(snapshot and snapshot["changed_at"] > cache_time for snapshot in snapshots)
