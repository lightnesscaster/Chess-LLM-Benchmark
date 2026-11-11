"""Model metadata refreshes without replacing the running application's files."""

import json

import pytest

import model_metadata as metadata
import web.live_models as live_models
from rating.cost_calculator import CostCalculator
from rating.leaderboard import Leaderboard
from tests.test_leaderboard_metadata import FakeRatingStore


@pytest.fixture(autouse=True)
def isolated_metadata(monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("LIVE_PRICING_URL", raising=False)
    monkeypatch.delenv("LIVE_MODEL_PUBLISH_DATES_URL", raising=False)
    monkeypatch.setattr(metadata, "_cache", {})
    live_models.clear_cache()
    yield
    live_models.clear_cache()


def test_local_and_explicit_custom_files_do_not_fetch(monkeypatch, tmp_path):
    def unexpected_fetch(url):
        pytest.fail(f"Unexpected remote read: {url}")

    monkeypatch.setattr(metadata, "_fetch", unexpected_fetch)
    assert metadata.load_pricing()["google/gemini-4-carbon"]["prompt"] == 0.000002
    custom = tmp_path / "prices.json"
    custom.write_text('{"vendor/custom": {"prompt": 1, "completion": 2}}')
    monkeypatch.setenv("RENDER", "true")
    assert metadata.load_pricing(custom)["vendor/custom"]["completion"] == 2


def test_openrouter_unknown_price_sentinels_do_not_reject_other_live_prices(monkeypatch):
    monkeypatch.setenv("LIVE_PRICING_URL", "https://example.test/pricing.json")
    monkeypatch.setattr(metadata, "_fetch", lambda url: json.dumps({
        "openrouter/auto": {"prompt": -1.0, "completion": -1.0},
        "google/gemini-4-carbon": {"prompt": 0.000004, "completion": 0.000020},
    }))
    assert metadata.load_pricing()["google/gemini-4-carbon"]["prompt"] == 0.000004


def test_concurrent_reads_serve_last_good_data_during_slow_refresh(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    monkeypatch.setenv("LIVE_PRICING_URL", "https://example.test/pricing.json")
    now = [0.0]
    monkeypatch.setattr(metadata, "monotonic", lambda: now[0])
    monkeypatch.setattr(metadata, "_fetch", lambda url: '{"vendor/new": {"prompt": 2, "completion": 10}}')
    assert metadata.load_pricing()["vendor/new"]["prompt"] == 2
    started, release = Event(), Event()

    def slow_fetch(url):
        started.set()
        assert release.wait(5)
        return '{"vendor/new": {"prompt": 4, "completion": 20}}'

    monkeypatch.setattr(metadata, "_fetch", slow_fetch)
    now[0] += 121
    with ThreadPoolExecutor(max_workers=2) as pool:
        refresh = pool.submit(metadata.load_pricing)
        assert started.wait(1)
        try:
            assert pool.submit(metadata.load_pricing).result(timeout=1)["vendor/new"]["prompt"] == 2
        finally:
            release.set()
        assert refresh.result(timeout=1)["vendor/new"]["prompt"] == 4


def test_refresh_reprices_games_and_invalidates_cached_pages(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    now = [1000.0]
    monkeypatch.setattr(metadata, "monotonic", lambda: now[0])
    monkeypatch.setattr(metadata, "time", lambda: now[0])
    prices = {"vendor/new": {"prompt": 0.000002, "completion": 0.000010}}
    dates = {"new (high)": {"model_id": "vendor/new", "created_date": "2026-11-10", "created_timestamp": 1794268800}}
    calls = []

    def fetch(url):
        calls.append(url)
        return json.dumps(prices if url.endswith("pricing.json") else dates)

    monkeypatch.setattr(metadata, "_fetch", fetch)
    monkeypatch.setattr(live_models, "_fetch", lambda url: json.dumps({
        "web_play_models": [], "llms": [{"player_id": "new (high)", "model_name": "vendor/new", "reasoning_effort": "high"}],
    }))
    calculator = CostCalculator()
    assert calculator.get_model_for_player("new (high)") == "vendor/new"
    assert calculator.calculate_game_cost({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}, "vendor/new") == 12
    assert metadata.load_publish_dates()["new (high)"]["created_date"] == "2026-11-10"
    assert not metadata.metadata_changed_since(1001)
    assert len(calls) == 2

    # A corrected price is observable after expiry, even with a warm page cache.
    now[0] = 1121
    prices["vendor/new"]["completion"] = 0.000020
    assert metadata.metadata_changed_since(1001)
    assert CostCalculator().calculate_game_cost({"completion_tokens": 1_000_000}, "vendor/new") == 20
    assert len(calls) == 4
    # An unchanged refresh must not rebuild the expensive leaderboard.
    now[0] = 1242
    assert not metadata.metadata_changed_since(1122)


def test_warm_leaderboard_refreshes_on_metadata_change_only(monkeypatch):
    from types import SimpleNamespace
    import web.app as web_app

    monkeypatch.setenv("LIVE_MODEL_PUBLISH_DATES_URL", "https://example.test/dates.json")
    now = [0.0]
    monkeypatch.setattr(metadata, "monotonic", lambda: now[0])
    dates = {"gemini-3.6-flash (medium)": {
        "model_id": "google/gemini-3.6-flash", "created_date": "2026-07-07", "created_timestamp": 1783382400,
    }}
    monkeypatch.setattr(metadata, "_fetch", lambda url: json.dumps(dates))
    reads = []

    def read_results():
        reads.append(True)
        return []

    monkeypatch.setattr(web_app, "RatingStore", lambda **_: FakeRatingStore())
    monkeypatch.setattr(web_app, "PGNLogger", lambda: SimpleNamespace(load_all_results=read_results))
    monkeypatch.setattr(web_app, "get_anchors_from_config", lambda: {})
    monkeypatch.setattr(web_app, "get_all_engine_ids_from_config", lambda: set())
    for key, value in (("_leaderboard_cache", []), ("_leaderboard_cache_time", 0), ("_leaderboard_cache_min_games", None), ("_leaderboard_refreshing", False)):
        monkeypatch.setattr(web_app, key, value)
    assert web_app.get_leaderboard_data()[0]["publish_date"] == "07/26"
    dates["gemini-3.6-flash (medium)"].update(created_date="2026-11-10", created_timestamp=1794268800)
    now[0] += 121
    assert web_app.get_leaderboard_data()[0]["publish_date"] == "11/26"
    now[0] += 121
    assert web_app.get_leaderboard_data()[0]["publish_date"] == "11/26"
    assert len(reads) == 2


@pytest.mark.parametrize("broken", ["offline", "{bad json", "[]", "{}", '{"vendor/new": {"prompt": -1, "completion": 2}}', '{"vendor/new": {"prompt": "2", "completion": 2}}', '{"vendor/new": {"prompt": NaN, "completion": 2}}'])
def test_failed_or_invalid_refresh_preserves_last_good_prices(monkeypatch, broken):
    monkeypatch.setenv("LIVE_PRICING_URL", "https://example.test/pricing.json")
    now = [0.0]
    monkeypatch.setattr(metadata, "monotonic", lambda: now[0])
    payload = [broken]

    def fetch(url):
        if payload[0] == "offline":
            raise OSError("offline")
        return payload[0]

    monkeypatch.setattr(metadata, "_fetch", fetch)
    assert "google/gemini-4-carbon" in metadata.load_pricing()
    now[0] += 121
    payload[0] = '{"vendor/new": {"prompt": 2, "completion": 10}}'
    assert metadata.load_pricing()["vendor/new"]["completion"] == 10
    now[0] += 121
    payload[0] = broken
    assert metadata.load_pricing()["vendor/new"]["completion"] == 10


def test_leaderboard_and_charts_use_live_release_metadata(monkeypatch):
    from web.timeline_chart import build_timeline_data
    from web.cost_chart import create_cost_chart

    monkeypatch.setenv("LIVE_MODEL_PUBLISH_DATES_URL", "https://example.test/dates.json")
    dates = {"new (high)": {
        "model_id": "google/new", "created_date": "2026-11-10", "created_timestamp": 1794268800,
        "display_cost_per_game": 0.25, "display_cost_estimated": True,
    }}
    monkeypatch.setattr(metadata, "_fetch", lambda url: json.dumps(dates))
    store = FakeRatingStore()
    rating = store.get_sorted_ratings()[0]
    rating.player_id = "new (high)"
    store.get_sorted_ratings = lambda **_: [rating]
    entry = Leaderboard(store).get_leaderboard()[0]
    assert entry["publish_date"] == "11/26"
    assert entry["publish_timestamp"] == 1794268800
    assert entry["avg_cost_per_game"] == 0.25
    assert any(trace.name == "Google" for trace in create_cost_chart([entry]).data)
    timeline = build_timeline_data([entry])
    assert [point["lab_name"] for point in timeline["points"]] == ["Google"]
    assert timeline["points"][0]["ts"] == 1794268800
    assert [record["id"] for record in timeline["records"]] == ["new (high)"]


def test_live_dates_reach_rating_seeding_freeze_and_scheduling(monkeypatch):
    from game.freeze_checker import FreezeChecker
    from game.match_scheduler import MatchScheduler
    from rating.rating_store import RatingStore

    monkeypatch.setenv("LIVE_MODEL_PUBLISH_DATES_URL", "https://example.test/dates.json")
    monkeypatch.setattr(metadata, "_fetch", lambda url: json.dumps({"new (low)": {
        "model_id": "vendor/new", "created_date": "2026-11-10", "created_timestamp": 1794268800,
    }}))
    for cls in (FreezeChecker, MatchScheduler, RatingStore):
        consumer = cls.__new__(cls)
        for field in ("_publish_dates", "_player_providers", "_models_by_provider", "_player_model_ids", "_models_by_model_id"):
            setattr(consumer, field, {})
        if cls is RatingStore:
            consumer._load_model_id_mapping()
        else:
            consumer._load_publish_dates()
            assert consumer._publish_dates["new (low)"] == 1794268800
        assert consumer._player_model_ids["new (low)"] == "vendor/new"


def test_invalid_release_dates_fall_back_and_empty_override_disables_live(monkeypatch):
    monkeypatch.setenv("LIVE_MODEL_PUBLISH_DATES_URL", "https://example.test/dates.json")
    monkeypatch.setattr(metadata, "_fetch", lambda url: '{"new": {"model_id": "vendor/new", "created_date": "not-a-date", "created_timestamp": 1}}')
    assert "gemini-4-carbon (medium)" in metadata.load_publish_dates()
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("LIVE_MODEL_PUBLISH_DATES_URL", "")
    assert "gemini-4-carbon (medium)" in metadata.load_publish_dates()
