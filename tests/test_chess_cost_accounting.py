import pytest

from rating.cost_calculator import CostCalculator


def calculator():
    value = CostCalculator()
    value.pricing["test"] = {"prompt": 1.0, "completion": 2.0,
                             "input_cache_read": 0.1, "input_cache_write": 1.25}
    return value


def test_chess_cost_excludes_runtime_and_prices_all_input_categories():
    usage = {"prompt_tokens": 10000, "completion_tokens": 100,
             "chess_prompt_tokens": 500, "chess_cached_input_tokens": 300,
             "chess_cache_creation_input_tokens": 50}
    assert calculator().calculate_game_cost(usage, "test") == pytest.approx(442.5)
    assert usage["prompt_tokens"] == 10000


def test_direct_api_cached_input_is_not_double_charged():
    assert calculator().calculate_game_cost({"prompt_tokens": 1000,
        "cached_input_tokens": 800, "completion_tokens": 100}, "test") == 480


def test_missing_cache_rate_uses_input_rate_and_labels_estimate():
    value = calculator()
    value.pricing["test"].pop("input_cache_read")
    value.player_to_model["api-player"] = "test"
    costs = {}
    value._add_player_cost(costs, "api-player", {"prompt_tokens": 1000,
        "completion_tokens": 100, "cached_input_tokens": 800,
        "cache_accounting_known": True})
    assert costs["api-player"]["total_cost"] == 1200
    assert costs["api-player"]["cost_estimated"]


def test_cache_write_ttl_uses_separate_rates():
    value = calculator()
    value.pricing["test"]["input_cache_write_1h"] = 2.0
    assert value.calculate_game_cost({"prompt_tokens": 100,
        "cache_creation_input_tokens": 100, "cache_creation_5m_input_tokens": 40,
        "cache_creation_1h_input_tokens": 60}, "test") == 170


def test_pricing_refresh_preserves_cache_rates(monkeypatch):
    from types import SimpleNamespace
    from scripts import fetch_pricing
    monkeypatch.setattr(fetch_pricing.requests, "get", lambda *a, **k: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"data": [{"id": "test",
        "pricing": {"prompt": "1", "completion": "2", "input_cache_read": "0.1",
                    "input_cache_write": "1.25", "input_cache_write_1h": "2"}}]}))
    assert fetch_pricing.fetch_pricing()["test"]["input_cache_read"] == 0.1


def test_legacy_runtime_input_is_not_priced_as_chess_or_subtracted_by_constant():
    value = CostCalculator()
    costs = {}
    value._add_player_cost(costs, "gpt-6-astra (medium)",
        {"prompt_tokens": 400000, "completion_tokens": 1000}, move_count=32)
    row = costs["gpt-6-astra (medium)"]
    assert row["total_cost"] == pytest.approx(0.05)
    assert row["cost_lower_bound"]
    assert row["games_missing_chess_input"] == 1


def test_estimated_chess_counts_are_labeled_and_budget_override_unchanged():
    value = CostCalculator()
    usage = {"prompt_tokens": 12000, "completion_tokens": 100,
             "chess_prompt_tokens": 500, "input_accounting_method": "estimated-test",
             "cache_accounting_known": False}
    costs = {}
    value._add_player_cost(costs, "gpt-5.6-sol (medium)", usage,
                           use_budget_overrides=False)
    assert costs["gpt-5.6-sol (medium)"]["cost_estimated"]
    assert costs["gpt-5.6-sol (medium)"]["total_cost"] == pytest.approx(0.0055)
    costs = {}
    value._add_player_cost(costs, "gpt-5.6-sol (medium)", usage)
    assert costs["gpt-5.6-sol (medium)"]["total_cost"] == 0
