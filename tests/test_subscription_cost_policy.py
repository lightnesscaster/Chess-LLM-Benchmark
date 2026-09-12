import pytest

from rating.cost_calculator import CostCalculator


@pytest.mark.parametrize("player", [
    "gpt-6-astra (medium)", "gpt-6-astra (max)",
    "claude-fable-5.1 (high)", "claude-fable-5 (medium)",
])
def test_premium_subscription_models_use_token_cost(player):
    calculator = CostCalculator()
    costs = {}
    calculator._add_player_cost(costs, player, {
        "prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000,
    })
    assert calculator.get_budget_cost_override(player) is None
    assert costs[player]["total_cost"] == pytest.approx(0.06)
    assert costs[player]["games_with_cost"] == 1


@pytest.mark.parametrize("player", [
    "gpt-5.6-luna (xhigh)", "gpt-5.6-sol (medium)",
    "claude-opus-5 (high)", "claude-sonnet-5 (max)", "claude-haiku-4.5 (low)",
])
def test_other_subscription_models_have_zero_budget_cost(player):
    calculator = CostCalculator()
    costs = {}
    calculator._add_player_cost(costs, player, {"prompt_tokens": 1000, "completion_tokens": 1000})
    assert calculator.get_budget_cost_override(player) == 0.0
    assert costs[player]["total_cost"] == 0.0
    assert costs[player]["games_with_cost"] == 1
