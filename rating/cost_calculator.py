"""
Cost calculation for LLM players based on token usage and OpenRouter pricing.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Protocol

import yaml

from utils import resolve_player_id

from game.models import GameResult

logger = logging.getLogger(__name__)


# Default threshold for filtering games by rating difference
DEFAULT_COST_RATING_THRESHOLD = 600


class RatingProvider(Protocol):
    """Protocol for objects that can provide player ratings."""
    def get(self, player_id: str) -> Optional[Any]:
        """Get rating info for a player. Returns object with .rating attribute or None."""
        ...


def filter_results_by_rating_diff(
    results: List[GameResult],
    rating_provider: RatingProvider,
    max_diff: int = DEFAULT_COST_RATING_THRESHOLD,
) -> List[GameResult]:
    """
    Filter game results to only include games where opponents are within rating threshold.

    This is used for cost calculations to get more accurate cost estimates by excluding
    mismatched games (e.g., strong LLM vs random-bot) which tend to be shorter/cheaper.

    Args:
        results: List of game results to filter
        rating_provider: Object with get(player_id) method returning rating info
        max_diff: Maximum rating difference to include (default: 600)

    Returns:
        Filtered list of game results where both players' ratings are within max_diff
    """
    filtered = []
    skipped_missing_rating = 0

    for result in results:
        white_rating = rating_provider.get(result.white_id)
        black_rating = rating_provider.get(result.black_id)

        if not white_rating or not black_rating:
            skipped_missing_rating += 1
            continue

        diff = abs(white_rating.rating - black_rating.rating)
        if diff <= max_diff:
            filtered.append(result)

    if skipped_missing_rating > 0:
        logger.debug(
            f"Cost filtering: skipped {skipped_missing_rating} games due to missing ratings"
        )

    logger.debug(
        f"Cost filtering: {len(filtered)}/{len(results)} games within {max_diff} rating diff"
    )

    return filtered


class CostCalculator:
    """
    Calculates costs based on token usage and model pricing.
    """

    # Project root directory (parent of rating/)
    _PROJECT_ROOT = Path(__file__).parent.parent
    # Codex CLI usage includes a large fixed prompt wrapper for every subagent
    # call. Display costs should price the chess prompt, not that local runtime
    # wrapper, while benchmark budget enforcement can still use explicit
    # budget_cost_per_game overrides.
    CODEX_FIXED_PROMPT_OVERHEAD_TOKENS = 16000

    def __init__(
        self,
        pricing_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize cost calculator.

        Args:
            pricing_path: Path to pricing.json (default: config/pricing.json)
            config_path: Path to benchmark.yaml for player_id -> model_name mapping
        """
        if pricing_path is None:
            pricing_path = self._PROJECT_ROOT / "config" / "pricing.json"
        if config_path is None:
            config_path = self._PROJECT_ROOT / "config" / "benchmark.yaml"
        self.pricing = self._load_pricing(pricing_path)
        self.player_to_model: Dict[str, str] = {}
        self.player_budget_cost_overrides: Dict[str, float] = {}
        self.player_exclude_runtime_token_cost: set[str] = set()
        self._load_player_config(config_path)

    def _load_pricing(self, path: str) -> Dict[str, Dict[str, float]]:
        """Load pricing data from JSON file."""
        pricing_file = Path(path)
        if not pricing_file.exists():
            return {}

        with open(pricing_file) as f:
            return json.load(f)

    def _load_player_config(self, config_path: str) -> None:
        """Load player cost metadata from benchmark config."""
        config_file = Path(config_path)
        if not config_file.exists():
            return

        with open(config_file) as f:
            config = yaml.safe_load(f)

        for llm in config.get("llms", []):
            player_id = llm.get("player_id")
            model_name = llm.get("model_name")
            if not player_id:
                continue

            player_ids = {player_id}
            full_id = resolve_player_id(player_id, llm.get("reasoning_effort"))
            player_ids.add(full_id)

            for pid in player_ids:
                if model_name:
                    self.player_to_model[pid] = model_name

                if llm.get("exclude_runtime_tokens_from_cost", False):
                    self.player_exclude_runtime_token_cost.add(pid)

                if "budget_cost_per_game" in llm:
                    try:
                        self.player_budget_cost_overrides[pid] = float(llm["budget_cost_per_game"])
                    except (TypeError, ValueError):
                        logger.warning(
                            "Invalid budget_cost_per_game for %s: %r",
                            pid,
                            llm["budget_cost_per_game"],
                        )

    def get_model_for_player(self, player_id: str, _depth: int = 0) -> Optional[str]:
        """Get the OpenRouter model name for a player_id."""
        # Prevent infinite recursion
        if _depth > 3:
            return None

        # Direct lookup
        if player_id in self.player_to_model:
            return self.player_to_model[player_id]

        # Try to find in pricing by partial match (e.g., "gpt-4" in "openai/gpt-4")
        for model_id in self.pricing:
            if player_id in model_id or model_id.endswith(f"/{player_id}"):
                return model_id

        # Strip common suffixes and try again
        # e.g., "gpt-5.1 (high)" -> "gpt-5.1", "claude-opus-4 (no thinking)" -> "claude-opus-4"
        base_id = re.sub(r'\s*\((max|xhigh|high|medium|low|minimal|thinking|no thinking)\)\s*$', '', player_id)
        if base_id != player_id:
            return self.get_model_for_player(base_id, _depth + 1)

        # Try with :free suffix (some models are free variants)
        for model_id in self.pricing:
            if model_id.endswith(f"/{player_id}:free"):
                return model_id

        return None

    def get_pricing(self, model_name: str) -> Optional[Dict[str, float]]:
        """Get pricing for a model."""
        return self.pricing.get(model_name)

    def get_budget_cost_override(self, player_id: str) -> Optional[float]:
        """Get explicit benchmark budget cost for a player, if configured."""
        return self.player_budget_cost_overrides.get(player_id)

    def exclude_runtime_tokens_from_cost(self, player_id: str) -> bool:
        """Whether runtime token usage should be ignored for cost accounting."""
        return player_id in self.player_exclude_runtime_token_cost

    def calculate_game_cost(
        self,
        tokens: Optional[Dict[str, int]],
        model_name: str,
    ) -> Optional[float]:
        """
        Calculate cost for a game's token usage.

        Args:
            tokens: Token usage dict with prompt_tokens, completion_tokens
            model_name: OpenRouter model name

        Returns:
            Cost in dollars, or None if pricing not available
        """
        if not tokens:
            return None

        pricing = self.get_pricing(model_name)
        if not pricing:
            return None

        prompt_tokens = tokens.get("prompt_tokens", 0)
        completion_tokens = tokens.get("completion_tokens", 0)

        prompt_cost = prompt_tokens * pricing.get("prompt", 0)
        completion_cost = completion_tokens * pricing.get("completion", 0)

        return prompt_cost + completion_cost

    def calculate_player_costs(
        self,
        results: List[GameResult],
        use_budget_overrides: bool = True,
        subtract_excluded_prompt_overhead: bool = False,
        include_uncosted_players: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate total costs per player from game results.

        Args:
            results: List of game results
            use_budget_overrides: Whether to use configured budget cost overrides.
                Benchmark scheduling uses this; display code should pass False so
                "$/Game" reflects token-priced usage instead of budget caps.
            subtract_excluded_prompt_overhead: For players with
                exclude_runtime_tokens_from_cost, subtract the Codex CLI fixed
                prompt overhead before pricing display costs. This still prices
                the remaining chess prompt input tokens plus output tokens.
            include_uncosted_players: Include players whose token cost could not be
                priced. Display code should pass False so unknown costs render as "-".

        Returns:
            Dict mapping player_id to cost stats:
            {
                "total_cost": float,
                "games_with_cost": int,
                "avg_cost_per_game": float,
                "total_tokens": int,
            }
        """
        player_costs: Dict[str, Dict[str, Any]] = {}

        for result in results:
            # Process white player
            if result.tokens_white:
                self._add_player_cost(
                    player_costs,
                    result.white_id,
                    result.tokens_white,
                    use_budget_overrides=use_budget_overrides,
                    subtract_excluded_prompt_overhead=subtract_excluded_prompt_overhead,
                    include_uncosted_players=include_uncosted_players,
                    move_count=self._get_move_count(
                        result.timing_white,
                        result.total_moves_white,
                    ),
                )

            # Process black player
            if result.tokens_black:
                self._add_player_cost(
                    player_costs,
                    result.black_id,
                    result.tokens_black,
                    use_budget_overrides=use_budget_overrides,
                    subtract_excluded_prompt_overhead=subtract_excluded_prompt_overhead,
                    include_uncosted_players=include_uncosted_players,
                    move_count=self._get_move_count(
                        result.timing_black,
                        result.total_moves_black,
                    ),
                )

        # Calculate averages
        for player_id, stats in player_costs.items():
            if stats["games_with_cost"] > 0:
                stats["avg_cost_per_game"] = (
                    stats["total_cost"] / stats["games_with_cost"]
                )
            else:
                stats["avg_cost_per_game"] = 0.0

        return player_costs

    def _add_player_cost(
        self,
        player_costs: Dict[str, Dict[str, Any]],
        player_id: str,
        tokens: Dict[str, int],
        use_budget_overrides: bool = True,
        include_uncosted_players: bool = True,
        subtract_excluded_prompt_overhead: bool = False,
        move_count: int = 0,
    ) -> None:
        """Add cost for a single game to player totals."""
        cost = None

        if use_budget_overrides and self.exclude_runtime_tokens_from_cost(player_id):
            override = self.get_budget_cost_override(player_id)
            if override is not None:
                cost = override
        else:
            model_name = self.get_model_for_player(player_id)
            if model_name:
                priced_tokens = tokens
                if (
                    subtract_excluded_prompt_overhead and
                    self.exclude_runtime_tokens_from_cost(player_id)
                ):
                    priced_tokens = self._subtract_fixed_prompt_overhead(
                        tokens,
                        move_count,
                    )
                cost = self.calculate_game_cost(priced_tokens, model_name)

        if cost is None and not include_uncosted_players:
            return

        if player_id not in player_costs:
            player_costs[player_id] = {
                "total_cost": 0.0,
                "games_with_cost": 0,
                "avg_cost_per_game": 0.0,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        if cost is not None:
            player_costs[player_id]["total_cost"] += cost
            player_costs[player_id]["games_with_cost"] += 1

        player_costs[player_id]["total_tokens"] += tokens.get("total_tokens", 0)
        player_costs[player_id]["prompt_tokens"] += tokens.get("prompt_tokens", 0)
        player_costs[player_id]["completion_tokens"] += tokens.get("completion_tokens", 0)

    @staticmethod
    def _get_move_count(timing: Optional[dict], total_moves: int) -> int:
        """Return the best available count of LLM calls made by one side."""
        if timing:
            try:
                return max(0, int(timing.get("move_count", 0) or 0))
            except (TypeError, ValueError):
                pass

        try:
            return max(0, int(total_moves or 0))
        except (TypeError, ValueError):
            return 0

    def _subtract_fixed_prompt_overhead(
        self,
        tokens: Dict[str, int],
        move_count: int,
    ) -> Dict[str, int]:
        """Remove Codex CLI fixed prompt overhead while preserving chess input tokens."""
        try:
            prompt_tokens = int(tokens.get("prompt_tokens", 0) or 0)
        except (TypeError, ValueError):
            prompt_tokens = 0

        try:
            completion_tokens = int(tokens.get("completion_tokens", 0) or 0)
        except (TypeError, ValueError):
            completion_tokens = 0

        calls = max(0, int(move_count or 0))
        overhead = calls * self.CODEX_FIXED_PROMPT_OVERHEAD_TOKENS
        adjusted_prompt_tokens = max(0, prompt_tokens - overhead)

        return {
            **tokens,
            "prompt_tokens": adjusted_prompt_tokens,
            "total_tokens": adjusted_prompt_tokens + completion_tokens,
        }
