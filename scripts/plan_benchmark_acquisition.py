#!/usr/bin/env python3
"""Preview automatic benchmark acquisition without model calls."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game.freeze_checker import FreezeChecker  # noqa: E402
from game.pgn_logger import PGNLogger  # noqa: E402
from game.stats_collector import StatsCollector  # noqa: E402
from position_benchmark.acquisition import (  # noqa: E402
    AcquisitionPanel,
    AcquisitionPolicy,
    load_acquisition_policy,
    load_acquisition_state,
    missing_panels,
)
from rating.cost_calculator import CostCalculator  # noqa: E402
from rating.rating_store import RatingStore  # noqa: E402
from utils import is_reasoning_model, resolve_player_id  # noqa: E402


UNKNOWN_FULL_CORE_COST = 1.0
PRIORITY_COST_SENSITIVITY = 0.5


@dataclass(frozen=True)
class AcquisitionPlanRow:
    """One configured model's read-only acquisition decision."""

    player_id: str
    status: str
    rating: float
    rating_deviation: float
    priority: float
    measured_panels: tuple[str, ...]
    publish_saved_panels: tuple[str, ...]
    source_player_id: str | None
    policy_deferred_panels: tuple[str, ...]
    conditional_panels: tuple[str, ...]
    missing_panels: tuple[str, ...]
    scheduled_panels: tuple[str, ...]
    deferred_panels: tuple[str, ...]
    planned_first_attempt_calls: int
    maximum_retry_calls: int
    estimated_base_cost: float


def selected_llm_configs(
    config: dict[str, Any],
    api_backend: str,
) -> dict[str, dict[str, Any]]:
    """Return the same normalized LLM configuration set selected by ``cli.py run``."""
    selected: dict[str, dict[str, Any]] = {}
    for raw in config.get("llms", []):
        if raw.get("unavailable") is True:
            continue
        configured_api = raw.get("api", "openrouter")
        if api_backend == "codex" and configured_api != "codex":
            continue
        if api_backend == "gemini" and configured_api != "gemini":
            continue
        if (
            api_backend == "openrouter"
            and configured_api in {"codex", "gemini"}
        ):
            continue
        player_id = resolve_player_id(raw["player_id"], raw.get("reasoning_effort"))
        selected[player_id] = raw
    return selected


def reasoning_player_ids(configs: dict[str, dict[str, Any]]) -> set[str]:
    """Apply the production reasoning-model classification without creating clients."""
    result: set[str] = set()
    for player_id, config in configs.items():
        reasoning = config.get("reasoning")
        if (
            config.get("reasoning_effort") is not None
            or config.get("reasoning_max_tokens") is not None
            or reasoning is True
            or (reasoning is not False and is_reasoning_model(player_id))
        ):
            result.add(player_id)
    return result


def panel_retry_upper_bound(panel: AcquisitionPanel) -> int:
    """Return the maximum conditional retry calls for one panel."""
    if panel.runner == "continuation":
        return panel.position_count
    return panel.position_count


def estimate_panel_cost(
    player_id: str,
    player_config: dict[str, Any],
    panel: AcquisitionPanel,
    *,
    reasoning_ids: set[str],
    cost_calculator: CostCalculator,
) -> float:
    """Mirror the scheduler's base-call estimate for one panel."""
    estimate = cost_calculator.estimate_position_benchmark_cost(
        player_id,
        model_name=player_config.get("model_name"),
        num_positions=panel.planned_first_attempt_calls,
        reasoning=player_id in reasoning_ids,
    )
    if estimate is not None:
        return estimate
    return UNKNOWN_FULL_CORE_COST * panel.planned_first_attempt_calls / 50


def build_acquisition_plan(
    configs: dict[str, dict[str, Any]],
    *,
    policy: AcquisitionPolicy,
    acquisition_state: dict[str, set[str]],
    rating_store: RatingStore,
    freeze_checker: FreezeChecker,
    cost_calculator: CostCalculator,
    reasoning_ids: set[str],
    max_cost: float,
) -> list[AcquisitionPlanRow]:
    """Build the scheduler-equivalent plan without model calls or state changes."""
    player_stats = freeze_checker.stats_collector.get_player_stats()
    candidates: list[dict[str, Any]] = []
    completed_rows: list[AcquisitionPlanRow] = []
    frozen_rows: list[AcquisitionPlanRow] = []

    for player_id, player_config in configs.items():
        rating = rating_store.get(player_id)
        missing = missing_panels(player_id, policy, acquisition_state)
        player_cost = freeze_checker.get_player_cost(player_id)
        priority = rating.rating_deviation / (
            1 + PRIORITY_COST_SENSITIVITY * player_cost
        )
        measured = acquisition_state.get(player_id, set())
        source = getattr(acquisition_state, "deferred_sources", {}).get(player_id)
        missing_names = tuple(panel.name for panel in missing)
        local = getattr(acquisition_state, "local_ready_panels", {}).get(player_id, set())
        billable = [panel for panel in missing if panel.name not in local]
        common = {
            "player_id": player_id,
            "rating": float(rating.rating),
            "rating_deviation": float(rating.rating_deviation),
            "priority": float(priority),
            "measured_panels": tuple(panel.name for panel in policy.panels if panel.name in measured),
            "source_player_id": source,
            "policy_deferred_panels": tuple(
                panel.name for panel in policy.panels
                if source and panel.name not in measured and panel.name not in missing_names
            ),
            "conditional_panels": tuple(
                panel.name for panel in billable
                if not measured and not (local - {"core"}) and panel.name != "core"
            ),
            "missing_panels": missing_names,
        }
        if not missing:
            completed_rows.append(
                AcquisitionPlanRow(
                    **common,
                    status="policy-deferred" if source else "complete",
                    publish_saved_panels=(),
                    scheduled_panels=(),
                    deferred_panels=(),
                    planned_first_attempt_calls=0,
                    maximum_retry_calls=0,
                    estimated_base_cost=0.0,
                )
            )
            continue
        if freeze_checker.is_frozen(
            player_id,
            rating.rating_deviation,
            player_stats,
        ):
            frozen_rows.append(
                AcquisitionPlanRow(
                    **common,
                    status="frozen",
                    publish_saved_panels=(),
                    scheduled_panels=(),
                    deferred_panels=tuple(panel.name for panel in missing),
                    planned_first_attempt_calls=0,
                    maximum_retry_calls=0,
                    estimated_base_cost=0.0,
                )
            )
            continue
        common["publish_saved_panels"] = tuple(panel.name for panel in missing if panel.name in local)
        candidates.append(
            {
                "common": common,
                "config": player_config,
                "missing": billable,
            }
        )

    candidates.sort(key=lambda row: row["common"]["priority"], reverse=True)
    projected_cost = 0.0
    queued_rows: list[AcquisitionPlanRow] = []
    for candidate in candidates:
        common = candidate["common"]
        scheduled: list[AcquisitionPanel] = []
        deferred: list[AcquisitionPanel] = []
        row_cost = 0.0
        for index, panel in enumerate(candidate["missing"]):
            panel_cost = estimate_panel_cost(
                common["player_id"],
                candidate["config"],
                panel,
                reasoning_ids=reasoning_ids,
                cost_calculator=cost_calculator,
            )
            if projected_cost + panel_cost >= max_cost:
                deferred = candidate["missing"][index:]
                break
            scheduled.append(panel)
            row_cost += panel_cost
            projected_cost += panel_cost

        status = (
            "queued-partial"
            if scheduled and deferred
            else "queued"
            if scheduled
            else "budget-deferred" if deferred
            else "publish-only"
        )
        queued_rows.append(
            AcquisitionPlanRow(
                **common,
                status=status,
                scheduled_panels=tuple(panel.name for panel in scheduled),
                deferred_panels=tuple(panel.name for panel in deferred),
                planned_first_attempt_calls=sum(
                    panel.planned_first_attempt_calls for panel in scheduled
                ),
                maximum_retry_calls=sum(
                    panel_retry_upper_bound(panel) for panel in scheduled
                ),
                estimated_base_cost=row_cost,
            )
        )

    return queued_rows + frozen_rows + completed_rows


def plan_summary(rows: list[AcquisitionPlanRow], policy: AcquisitionPolicy) -> dict[str, Any]:
    """Return machine-readable aggregate and row details."""
    scheduled = [row for row in rows if row.scheduled_panels]
    return {
        "policy_version": policy.version,
        "zero_model_calls": True,
        "configured_models": len(rows),
        "complete_models": sum(row.status == "complete" for row in rows),
        "policy_deferred_models": sum(row.status == "policy-deferred" for row in rows),
        "frozen_models": sum(row.status == "frozen" for row in rows),
        "queued_models": len(scheduled),
        "publish_only_models": sum(row.status == "publish-only" for row in rows),
        "budget_deferred_models": sum(
            row.status in {"queued-partial", "budget-deferred"} for row in rows
        ),
        "fully_budget_deferred_models": sum(
            row.status == "budget-deferred" for row in rows
        ),
        "planned_first_attempt_calls": sum(
            row.planned_first_attempt_calls for row in scheduled
        ),
        "maximum_retry_calls": sum(row.maximum_retry_calls for row in scheduled),
        "estimated_base_cost": sum(row.estimated_base_cost for row in scheduled),
        "rows": [asdict(row) for row in rows],
    }


def print_plan(summary: dict[str, Any]) -> None:
    """Print a concise acquisition preflight."""
    print("AUTOMATIC BENCHMARK ACQUISITION PREFLIGHT")
    print(f"Policy: {summary['policy_version']}")
    print("Model/API calls made: 0")
    print(
        "Models: "
        f"{summary['queued_models']} queued, "
        f"{summary['complete_models']} measured complete, "
        f"{summary['policy_deferred_models']} policy-deferred, "
        f"{summary['frozen_models']} frozen, "
        f"{summary['budget_deferred_models']} budget-deferred, "
        f"{summary['publish_only_models']} publish-only"
    )
    print(
        f"Scheduled workload: {summary['planned_first_attempt_calls']} base calls + "
        f"up to {summary['maximum_retry_calls']} retries, "
        f"estimated base cost ${summary['estimated_base_cost']:.4f}"
    )
    if any(row["conditional_panels"] for row in summary["rows"]):
        print("Estimates include potential supplemental work pending core results.")
    print()
    for row in summary["rows"]:
        if not row["missing_panels"] and not row["policy_deferred_panels"]:
            continue
        scheduled = ",".join(row["scheduled_panels"]) or "-"
        deferred = ",".join(row["deferred_panels"]) or "-"
        policy_note = (
            f"  policy-defer={','.join(row['policy_deferred_panels'])}"
            f" (source: {row['source_player_id']})" if row["policy_deferred_panels"] else ""
        )
        if row["publish_saved_panels"]:
            policy_note += f"  publish-saved={','.join(row['publish_saved_panels'])}"
        if row["conditional_panels"]:
            policy_note += f"  pending-core={','.join(row['conditional_panels'])}"
        print(
            f"{row['status']:>15}  {row['player_id']}  "
            f"RD={row['rating_deviation']:.0f}  "
            f"run={scheduled}  defer={deferred}  "
            f"calls={row['planned_first_attempt_calls']}  "
            f"cost=${row['estimated_base_cost']:.4f}{policy_note}"
        )


def run_preflight(
    *,
    config_path: Path,
    api_backend: str,
    max_cost: float | None,
    json_output: Path | None = None,
) -> dict[str, Any]:
    """Load production state and print a zero-call acquisition plan."""
    print("Loading ratings and game history for freeze-aware preflight...", flush=True)
    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}
    configs = selected_llm_configs(config, api_backend)
    reasoning_ids = reasoning_player_ids(configs)
    engine_ids = {engine["player_id"] for engine in config.get("engines", [])}
    anchor_ids = {
        engine["player_id"]
        for engine in config.get("engines", [])
        if engine.get("anchor", True)
    }
    ghost_ids = {
        engine["player_id"]
        for engine in config.get("engines", [])
        if engine.get("ghost", False)
    }

    rating_store = RatingStore(
        path="data/ratings.json",
        anchor_ids=anchor_ids,
        ghost_ids=ghost_ids,
    )
    pgn_logger = PGNLogger()
    stats_collector = StatsCollector()
    stats_collector.add_results(pgn_logger.load_all_results())
    freeze_checker = FreezeChecker(
        rating_store,
        stats_collector,
        reasoning_ids,
        engine_ids,
    )
    policy = load_acquisition_policy()
    state = load_acquisition_state(policy, use_firestore=rating_store._use_firestore)
    cost_calculator = CostCalculator(config_path=config_path)
    effective_max_cost = (
        float(max_cost)
        if max_cost is not None
        else float(config.get("benchmark", {}).get("max_cost", math.inf))
    )
    rows = build_acquisition_plan(
        configs,
        policy=policy,
        acquisition_state=state,
        rating_store=rating_store,
        freeze_checker=freeze_checker,
        cost_calculator=cost_calculator,
        reasoning_ids=reasoning_ids,
        max_cost=effective_max_cost,
    )
    summary = plan_summary(rows, policy)
    summary["api_backend"] = api_backend
    summary["max_cost"] = effective_max_cost
    print_plan(summary)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nSaved plan to {json_output}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, default=Path("config/benchmark.yaml"))
    parser.add_argument("--api", choices=["openrouter", "gemini", "codex"], default="openrouter")
    parser.add_argument("--max-cost", type=float)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    run_preflight(
        config_path=args.config,
        api_backend=args.api,
        max_cost=args.max_cost,
        json_output=args.json_output,
    )


if __name__ == "__main__":
    main()
