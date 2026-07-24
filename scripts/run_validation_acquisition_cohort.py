#!/usr/bin/env python3
"""Acquire standardized benchmark panels for a fixed validation cohort."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import chess.engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from position_benchmark.acquisition import (  # noqa: E402
    _record_readiness,
    load_acquisition_policy,
    load_acquisition_state,
    missing_panels,
)
from position_benchmark.run_benchmark import (  # noqa: E402
    config_uses_reasoning,
    load_player_configs,
    run_benchmark_for_scheduler,
)
from rating.cost_calculator import CostCalculator  # noqa: E402
from scripts.run_stability_probe import (  # noqa: E402
    _merge_json_record,
    run_probe_for_scheduler,
    save_player_record,
)


DEFAULT_WORK_DIR = ROOT / "data" / "acquisition_cohort"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def estimated_panel_cost(
    calculator: CostCalculator,
    player_id: str,
    config: dict[str, Any],
    planned_calls: int,
) -> float:
    estimate = calculator.estimate_position_benchmark_cost(
        player_id,
        model_name=config.get("model_name"),
        num_positions=planned_calls,
        reasoning=config_uses_reasoning(config),
        use_budget_overrides=False,
    )
    if estimate is None:
        raise ValueError(f"{player_id}: no token-priced estimate is available")
    return float(estimate)


def actual_panel_cost(
    calculator: CostCalculator,
    config: dict[str, Any],
    token_usage: dict[str, Any],
) -> float:
    model_name = str(config.get("model_name") or "")
    cost = calculator.calculate_game_cost(
        {
            "prompt_tokens": int(token_usage.get("prompt", 0) or 0),
            "completion_tokens": int(token_usage.get("completion", 0) or 0),
        },
        model_name,
    )
    if cost is None:
        raise ValueError(f"{model_name}: no token price is available")
    return float(cost)


def sync_core_record(player_id: str, record: dict[str, Any]) -> None:
    """Best-effort sync after a validated staged core record is promoted."""
    try:
        from firebase_client import (
            BENCHMARK_RESULTS_COLLECTION,
            get_firestore_client,
        )

        db = get_firestore_client()
        db.collection(BENCHMARK_RESULTS_COLLECTION).document(player_id).set(record)
    except Exception as exc:
        print(f"    Warning: failed to sync core result to Firestore: {exc}")


async def run(args: argparse.Namespace) -> int:
    policy = load_acquisition_policy()
    state = load_acquisition_state(policy)
    configs = load_player_configs()
    calculator = CostCalculator()
    players = list(dict.fromkeys(args.players))
    unknown = [player_id for player_id in players if player_id not in configs]
    if unknown:
        raise SystemExit("Missing player config(s): " + ", ".join(unknown))

    work_dir = args.work_dir.resolve()
    ledger_path = work_dir / "ledger.json"
    shard_dir = work_dir / "shards"
    ledger = load_json(
        ledger_path,
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "max_cost": args.max_cost,
            "players": players,
            "actual_cost": 0.0,
            "panels": [],
        },
    )
    if float(ledger.get("max_cost")) != args.max_cost:
        raise SystemExit(
            f"Existing ledger does not match this cohort/budget: {ledger_path}"
        )
    ledger_players = list(ledger.get("players") or [])
    if ledger_players != players:
        if not args.extend_ledger:
            raise SystemExit(
                f"Existing ledger does not match this cohort: {ledger_path} "
                "(pass --extend-ledger to append replacement players)"
            )
        ledger["players"] = list(
            dict.fromkeys([*ledger_players, *players])
        )

    queued: list[tuple[str, Any, float]] = []
    for player_id in players:
        config = configs[player_id]
        for panel in missing_panels(player_id, policy, state):
            estimate = estimated_panel_cost(
                calculator,
                player_id,
                config,
                panel.planned_first_attempt_calls,
            )
            queued.append((player_id, panel, estimate))

    print(
        f"Validation cohort: {len(players)} players, {len(queued)} missing panels, "
        f"${float(ledger['actual_cost']):.4f} already spent"
    )
    print(
        f"Estimated remaining first-attempt cost: "
        f"${sum(item[2] for item in queued):.4f}"
    )
    print(
        f"Hard ceiling: ${args.max_cost:.2f}; "
        f"pre-panel safety multiplier: {args.safety_multiplier:.2f}x"
    )
    if args.dry_run:
        for player_id, panel, estimate in queued:
            print(f"  {player_id}: {panel.name} ${estimate:.4f}")
        return 0

    stockfish = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    try:
        for player_id in players:
            config = configs[player_id]
            for panel in missing_panels(player_id, policy, state):
                estimate = estimated_panel_cost(
                    calculator,
                    player_id,
                    config,
                    panel.planned_first_attempt_calls,
                )
                spent = float(ledger["actual_cost"])
                guarded_cost = estimate * args.safety_multiplier
                if spent + guarded_cost >= args.max_cost:
                    print(
                        f"STOPPING before {player_id}/{panel.name}: "
                        f"${spent:.4f} spent + ${guarded_cost:.4f} guarded estimate "
                        f">= ${args.max_cost:.2f}"
                    )
                    return 2

                positions_data = load_json(panel.positions_path, {})
                positions = (
                    positions_data.get("positions", [])
                    if isinstance(positions_data, dict)
                    else positions_data
                )
                shard_path = shard_dir / f"{panel.name}.json"
                print(
                    f"\n{player_id}: running {panel.name} "
                    f"(estimate ${estimate:.4f}, spent ${spent:.4f})",
                    flush=True,
                )
                if panel.runner == "positions":
                    result = await run_benchmark_for_scheduler(
                        player_id=player_id,
                        player_config=config,
                        stockfish=stockfish,
                        positions=positions,
                        depth=panel.score_depth,
                        original_indices=None,
                        results_path=shard_path,
                        sync_firestore=False,
                    )
                else:
                    result = await run_probe_for_scheduler(
                        player_id=player_id,
                        player_config=config,
                        stockfish=stockfish,
                        positions=positions,
                        positions_path=panel.positions_path,
                        results_path=shard_path,
                        position_limit=panel.position_count,
                        probe_plies=panel.probe_plies,
                        score_depth=panel.score_depth,
                        random_seed=panel.random_seed,
                    )

                token_usage = result.get(
                    "token_usage", {"prompt": 0, "completion": 0}
                )
                actual = actual_panel_cost(calculator, config, token_usage)
                ledger["actual_cost"] = spent + actual
                ledger["panels"].append(
                    {
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "player_id": player_id,
                        "panel": panel.name,
                        "success": bool(result.get("success")),
                        "estimated_cost": estimate,
                        "actual_cost": actual,
                        "token_usage": token_usage,
                        "model_calls": int(result.get("model_calls", 0) or 0),
                        "error": result.get("error"),
                    }
                )
                save_json(ledger_path, ledger)
                print(
                    f"    Actual panel cost ${actual:.4f}; "
                    f"cumulative ${float(ledger['actual_cost']):.4f}",
                    flush=True,
                )
                if float(ledger["actual_cost"]) >= args.max_cost:
                    print(
                        f"STOPPING: cumulative spend reached "
                        f"${float(ledger['actual_cost']):.4f}"
                    )
                    return 2
                if not result.get("success"):
                    print(f"    FAILED: {result.get('error', 'unknown error')}")
                    break

                shard = load_json(shard_path, {})
                record = shard.get(player_id)
                if not isinstance(record, dict):
                    print("    FAILED: staged result record is missing")
                    break
                readiness = _record_readiness(panel, record, positions_data)
                if not readiness.is_ready:
                    print(f"    FAILED readiness: {readiness.reason}")
                    break

                if panel.runner == "continuation":
                    save_player_record(panel.results_path, player_id, record)
                else:
                    _merge_json_record(panel.results_path, player_id, record)
                if panel.name == "core":
                    sync_core_record(player_id, record)
                state.setdefault(player_id, set()).add(panel.name)
                print(f"    Promoted validated {panel.name} record")
    finally:
        stockfish.quit()

    print(
        f"\nCohort acquisition complete: ${float(ledger['actual_cost']):.4f} spent"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", nargs="+", required=True)
    parser.add_argument("--max-cost", type=float, required=True)
    parser.add_argument(
        "--safety-multiplier",
        type=float,
        default=1.5,
        help="Guarded estimate checked before starting each panel",
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument(
        "--extend-ledger",
        action="store_true",
        help="Append --players to an existing cumulative-budget ledger",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
