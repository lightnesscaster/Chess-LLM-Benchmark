#!/usr/bin/env python3
"""Evaluate locked stability-cap shadow predictions against game-only ratings."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from position_benchmark.stability_cap_shadow import (  # noqa: E402
    POLICY_PATH,
    SHADOW_LEDGER_PATH,
)


def load(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[max(0, min(len(ordered) - 1, index))]


def candidate_metrics(
    rows: list[dict[str, Any]],
    candidate: str,
) -> dict[str, float]:
    errors = [
        row["predictions"][candidate] - row["target"]["rating"]
        for row in rows
    ]
    return {
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(
            sum(error * error for error in errors) / len(errors)
        ),
        "bias": sum(errors) / len(errors),
        "max_absolute_error": max(abs(error) for error in errors),
    }


def group_bootstrap(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    reference: str,
    challenger: str,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    names = sorted(groups)
    rng = random.Random(seed + sum(ord(char) for char in group_key))
    deltas = []
    for _ in range(resamples):
        sampled = [rng.choice(names) for _ in names]
        selected = [row for name in sampled for row in groups[name]]
        reference_mae = candidate_metrics(selected, reference)["mae"]
        challenger_mae = candidate_metrics(selected, challenger)["mae"]
        deltas.append(challenger_mae - reference_mae)
    return {
        "probability_mae_improves": (
            sum(delta < 0.0 for delta in deltas) / len(deltas)
        ),
        "mae_delta_p05": percentile(deltas, 0.05),
        "mae_delta_p50": percentile(deltas, 0.50),
        "mae_delta_p95": percentile(deltas, 0.95),
    }


def leave_one_group_out(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    reference: str,
    challenger: str,
) -> dict[str, float]:
    results = {}
    for group in sorted({row[group_key] for row in rows}):
        remaining = [row for row in rows if row[group_key] != group]
        if not remaining:
            continue
        results[group] = (
            candidate_metrics(remaining, challenger)["mae"]
            - candidate_metrics(remaining, reference)["mae"]
        )
    return results


def build_rows(
    ledger: dict[str, Any],
    ratings: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_policy = policy["primary_target"]
    reference = policy["comparison"]["reference_candidate"]
    challenger = policy["comparison"]["challenger_candidate"]
    eligible = []
    mature = []
    for player_id, record in ledger.get("entries", {}).items():
        if not record.get("eligibility", {}).get("prospective_holdout"):
            continue
        target = ratings.get(player_id)
        row = {
            "player_id": player_id,
            "family": record["family"],
            "lab": record["lab"],
            "recorded_at": record["recorded_at"],
            "predictions": {
                reference: float(record["candidates"][reference]),
                challenger: float(record["candidates"][challenger]),
            },
            "affected": not math.isclose(
                float(record["candidates"][reference]),
                float(record["candidates"][challenger]),
                abs_tol=1e-9,
            ),
            "target": None,
            "mature": False,
            "maturity_reason": "missing game-only rating",
        }
        if isinstance(target, dict):
            games = int(target.get("games_played", 0) or 0)
            games_rd = float(
                target.get(
                    "games_rd",
                    target.get("rating_deviation", 350.0),
                )
            )
            row["target"] = {
                "rating": float(target["rating"]),
                "games": games,
                "games_rd": games_rd,
            }
            if games < int(target_policy["minimum_games"]):
                row["maturity_reason"] = "too few games"
            elif games_rd > float(target_policy["maximum_games_rd"]):
                row["maturity_reason"] = "games RD too high"
            else:
                row["mature"] = True
                row["maturity_reason"] = "ready"
                mature.append(row)
        eligible.append(row)
    return eligible, mature


def evaluate(
    *,
    policy_path: Path,
    ledger_path: Path,
    ratings_path: Path,
) -> dict[str, Any]:
    policy = load(policy_path)
    ledger = load(ledger_path)
    ratings = load(ratings_path)
    eligible, mature = build_rows(ledger, ratings, policy)
    comparison = policy["comparison"]
    reference = comparison["reference_candidate"]
    challenger = comparison["challenger_candidate"]
    evaluated = (
        [row for row in mature if row["affected"]]
        if comparison["evaluate_affected_configurations_only"]
        else mature
    )
    affected_families = sorted({row["family"] for row in evaluated})
    affected_labs = sorted({row["lab"] for row in evaluated})
    coverage_policy = policy["coverage_gate"]
    coverage_checks = {
        "mature_holdout_configurations": (
            len(mature)
            >= int(coverage_policy["minimum_mature_holdout_configurations"])
        ),
        "affected_holdout_configurations": (
            len(evaluated)
            >= int(coverage_policy["minimum_affected_holdout_configurations"])
        ),
        "affected_families": (
            len(affected_families)
            >= int(coverage_policy["minimum_affected_families"])
        ),
        "affected_labs": (
            len(affected_labs)
            >= int(coverage_policy["minimum_affected_labs"])
        ),
    }
    coverage_passed = all(coverage_checks.values())

    comparison_result = None
    promotion_checks: dict[str, bool] = {}
    if evaluated:
        reference_metrics = candidate_metrics(evaluated, reference)
        challenger_metrics = candidate_metrics(evaluated, challenger)
        bootstrap_policy = policy["bootstrap"]
        family_bootstrap = group_bootstrap(
            evaluated,
            group_key="family",
            reference=reference,
            challenger=challenger,
            resamples=int(bootstrap_policy["resamples"]),
            seed=int(bootstrap_policy["random_seed"]),
        )
        lab_bootstrap = group_bootstrap(
            evaluated,
            group_key="lab",
            reference=reference,
            challenger=challenger,
            resamples=int(bootstrap_policy["resamples"]),
            seed=int(bootstrap_policy["random_seed"]),
        )
        leave_one_lab = leave_one_group_out(
            evaluated,
            group_key="lab",
            reference=reference,
            challenger=challenger,
        )
        mae_delta = challenger_metrics["mae"] - reference_metrics["mae"]
        rmse_delta = challenger_metrics["rmse"] - reference_metrics["rmse"]
        absolute_bias_increase = (
            abs(challenger_metrics["bias"]) - abs(reference_metrics["bias"])
        )
        maximum_leave_one_lab_delta = (
            max(leave_one_lab.values()) if leave_one_lab else None
        )
        gate = policy["promotion_gate"]
        promotion_checks = {
            "family_bootstrap": (
                family_bootstrap["probability_mae_improves"]
                >= float(
                    gate[
                        "minimum_family_bootstrap_probability_mae_improves"
                    ]
                )
            ),
            "lab_bootstrap": (
                lab_bootstrap["probability_mae_improves"]
                >= float(
                    gate["minimum_lab_bootstrap_probability_mae_improves"]
                )
            ),
            "mae_improvement": (
                mae_delta
                <= -float(gate["minimum_mae_improvement_elo"])
            ),
            "rmse": (
                rmse_delta <= float(gate["maximum_rmse_delta_elo"])
            ),
            "bias": (
                absolute_bias_increase
                <= float(gate["maximum_absolute_bias_increase_elo"])
            ),
            "leave_one_lab_out": (
                maximum_leave_one_lab_delta is not None
                and maximum_leave_one_lab_delta
                <= float(
                    gate[
                        "maximum_leave_one_lab_out_mae_delta_elo"
                    ]
                )
            ),
        }
        comparison_result = {
            "reference": {
                "candidate": reference,
                **reference_metrics,
            },
            "challenger": {
                "candidate": challenger,
                **challenger_metrics,
            },
            "mae_delta": mae_delta,
            "rmse_delta": rmse_delta,
            "absolute_bias_increase": absolute_bias_increase,
            "family_bootstrap": family_bootstrap,
            "lab_bootstrap": lab_bootstrap,
            "leave_one_lab_out_mae_delta": leave_one_lab,
            "maximum_leave_one_lab_out_mae_delta": (
                maximum_leave_one_lab_delta
            ),
        }

    promotion_passed = (
        coverage_passed
        and bool(promotion_checks)
        and all(promotion_checks.values())
    )
    if not coverage_passed:
        status = "collecting"
    elif promotion_passed:
        status = "promotion-candidate"
    else:
        status = "hold-failed-promotion-gate"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy_version": policy["policy_version"],
        "production_effect": "none",
        "status": status,
        "inputs": {
            "policy": {
                "path": display_path(policy_path),
                "sha256": sha256(policy_path),
            },
            "ledger": {
                "path": display_path(ledger_path),
                "sha256": sha256(ledger_path),
            },
            "ratings": {
                "path": display_path(ratings_path),
                "sha256": sha256(ratings_path),
            },
        },
        "coverage": {
            "prospective_holdout_configurations": len(eligible),
            "mature_holdout_configurations": len(mature),
            "affected_mature_configurations": len(evaluated),
            "affected_families": affected_families,
            "affected_labs": affected_labs,
            "checks": coverage_checks,
            "passed": coverage_passed,
        },
        "comparison": comparison_result,
        "promotion_checks": promotion_checks,
        "promotion_passed": promotion_passed,
        "rows": eligible,
    }


def render_report(
    analysis: dict[str, Any],
    policy: dict[str, Any],
) -> str:
    coverage = analysis["coverage"]
    comparison = analysis["comparison"]
    if comparison is None:
        result = (
            "No mature affected holdout configuration is available yet. "
            "Production remains unchanged."
        )
    else:
        result = (
            f"The challenger-minus-production MAE delta is "
            f"{comparison['mae_delta']:+.1f} Elo across "
            f"{coverage['affected_mature_configurations']} affected mature "
            f"configurations. Family-bootstrap P(improves) is "
            f"{comparison['family_bootstrap']['probability_mae_improves']:.3f}; "
            f"lab-bootstrap P(improves) is "
            f"{comparison['lab_bootstrap']['probability_mae_improves']:.3f}."
        )
    check_lines = "\n".join(
        f"- `{name}`: {'pass' if passed else 'pending/fail'}"
        for name, passed in {
            **coverage["checks"],
            **analysis["promotion_checks"],
        }.items()
    )
    target = policy["primary_target"]
    return f"""# Prospective depth-30 stability-cap holdout

Status: **{analysis['status']}**. Production effect: **none**.

{result}

## Coverage

- Prospective configurations: {coverage['prospective_holdout_configurations']}
- Mature configurations: {coverage['mature_holdout_configurations']}
- Affected mature configurations: {coverage['affected_mature_configurations']}
- Affected families: {len(coverage['affected_families'])}
- Affected labs: {len(coverage['affected_labs'])}

## Fixed checks

{check_lines}

The primary target is `{target['name']}` with at least
{target['minimum_games']} games and games RD at most
{target['maximum_games_rd']}. Predictions are locked before the first game and
cannot alter production ratings. Promotion requires every fixed coverage and
performance check in the policy to pass.
"""


def write_awaiting_target(
    *,
    policy: dict[str, Any],
    policy_path: Path,
    ledger_path: Path,
    ratings_path: Path,
    output_path: Path,
    report_path: Path,
) -> None:
    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy_version": policy["policy_version"],
        "production_effect": "none",
        "status": "awaiting-game-only-ratings",
        "missing_ratings_path": str(ratings_path),
        "refresh_command": policy["primary_target"]["refresh_command"],
        "inputs": {
            "policy": str(policy_path),
            "ledger": str(ledger_path),
        },
        "coverage": {
            "prospective_holdout_configurations": 0,
            "mature_holdout_configurations": 0,
            "affected_mature_configurations": 0,
            "affected_families": [],
            "affected_labs": [],
            "checks": {},
            "passed": False,
        },
        "comparison": None,
        "promotion_checks": {},
        "promotion_passed": False,
        "rows": [],
    }
    output_path.write_text(json.dumps(analysis, indent=2) + "\n")
    report_path.write_text(
        "# Prospective depth-30 stability-cap holdout\n\n"
        "Status: **awaiting game-only ratings**. Production effect: **none**.\n\n"
        f"Refresh the isolated target with:\n\n`{policy['primary_target']['refresh_command']}`\n"
    )


def evaluate_and_write(
    *,
    policy_path: Path = POLICY_PATH,
    ledger_path: Path = SHADOW_LEDGER_PATH,
    ratings_path: Path | None = None,
    output_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the current ledger and write its research-only decision artifacts."""
    policy = load(policy_path)
    ratings_path = ratings_path or ROOT / policy["primary_target"]["default_path"]
    output_path = output_path or ROOT / policy["evaluation_output"]
    report_path = report_path or ROOT / policy["evaluation_report"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not ratings_path.exists():
        write_awaiting_target(
            policy=policy,
            policy_path=policy_path,
            ledger_path=ledger_path,
            ratings_path=ratings_path,
            output_path=output_path,
            report_path=report_path,
        )
        return load(output_path)

    analysis = evaluate(
        policy_path=policy_path,
        ledger_path=ledger_path,
        ratings_path=ratings_path,
    )
    output_path.write_text(json.dumps(analysis, indent=2) + "\n")
    report_path.write_text(render_report(analysis, policy))
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--ledger", type=Path, default=SHADOW_LEDGER_PATH)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    policy = load(args.policy)
    ratings_path = args.ratings or ROOT / policy["primary_target"]["default_path"]
    analysis = evaluate_and_write(
        policy_path=args.policy,
        ledger_path=args.ledger,
        ratings_path=ratings_path,
        output_path=args.output,
        report_path=args.report,
    )
    if analysis["status"] == "awaiting-game-only-ratings":
        print(f"Awaiting game-only ratings: {ratings_path}")
        return

    output_path = args.output or ROOT / policy["evaluation_output"]
    report_path = args.report or ROOT / policy["evaluation_report"]
    print(
        f"Saved {output_path} ({analysis['status']}; "
        f"production effect: none)"
    )
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
