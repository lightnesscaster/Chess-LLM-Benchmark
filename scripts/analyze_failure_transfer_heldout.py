#!/usr/bin/env python3
"""Validate and analyze the frozen cross-family failure-transfer experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = (
    ROOT
    / "position_benchmark/validation/2026-07-17-failure-transfer-heldout-plan.json"
)


def load(path: Path) -> Any:
    """Load a JSON artifact."""
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, trials: int) -> list[float] | None:
    """Return a percentage-scale 95% Wilson interval."""
    if not trials:
        return None
    z = 1.96
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return [100 * max(0.0, center - half), 100 * min(1.0, center + half)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize legality, retry, and CPL evidence for result rows."""
    cpls = [float(row["cpl"]) for row in rows]
    legal_cpls = [float(row["cpl"]) for row in rows if row["is_legal"] is True]
    illegals = sum(row["is_legal"] is False for row in rows)
    retry_rows = [row for row in rows if row.get("retry_attempted") is True]
    return {
        "attempts": len(rows),
        "illegals": illegals,
        "illegal_pct": 100 * illegals / len(rows) if rows else None,
        "illegal_wilson_95pct": wilson(illegals, len(rows)),
        "mean_cpl": statistics.mean(cpls) if cpls else None,
        "median_cpl": statistics.median(cpls) if cpls else None,
        "legal_only_mean_cpl": statistics.mean(legal_cpls) if legal_cpls else None,
        "retry_attempts": len(retry_rows),
        "retry_recoveries": sum(
            row.get("retry_is_legal") is True for row in retry_rows
        ),
        "retry_failures": sum(
            row.get("retry_is_legal") is False for row in retry_rows
        ),
    }


def one_sided_exact_mcnemar(failure_only: int, control_only: int) -> float:
    """Test whether failure-only discordance exceeds control-only discordance."""
    discordant = failure_only + control_only
    if not discordant:
        return 1.0
    return sum(
        math.comb(discordant, count) * 0.5**discordant
        for count in range(failure_only, discordant + 1)
    )


def analyze(plan: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """Validate results against the frozen plan and calculate its endpoints."""
    target_family = {
        target["player_id"]: target["family"] for target in plan["targets"]
    }
    expected_players = list(target_family)
    if set(results) != set(expected_players):
        missing = sorted(set(expected_players) - set(results))
        extra = sorted(set(results) - set(expected_players))
        raise ValueError(f"Target mismatch: missing={missing}, extra={extra}")

    pairs = plan["position_pairs"]
    expected_ids = [
        position_id
        for pair in pairs
        for position_id in (pair["failure_id"], pair["control_id"])
    ]
    rows: list[dict[str, Any]] = []
    pair_outcomes: list[dict[str, Any]] = []
    per_target: dict[str, Any] = {}
    for player_id in expected_players:
        record = results[player_id]
        player_rows = record.get("results", [])
        by_id = {row.get("position_id"): row for row in player_rows}
        if set(by_id) != set(expected_ids) or len(player_rows) != len(expected_ids):
            raise ValueError(f"{player_id}: rows do not match frozen position IDs")
        if int(record.get("summary", {}).get("positions_skipped", 0)) != 0:
            raise ValueError(f"{player_id}: skipped or errored positions remain")
        if any(int(row.get("stockfish_depth", 0)) != plan["stockfish_depth"] for row in player_rows):
            raise ValueError(f"{player_id}: result is not uniformly depth 30")
        if any(row.get("panel") != "failure-transfer-positive-v1" for row in player_rows):
            raise ValueError(f"{player_id}: unexpected panel marker")

        annotated = [
            {
                **row,
                "target_player_id": player_id,
                "target_family": target_family[player_id],
                "candidate_kind": (
                    "source_failure"
                    if row["position_id"].startswith("failure-")
                    else "matched_control"
                ),
            }
            for row in player_rows
        ]
        rows.extend(annotated)
        per_target[player_id] = {
            "family": target_family[player_id],
            "failure_states": summarize(
                [row for row in annotated if row["candidate_kind"] == "source_failure"]
            ),
            "matched_controls": summarize(
                [row for row in annotated if row["candidate_kind"] == "matched_control"]
            ),
        }
        for pair in pairs:
            failure = by_id[pair["failure_id"]]
            control = by_id[pair["control_id"]]
            pair_outcomes.append(
                {
                    "target_player_id": player_id,
                    "target_family": target_family[player_id],
                    "failure_id": pair["failure_id"],
                    "control_id": pair["control_id"],
                    "failure_illegal": failure["is_legal"] is False,
                    "control_illegal": control["is_legal"] is False,
                }
            )

    failure_rows = [row for row in rows if row["candidate_kind"] == "source_failure"]
    control_rows = [row for row in rows if row["candidate_kind"] == "matched_control"]
    failure_only = sum(
        pair["failure_illegal"] and not pair["control_illegal"]
        for pair in pair_outcomes
    )
    control_only = sum(
        pair["control_illegal"] and not pair["failure_illegal"]
        for pair in pair_outcomes
    )
    p_value = one_sided_exact_mcnemar(failure_only, control_only)

    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_rows[row["target_family"]].append(row)
    per_family = {
        family: {
            "failure_states": summarize(
                [row for row in family_result_rows if row["candidate_kind"] == "source_failure"]
            ),
            "matched_controls": summarize(
                [row for row in family_result_rows if row["candidate_kind"] == "matched_control"]
            ),
        }
        for family, family_result_rows in sorted(family_rows.items())
    }
    positive_families = [
        family
        for family, summary in per_family.items()
        if summary["failure_states"]["illegals"]
        > summary["matched_controls"]["illegals"]
    ]

    per_candidate = {}
    for pair in pairs:
        failure_id = pair["failure_id"]
        control_id = pair["control_id"]
        per_candidate[failure_id] = {
            "failure_states": summarize(
                [row for row in failure_rows if row["position_id"] == failure_id]
            ),
            "matched_controls": summarize(
                [row for row in control_rows if row["position_id"] == control_id]
            ),
        }

    failure_summary = summarize(failure_rows)
    control_summary = summarize(control_rows)
    gate = plan["promotion_gate"]
    passed = (
        failure_summary["illegals"] > control_summary["illegals"]
        and p_value <= float(gate["one_sided_exact_mcnemar_p_max"])
        and len(positive_families) >= int(gate["minimum_positive_distinct_families"])
    )
    actual_cost = sum(
        float(record.get("summary", {}).get("run_actual_api_cost") or 0)
        for record in results.values()
    )
    model_calls = sum(
        int(record.get("summary", {}).get("run_model_calls") or 0)
        for record in results.values()
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": plan["experiment_id"],
        "frozen_at": plan["frozen_at"],
        "production_effect": "none",
        "stockfish_depth": plan["stockfish_depth"],
        "primary": {
            "failure_states": failure_summary,
            "matched_controls": control_summary,
            "illegal_rate_difference_points": (
                failure_summary["illegal_pct"] - control_summary["illegal_pct"]
            ),
            "matched_pairs": len(pair_outcomes),
            "failure_only_discordant": failure_only,
            "control_only_discordant": control_only,
            "one_sided_exact_mcnemar_p": p_value,
            "positive_families": positive_families,
            "positive_family_count": len(positive_families),
            "promotion_gate_passed": passed,
            "decision": (
                "promote-to-broader-calibration-candidate"
                if passed
                else "retain-research-only"
            ),
        },
        "per_family": per_family,
        "per_target": per_target,
        "per_candidate": per_candidate,
        "pair_outcomes": pair_outcomes,
        "cost": {
            "estimated_base_usd": plan["estimated_base_cost_usd"],
            "estimated_retry_upper_bound_usd": plan[
                "estimated_worst_case_cost_usd"
            ],
            "actual_reported_usd": actual_cost,
            "model_calls": model_calls,
            "estimator_note": (
                "The CLI cap is a pre-call estimate guard, not a runtime spend stop."
            ),
        },
        "interpretation_limits": plan["interpretation_limits"],
    }


def render_report(analysis: dict[str, Any]) -> str:
    """Render the validated analysis as a concise Markdown report."""
    primary = analysis["primary"]
    failure = primary["failure_states"]
    control = primary["matched_controls"]
    decision = (
        "PASS — promote only to a broader calibration candidate"
        if primary["promotion_gate_passed"]
        else "FAIL — retain as research-only"
    )
    family_lines = []
    for family, summary in analysis["per_family"].items():
        family_lines.append(
            f"| {family} | {summary['failure_states']['illegals']}/"
            f"{summary['failure_states']['attempts']} | "
            f"{summary['matched_controls']['illegals']}/"
            f"{summary['matched_controls']['attempts']} |"
        )
    target_lines = []
    for target, summary in analysis["per_target"].items():
        target_lines.append(
            f"| {target} | {summary['failure_states']['illegals']}/3 | "
            f"{summary['matched_controls']['illegals']}/3 |"
        )
    candidate_lines = []
    for candidate, summary in analysis["per_candidate"].items():
        candidate_lines.append(
            f"| {candidate} | {summary['failure_states']['illegals']}/8 | "
            f"{summary['matched_controls']['illegals']}/8 |"
        )
    retry_recoveries = failure["retry_recoveries"] + control["retry_recoveries"]
    retry_attempts = failure["retry_attempts"] + control["retry_attempts"]
    return f"""# Cross-family failure-transfer held-out result — 2026-07-17

The frozen primary decision is **{decision}**.

## Primary result

| Endpoint | Failure states | Matched controls |
| --- | ---: | ---: |
| Illegal first answers | {failure['illegals']}/{failure['attempts']} ({failure['illegal_pct']:.1f}%) | {control['illegals']}/{control['attempts']} ({control['illegal_pct']:.1f}%) |
| Mean depth-30 CPL | {failure['mean_cpl']:.1f} | {control['mean_cpl']:.1f} |
| Legal-only mean depth-30 CPL | {failure['legal_only_mean_cpl']:.1f} | {control['legal_only_mean_cpl']:.1f} |

There were {primary['failure_only_discordant']} failure-only and
{primary['control_only_discordant']} control-only discordant pairs. The frozen
one-sided exact McNemar probability is {primary['one_sided_exact_mcnemar_p']:.6g}.
The direction was positive in {primary['positive_family_count']} families:
{', '.join(primary['positive_families']) or 'none'}.

## By family

| Family | Failure illegals | Control illegals |
| --- | ---: | ---: |
{chr(10).join(family_lines)}

## By configuration

| Configuration | Failure illegals | Control illegals |
| --- | ---: | ---: |
{chr(10).join(target_lines)}

## By frozen candidate pair

| Failure candidate | Failure illegals | Control illegals |
| --- | ---: | ---: |
{chr(10).join(candidate_lines)}

All illegal first answers triggered the production conditional-retry protocol.
Recovery was {retry_recoveries}/{retry_attempts}.

The run made {analysis['cost']['model_calls']} model calls and reports an actual
OpenRouter cost of ${analysis['cost']['actual_reported_usd']:.4f}. The $0.3252
preflight bound was an estimated-cost admission guard, not a runtime spend stop.

## Interpretation

This validates or rejects only the frozen three-position shortlist. Because the
shortlist was selected using GPT-5.6 outcomes, it does not estimate the prevalence
of transferable failure positions in general. The observed lift was concentrated
in `failure-transfer-luna-003` (4/8 versus 1/8); the two Sol-sourced pairs were
flat or reversed. Passing would permit a broader, pre-frozen calibration
experiment; this result did not pass and has no production rating effect.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--analysis-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()

    plan = load(args.plan)
    positions_path = ROOT / plan["positions"]
    if sha256(positions_path) != plan["positions_sha256"]:
        raise ValueError("Frozen position file hash mismatch")
    results_path = args.results or ROOT / plan["results"]
    analysis_path = args.analysis_output or ROOT / plan["analysis"]
    report_path = args.report_output or ROOT / plan["report"]
    analysis = analyze(plan, load(results_path))
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(analysis))
    print(f"Saved {analysis_path}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
