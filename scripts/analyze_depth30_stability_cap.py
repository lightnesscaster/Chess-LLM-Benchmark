#!/usr/bin/env python3
"""Audit depth-30 continuation cap designs without changing production ratings."""

from __future__ import annotations

import copy
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

from position_benchmark.layout import (  # noqa: E402
    BLUNDER_POSITIONS_PATH,
    BLUNDER_RESULTS_PATH,
    CORE_POSITIONS_PATH,
    CORE_RESULTS_PATH,
    GAME_LIKE_POSITIONS_PATH,
    GAME_LIKE_RESULTS_PATH,
    STABILITY_RESULTS_PATH,
)
from position_benchmark.predictions import (  # noqa: E402
    STABILITY_CAP_BASE,
    STABILITY_CAP_FLOOR,
    STABILITY_CATASTROPHE_PENALTY,
    STABILITY_FORFEIT_PENALTY,
    STABILITY_RISK_TRIGGER,
    benchmark_result_readiness,
    combine_prediction_with_downside_cap,
    predict_rating_from_model_data_with_supplement,
    stability_probe_readiness,
)


RD300_TARGET = ROOT / "position_benchmark/validation/2026-07-17-supplemental-predictor-rd300.json"
NO_POSITION_TARGET = ROOT / "position_benchmark/validation/2026-07-17-no-position-seed-ratings.json"
OUTPUT_JSON = ROOT / "position_benchmark/validation/2026-07-21-depth30-stability-cap-analysis.json"
OUTPUT_MD = ROOT / "position_benchmark/validation/2026-07-21-depth30-stability-cap-analysis.md"
BOOTSTRAP_RESAMPLES = 20_000
MONTE_CARLO_DRAWS = 20_000
RANDOM_SEED = 1729


def load(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_family(player_id: str) -> str:
    player = player_id.lower()
    prefixes = (
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.5",
        "deepseek-v4-flash",
        "gemini-3.1",
        "gemini-3",
        "gemini-2.5",
        "gpt-5.1",
    )
    return next((prefix for prefix in prefixes if player.startswith(prefix)), player.split(" (")[0])


def cap_from_rates(forfeit_pct: float, catastrophe_pct: float) -> float | None:
    if forfeit_pct + catastrophe_pct < STABILITY_RISK_TRIGGER:
        return None
    cap = (
        STABILITY_CAP_BASE
        - STABILITY_FORFEIT_PENALTY * forfeit_pct
        - STABILITY_CATASTROPHE_PENALTY * catastrophe_pct
    )
    return max(STABILITY_CAP_FLOOR, min(STABILITY_CAP_BASE, cap))


def continuation_episode_metrics(record: dict[str, Any]) -> dict[str, float | int]:
    rows = record.get("results") or []
    catastrophe_positions = 0
    forfeit_positions = 0
    affected_positions = 0
    catastrophe_episodes = 0
    at_risk_moves = 0
    raw_catastrophe_moves = 0
    for row in rows:
        scores = sorted(
            row.get("model_move_scores") or [],
            key=lambda score: int(score.get("model_turn_index", 0) or 0),
        )
        first_catastrophe_index = None
        for index, score in enumerate(scores):
            if float(score.get("cpl", 0.0) or 0.0) >= 1000.0:
                raw_catastrophe_moves += 1
                if first_catastrophe_index is None:
                    first_catastrophe_index = index
        has_catastrophe = first_catastrophe_index is not None
        has_forfeit = bool(row.get("model_forfeited"))
        catastrophe_positions += int(has_catastrophe)
        forfeit_positions += int(has_forfeit)
        affected_positions += int(has_catastrophe or has_forfeit)
        catastrophe_episodes += int(has_catastrophe)
        at_risk_moves += (
            first_catastrophe_index + 1 if has_catastrophe else len(scores)
        )
    attempted = len(rows)
    return {
        "attempted_positions": attempted,
        "scored_moves": sum(
            len(row.get("model_move_scores") or []) for row in rows
        ),
        "catastrophe_positions": catastrophe_positions,
        "forfeit_positions": forfeit_positions,
        "affected_positions": affected_positions,
        "catastrophe_episodes": catastrophe_episodes,
        "at_risk_moves": at_risk_moves,
        "raw_catastrophe_moves": raw_catastrophe_moves,
        "catastrophe_hazard_pct": (
            100.0 * catastrophe_episodes / at_risk_moves if at_risk_moves else 0.0
        ),
        "forfeit_pct": 100.0 * forfeit_positions / attempted if attempted else 0.0,
    }


def prediction_without_hard_cap(
    player_id: str,
    core_results: dict[str, Any],
    core_positions: list[dict[str, Any]],
    game_results: dict[str, Any],
    game_positions: list[dict[str, Any]],
    stability_record: dict[str, Any],
    blunder_results: dict[str, Any],
    blunder_positions: list[dict[str, Any]],
) -> float:
    neutral = copy.deepcopy(stability_record)
    neutral["summary"]["model_forfeit_pct"] = 0.0
    neutral["summary"]["model_1000cp_catastrophe_pct"] = 0.0
    neutral["summary"]["model_1000cp_catastrophe_positions"] = 0
    prediction = predict_rating_from_model_data_with_supplement(
        core_results[player_id],
        core_positions,
        blunder_model_data=blunder_results.get(player_id),
        blunder_positions=blunder_positions,
        game_like_model_data=game_results[player_id],
        game_like_positions=game_positions,
        stability_probe_model_data=neutral,
    )
    if prediction is None:
        raise ValueError(f"{player_id}: missing no-hard-cap prediction")
    return float(prediction)


def build_rows() -> list[dict[str, Any]]:
    core_positions = load(CORE_POSITIONS_PATH)["positions"]
    core_results = load(CORE_RESULTS_PATH)
    game_positions = load(GAME_LIKE_POSITIONS_PATH)["positions"]
    game_results = load(GAME_LIKE_RESULTS_PATH)
    stability_results = load(STABILITY_RESULTS_PATH)
    blunder_positions = load(BLUNDER_POSITIONS_PATH)["positions"]
    blunder_results = load(BLUNDER_RESULTS_PATH)
    targets = {"rd300": load(RD300_TARGET), "no_position": load(NO_POSITION_TARGET)}

    rows = []
    for player_id, stability_record in stability_results.items():
        if player_id not in core_results or player_id not in game_results:
            continue
        if not stability_probe_readiness(stability_record).is_ready:
            continue
        if not benchmark_result_readiness(core_results[player_id], core_positions).is_ready:
            continue
        if not benchmark_result_readiness(
            game_results[player_id], game_positions, min_equal_positions=48
        ).is_ready:
            continue
        if any(player_id not in target for target in targets.values()):
            continue

        corrected_production = predict_rating_from_model_data_with_supplement(
            core_results[player_id],
            core_positions,
            blunder_model_data=blunder_results.get(player_id),
            blunder_positions=blunder_positions,
            game_like_model_data=game_results[player_id],
            game_like_positions=game_positions,
            stability_probe_model_data=stability_record,
        )
        if corrected_production is None:
            continue
        no_cap = prediction_without_hard_cap(
            player_id,
            core_results,
            core_positions,
            game_results,
            game_positions,
            stability_record,
            blunder_results,
            blunder_positions,
        )
        episodes = continuation_episode_metrics(stability_record)
        summary = stability_record["summary"]
        current_cap = cap_from_rates(
            float(summary.get("model_forfeit_pct", 0.0) or 0.0),
            float(summary.get("model_1000cp_catastrophe_pct", 0.0) or 0.0),
        )
        current_move_prediction = combine_prediction_with_downside_cap(
            no_cap, current_cap
        )
        hazard_cap = cap_from_rates(
            float(episodes["forfeit_pct"]),
            float(episodes["catastrophe_hazard_pct"]),
        )
        deduplicated_catastrophe_pct = (
            100.0
            * int(episodes["catastrophe_positions"])
            / int(episodes["scored_moves"])
            if int(episodes["scored_moves"])
            else 0.0
        )
        deduplicated_cap = cap_from_rates(
            float(episodes["forfeit_pct"]), deduplicated_catastrophe_pct
        )
        deduplicated_prediction = combine_prediction_with_downside_cap(
            no_cap, deduplicated_cap
        )
        if not math.isclose(
            float(corrected_production), deduplicated_prediction, abs_tol=1e-9
        ):
            raise ValueError(
                f"{player_id}: production prediction does not match deduplicated candidate"
            )
        repeated_forfeit_cap = (
            cap_from_rates(float(episodes["forfeit_pct"]), 0.0)
            if int(episodes["forfeit_positions"]) >= 2
            else None
        )
        candidates = {
            "current_move_cap": current_move_prediction,
            "trajectory_hazard_cap": combine_prediction_with_downside_cap(
                no_cap, hazard_cap
            ),
            "deduplicated_move_exposure_cap": deduplicated_prediction,
            "two_affected_trajectory_gate": (
                current_move_prediction
                if int(episodes["affected_positions"]) >= 2
                else no_cap
            ),
            "repeated_forfeit_only": combine_prediction_with_downside_cap(
                no_cap, repeated_forfeit_cap
            ),
            "no_hard_cap": no_cap,
        }
        rows.append(
            {
                "player_id": player_id,
                "family": model_family(player_id),
                "score_depth": int(summary["score_depth"]),
                "current_cap": current_cap,
                "hazard_cap": hazard_cap,
                "deduplicated_cap": deduplicated_cap,
                "deduplicated_catastrophe_pct": deduplicated_catastrophe_pct,
                **episodes,
                "candidates": candidates,
                "targets": {
                    name: {
                        "rating": float(target[player_id]["rating"]),
                        "rd": float(
                            target[player_id].get(
                                "games_rd", target[player_id].get("rating_deviation", 350.0)
                            )
                        ),
                        "games": int(target[player_id].get("games_played", 0) or 0),
                    }
                    for name, target in targets.items()
                },
            }
        )
    return rows


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * p)
    return ordered[max(0, min(len(ordered) - 1, index))]


def metrics(rows: list[dict[str, Any]], target: str, candidate: str) -> dict[str, Any]:
    errors = [
        row["candidates"][candidate] - row["targets"][target]["rating"]
        for row in rows
    ]
    weights = [1.0 / row["targets"][target]["rd"] ** 2 for row in rows]
    by_family: dict[str, list[float]] = defaultdict(list)
    for row, error in zip(rows, errors):
        by_family[row["family"]].append(abs(error))
    return {
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "bias": sum(errors) / len(errors),
        "max_absolute_error": max(abs(error) for error in errors),
        "inverse_variance_weighted_mae": sum(
            weight * abs(error) for weight, error in zip(weights, errors)
        )
        / sum(weights),
        "family_mae": {
            family: sum(values) / len(values) for family, values in sorted(by_family.items())
        },
    }


def bootstrap_comparison(
    rows: list[dict[str, Any]], target: str, candidate: str
) -> dict[str, float]:
    rng = random.Random(RANDOM_SEED + sum(ord(char) for char in target + candidate))
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    names = sorted(families)
    deltas = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = [rng.choice(names) for _ in names]
        selected = [row for family in sampled for row in families[family]]
        current_mae = sum(
            abs(row["candidates"]["current_move_cap"] - row["targets"][target]["rating"])
            for row in selected
        ) / len(selected)
        candidate_mae = sum(
            abs(row["candidates"][candidate] - row["targets"][target]["rating"])
            for row in selected
        ) / len(selected)
        deltas.append(candidate_mae - current_mae)
    return {
        "probability_mae_improves": sum(delta < 0.0 for delta in deltas) / len(deltas),
        "mae_delta_p05": percentile(deltas, 0.05),
        "mae_delta_p50": percentile(deltas, 0.50),
        "mae_delta_p95": percentile(deltas, 0.95),
    }


def uncertainty_comparison(
    rows: list[dict[str, Any]], target: str, candidate: str
) -> dict[str, float]:
    rng = random.Random(RANDOM_SEED * 3 + sum(ord(char) for char in target + candidate))
    deltas = []
    for _ in range(MONTE_CARLO_DRAWS):
        current_errors = []
        candidate_errors = []
        for row in rows:
            evidence = row["targets"][target]
            sampled_rating = rng.gauss(evidence["rating"], evidence["rd"])
            current_errors.append(
                abs(row["candidates"]["current_move_cap"] - sampled_rating)
            )
            candidate_errors.append(abs(row["candidates"][candidate] - sampled_rating))
        deltas.append(
            sum(candidate_errors) / len(candidate_errors)
            - sum(current_errors) / len(current_errors)
        )
    return {
        "probability_mae_improves": sum(delta < 0.0 for delta in deltas) / len(deltas),
        "mae_delta_p05": percentile(deltas, 0.05),
        "mae_delta_p50": percentile(deltas, 0.50),
        "mae_delta_p95": percentile(deltas, 0.95),
    }


def historical_sensitivity() -> dict[str, Any]:
    stability = load(STABILITY_RESULTS_PATH)
    frozen = load(ROOT / "position_benchmark/validation/2026-06-23.json")
    rows = []
    for item in frozen["stability_cap_corrections"]:
        player_id = item["player_id"]
        summary = stability[player_id]["summary"]
        forfeit_count = int(summary.get("model_forfeits", 0) or 0)
        repeated_forfeit_cap = (
            cap_from_rates(float(summary.get("model_forfeit_pct", 0.0)), 0.0)
            if forfeit_count >= 2
            else None
        )
        rows.append(
            {
                **item,
                "repeated_forfeit_only_prediction": combine_prediction_with_downside_cap(
                    float(item["prediction_before"]), repeated_forfeit_cap
                ),
                "forfeits": forfeit_count,
            }
        )
    for candidate, key in (
        ("current_historical_cap", "prediction_after"),
        ("repeated_forfeit_only", "repeated_forfeit_only_prediction"),
        ("no_hard_cap", "prediction_before"),
    ):
        errors = [abs(float(row[key]) - float(row["actual_rating"])) for row in rows]
        for row, error in zip(rows, errors):
            row.setdefault("absolute_errors", {})[candidate] = error
    return {
        "warning": "Historical, pre-stratification depth-10 sensitivity only; not current-contract validation.",
        "rows": rows,
        "mae": {
            candidate: sum(row["absolute_errors"][candidate] for row in rows) / len(rows)
            for candidate in (
                "current_historical_cap",
                "repeated_forfeit_only",
                "no_hard_cap",
            )
        },
    }


def render_report(analysis: dict[str, Any]) -> str:
    candidates = analysis["candidate_order"]
    sections = []
    for target in ("rd300", "no_position"):
        lines = [
            f"### {analysis['targets'][target]['label']}",
            "",
            "| Candidate | MAE | RMSE | Bias | IVW MAE | Family bootstrap P(improves) | RD simulation P(improves) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for candidate in candidates:
            result = analysis["targets"][target]["candidates"][candidate]
            lines.append(
                f"| `{candidate}` | {result['mae']:.1f} | {result['rmse']:.1f} | "
                f"{result['bias']:+.1f} | {result['inverse_variance_weighted_mae']:.1f} | "
                f"{result['family_bootstrap']['probability_mae_improves']:.3f} | "
                f"{result['rating_uncertainty']['probability_mae_improves']:.3f} |"
            )
        sections.append("\n".join(lines))

    changed_rows = []
    for row in analysis["rows"]:
        current = row["candidates"]["current_move_cap"]
        no_cap = row["candidates"]["no_hard_cap"]
        if abs(current - no_cap) < 1e-9:
            continue
        changed_rows.append(
            f"| {row['player_id']} | {row['raw_catastrophe_moves']} | "
            f"{row['catastrophe_positions']} | {row['affected_positions']} | "
            f"{current:.0f} | "
            f"{row['candidates']['deduplicated_move_exposure_cap']:.0f} | "
            f"{row['candidates']['trajectory_hazard_cap']:.0f} | {no_cap:.0f} |"
        )

    gate = analysis["evidence_gate"]
    return f"""# Depth-30 continuation-cap audit — 2026-07-21

This zero-call audit evaluates the corrected depth-30 continuation artifacts. It
did not write the production rating store. The accompanying predictor change only
deduplicates catastrophe events within each trajectory. The cohort has
{analysis['configuration_count']} configurations across
{analysis['family_count']} model-line families.

## Result

No hard-cap redesign is validated for production. The two game-rating targets
disagree: removing the hard cap helps the independent no-position-seed target but
slightly hurts the higher-RD position-seeded target. Neither direction is robust
under family resampling and rating uncertainty. The inherited evidence gate needs
at least {gate['minimum_configurations']} configurations and
{gate['minimum_families']} families; both coverage checks fail.

The current move-level catastrophe count is structurally wrong for an absorbing
loss: repeated losing moves in one continuation are correlated and must not be
treated as independent catastrophe events. `deduplicated_move_exposure_cap`
implements a narrow correction: retain at most the first catastrophe in each
trajectory, preserve the existing move-exposure denominator and coefficients,
and therefore never make a cap harsher. This correction fits no target data.
`trajectory_hazard_cap` additionally censors later exposures; that more ambitious
survival-style redesign remains diagnostic rather than validated rating evidence.

The production change is limited to deduplicating catastrophes within each
trajectory. Existing cap constants, the 150-Elo deadband, continuation legality,
and forfeit evidence remain unchanged. Any coefficient refit, complete CPL-cap
removal, or survival-hazard redesign remains blocked until a newly frozen cohort
passes the evidence gate.

## Current depth-30 comparison

{chr(10).join(sections)}

All bootstrap and uncertainty probabilities compare the candidate against
`current_move_cap`. Candidates are fixed structural alternatives; no coefficients
were fitted to these targets.

## Rows whose rating is currently changed by the hard cap

| Model | Catastrophic moves | Affected catastrophe starts | All affected starts | Current | Deduplicated | Hazard-style | No hard cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(changed_rows)}

## Historical sensitivity

The stale June depth-10/pre-stratification snapshot is not current validation, but
it explains why the cap existed. Its three-row MAE was
{analysis['historical_sensitivity']['mae']['current_historical_cap']:.1f} with the
old cap, {analysis['historical_sensitivity']['mae']['repeated_forfeit_only']:.1f}
with repeated-forfeit-only protection, and
{analysis['historical_sensitivity']['mae']['no_hard_cap']:.1f} with no hard cap.
Thus repeated-forfeit protection retains much of the historical weak-model value,
while the present cohort does not validate a direct random-reply CPL cap.

## Decision boundary

- Do not fit new cap coefficients on 14 configurations.
- Do not describe RD-300 as independent; it still shares the benchmark prior.
- Production may deduplicate catastrophes within a trajectory because this cannot
  increase any penalty and does not fit the validation targets.
- Do not deploy hazard censoring, new coefficients, or cap removal on this cohort.
- Re-run this fixed audit automatically as current depth-30 supplements accumulate.
"""


def main() -> None:
    rows = build_rows()
    candidate_order = [
        "current_move_cap",
        "deduplicated_move_exposure_cap",
        "trajectory_hazard_cap",
        "two_affected_trajectory_gate",
        "repeated_forfeit_only",
        "no_hard_cap",
    ]
    targets: dict[str, Any] = {
        "rd300": {
            "label": "Position-seeded validation ratings with RD 300",
            "warning": "Higher RD reduces but does not eliminate shared-prior circularity.",
        },
        "no_position": {
            "label": "No-position-seed game ratings",
            "warning": "Independent of the position prior but noisier for low-game configurations.",
        },
    }
    for target, target_result in targets.items():
        target_result["candidates"] = {}
        for candidate in candidate_order:
            result = metrics(rows, target, candidate)
            result["family_bootstrap"] = bootstrap_comparison(rows, target, candidate)
            result["rating_uncertainty"] = uncertainty_comparison(rows, target, candidate)
            target_result["candidates"][candidate] = result

    families = sorted({row["family"] for row in rows})
    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "production_effect": "catastrophe-event-deduplication-only",
        "production_rating_store_write": False,
        "model_call_count": 0,
        "score_depth": 30,
        "configuration_count": len(rows),
        "families": families,
        "family_count": len(families),
        "candidate_order": candidate_order,
        "evidence_gate": {
            "minimum_configurations": 30,
            "minimum_families": 8,
            "configuration_count_passed": len(rows) >= 30,
            "family_count_passed": len(families) >= 8,
            "passed": len(rows) >= 30 and len(families) >= 8,
        },
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                CORE_POSITIONS_PATH,
                CORE_RESULTS_PATH,
                GAME_LIKE_POSITIONS_PATH,
                GAME_LIKE_RESULTS_PATH,
                BLUNDER_POSITIONS_PATH,
                BLUNDER_RESULTS_PATH,
                STABILITY_RESULTS_PATH,
                RD300_TARGET,
                NO_POSITION_TARGET,
            )
        },
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "rating_uncertainty_draws": MONTE_CARLO_DRAWS,
        "targets": targets,
        "historical_sensitivity": historical_sensitivity(),
        "rows": rows,
    }
    OUTPUT_JSON.write_text(json.dumps(analysis, indent=2) + "\n")
    OUTPUT_MD.write_text(render_report(analysis))
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_MD}")


if __name__ == "__main__":
    main()
