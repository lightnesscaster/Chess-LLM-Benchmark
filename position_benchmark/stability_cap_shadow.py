"""Prospective, no-production-effect shadow predictions for stability-cap designs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from position_benchmark.layout import (
    BLUNDER_POSITIONS_PATH,
    BLUNDER_RESULTS_PATH,
    CORE_POSITIONS_PATH,
    CORE_RESULTS_PATH,
    GAME_LIKE_POSITIONS_PATH,
    GAME_LIKE_RESULTS_PATH,
    STABILITY_RESULTS_PATH,
)
from position_benchmark.predictions import (
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


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT
    / "position_benchmark/validation/depth30-cap-prospective-policy.json"
)
SHADOW_LEDGER_PATH = (
    ROOT / "position_benchmark/results/stability_cap_shadow.json"
)
SHADOW_SCHEMA_VERSION = "depth30-cap-shadow-v1"
PRODUCTION_REFERENCE_CANDIDATE = "deduplicated_move_exposure_cap"


@dataclass(frozen=True)
class ShadowGameGuard:
    """Decision returned before a configuration may create saved game evidence."""

    allowed: bool
    status: str
    message: str
    record: dict[str, Any] | None = None


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _sha256_bytes(encoded)


def model_family(player_id: str) -> str:
    """Return a model-line family for correlated-evidence checks."""
    player = player_id.lower()
    prefixes = (
        "gemini-3.6",
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
    return next(
        (prefix for prefix in prefixes if player.startswith(prefix)),
        player.split(" (")[0],
    )


def model_lab(player_id: str) -> str:
    """Return the model developer used for correlated-evidence checks."""
    player = player_id.lower()
    lab_prefixes = {
        "openai": ("gpt-",),
        "google": ("gemini-", "gemma-"),
        "deepseek": ("deepseek-",),
        "alibaba": ("qwen",),
        "meta": ("llama-",),
        "mistral": ("mistral-",),
        "moonshot": ("kimi-",),
        "zhipu": ("glm-",),
    }
    for lab, prefixes in lab_prefixes.items():
        if player.startswith(prefixes):
            return lab
    return player.split(" (")[0]


def cap_from_rates(
    forfeit_pct: float,
    catastrophe_pct: float,
) -> float | None:
    """Apply the frozen production cap coefficients to supplied risk rates."""
    if forfeit_pct + catastrophe_pct < STABILITY_RISK_TRIGGER:
        return None
    cap = (
        STABILITY_CAP_BASE
        - STABILITY_FORFEIT_PENALTY * forfeit_pct
        - STABILITY_CATASTROPHE_PENALTY * catastrophe_pct
    )
    return max(STABILITY_CAP_FLOOR, min(STABILITY_CAP_BASE, cap))


def continuation_episode_metrics(
    record: dict[str, Any],
) -> dict[str, float | int]:
    """Summarize continuation failures without double-counting one trajectory."""
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
    scored_moves = sum(
        len(row.get("model_move_scores") or []) for row in rows
    )
    return {
        "attempted_positions": attempted,
        "scored_moves": scored_moves,
        "catastrophe_positions": catastrophe_positions,
        "forfeit_positions": forfeit_positions,
        "affected_positions": affected_positions,
        "catastrophe_episodes": catastrophe_episodes,
        "at_risk_moves": at_risk_moves,
        "raw_catastrophe_moves": raw_catastrophe_moves,
        "catastrophe_hazard_pct": (
            100.0 * catastrophe_episodes / at_risk_moves
            if at_risk_moves
            else 0.0
        ),
        "forfeit_pct": (
            100.0 * forfeit_positions / attempted if attempted else 0.0
        ),
    }


def _prediction_without_hard_cap(
    core_record: dict[str, Any],
    core_positions: list[dict[str, Any]],
    game_like_record: dict[str, Any],
    game_like_positions: list[dict[str, Any]],
    stability_record: dict[str, Any],
    blunder_record: dict[str, Any] | None,
    blunder_positions: list[dict[str, Any]] | None,
) -> float:
    neutral = copy.deepcopy(stability_record)
    neutral["summary"]["model_forfeit_pct"] = 0.0
    neutral["summary"]["model_1000cp_catastrophe_pct"] = 0.0
    neutral["summary"]["model_1000cp_catastrophe_positions"] = 0
    prediction = predict_rating_from_model_data_with_supplement(
        core_record,
        core_positions,
        blunder_model_data=blunder_record,
        blunder_positions=blunder_positions,
        game_like_model_data=game_like_record,
        game_like_positions=game_like_positions,
        stability_probe_model_data=neutral,
    )
    if prediction is None:
        raise ValueError("missing no-hard-cap prediction")
    return float(prediction)


def calculate_candidate_predictions(
    *,
    core_record: dict[str, Any],
    core_positions: list[dict[str, Any]],
    game_like_record: dict[str, Any],
    game_like_positions: list[dict[str, Any]],
    stability_record: dict[str, Any],
    blunder_record: dict[str, Any] | None = None,
    blunder_positions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calculate every frozen candidate from one complete automatic suite."""
    core_readiness = benchmark_result_readiness(core_record, core_positions)
    if not core_readiness.is_ready:
        raise ValueError(f"core is not ready: {core_readiness.reason}")
    game_readiness = benchmark_result_readiness(
        game_like_record,
        game_like_positions,
        min_equal_positions=48,
    )
    if not game_readiness.is_ready:
        raise ValueError(f"game-like is not ready: {game_readiness.reason}")
    stability_readiness = stability_probe_readiness(stability_record)
    if not stability_readiness.is_ready:
        raise ValueError(
            f"continuation stability is not ready: {stability_readiness.reason}"
        )

    production_prediction = predict_rating_from_model_data_with_supplement(
        core_record,
        core_positions,
        blunder_model_data=blunder_record,
        blunder_positions=blunder_positions,
        game_like_model_data=game_like_record,
        game_like_positions=game_like_positions,
        stability_probe_model_data=stability_record,
    )
    if production_prediction is None:
        raise ValueError("missing production prediction")
    production_prediction = float(production_prediction)
    no_cap = _prediction_without_hard_cap(
        core_record,
        core_positions,
        game_like_record,
        game_like_positions,
        stability_record,
        blunder_record,
        blunder_positions,
    )
    episodes = continuation_episode_metrics(stability_record)
    summary = stability_record["summary"]
    current_cap = cap_from_rates(
        float(summary.get("model_forfeit_pct", 0.0) or 0.0),
        float(summary.get("model_1000cp_catastrophe_pct", 0.0) or 0.0),
    )
    current_move_prediction = combine_prediction_with_downside_cap(
        no_cap,
        current_cap,
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
        float(episodes["forfeit_pct"]),
        deduplicated_catastrophe_pct,
    )
    deduplicated_prediction = combine_prediction_with_downside_cap(
        no_cap,
        deduplicated_cap,
    )
    if not math.isclose(
        production_prediction,
        deduplicated_prediction,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "production prediction does not match the frozen reference candidate"
        )
    repeated_forfeit_cap = (
        cap_from_rates(float(episodes["forfeit_pct"]), 0.0)
        if int(episodes["forfeit_positions"]) >= 2
        else None
    )
    candidates = {
        "current_move_cap": current_move_prediction,
        PRODUCTION_REFERENCE_CANDIDATE: deduplicated_prediction,
        "trajectory_hazard_cap": combine_prediction_with_downside_cap(
            no_cap,
            hazard_cap,
        ),
        "two_affected_trajectory_gate": (
            current_move_prediction
            if int(episodes["affected_positions"]) >= 2
            else no_cap
        ),
        "repeated_forfeit_only": combine_prediction_with_downside_cap(
            no_cap,
            repeated_forfeit_cap,
        ),
        "no_hard_cap": no_cap,
    }
    return {
        "production_reference_candidate": PRODUCTION_REFERENCE_CANDIDATE,
        "production_prediction": production_prediction,
        "candidates": candidates,
        "evidence": {
            **episodes,
            "score_depth": int(summary["score_depth"]),
            "current_cap": current_cap,
            "hazard_cap": hazard_cap,
            "deduplicated_cap": deduplicated_cap,
            "deduplicated_catastrophe_pct": deduplicated_catastrophe_pct,
        },
    }


def _load_policy_and_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    policy = _load_json(POLICY_PATH)
    freeze_config = policy["development_freeze"]
    freeze_path = ROOT / freeze_config["path"]
    actual_hash = _sha256_path(freeze_path)
    if actual_hash != freeze_config["sha256"]:
        raise ValueError(
            f"development freeze hash mismatch: {freeze_path}"
        )
    return policy, _load_json(freeze_path)


def _load_player_inputs(player_id: str) -> dict[str, Any]:
    core_results = _load_json(CORE_RESULTS_PATH)
    game_like_results = _load_json(GAME_LIKE_RESULTS_PATH)
    stability_results = _load_json(STABILITY_RESULTS_PATH)
    if player_id not in core_results:
        raise ValueError(f"{player_id}: missing core result")
    if player_id not in game_like_results:
        raise ValueError(f"{player_id}: missing game-like result")
    if player_id not in stability_results:
        raise ValueError(f"{player_id}: missing continuation result")

    blunder_results = (
        _load_json(BLUNDER_RESULTS_PATH)
        if BLUNDER_RESULTS_PATH.exists()
        else {}
    )
    return {
        "core_record": core_results[player_id],
        "core_positions": _load_json(CORE_POSITIONS_PATH)["positions"],
        "game_like_record": game_like_results[player_id],
        "game_like_positions": _load_json(GAME_LIKE_POSITIONS_PATH)["positions"],
        "stability_record": stability_results[player_id],
        "blunder_record": blunder_results.get(player_id),
        "blunder_positions": (
            _load_json(BLUNDER_POSITIONS_PATH)["positions"]
            if BLUNDER_POSITIONS_PATH.exists()
            else None
        ),
    }


def build_shadow_record(
    player_id: str,
    *,
    recorded_at: str | None = None,
    game_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one immutable prediction record before evaluating game outcomes."""
    policy, freeze = _load_policy_and_freeze()
    inputs = _load_player_inputs(player_id)
    prediction = calculate_candidate_predictions(**inputs)
    recorded_at = recorded_at or datetime.now(timezone.utc).isoformat()
    recorded_time = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
    frozen_time = datetime.fromisoformat(
        freeze["frozen_at"].replace("Z", "+00:00")
    )
    is_development = player_id in set(
        freeze["development_configuration_ids"]
    )
    games_at_recording = (
        int(game_snapshot.get("games_played", 0) or 0)
        if game_snapshot is not None
        else None
    )
    timing_verified = games_at_recording == 0
    eligible = (
        not is_development
        and recorded_time > frozen_time
        and timing_verified
    )
    if is_development:
        status = "development-excluded"
    elif recorded_time <= frozen_time:
        status = "pre-freeze-excluded"
    elif game_snapshot is None:
        status = "game-timing-unverified"
    elif not timing_verified:
        status = "games-already-observed"
    else:
        status = "prospective-holdout"

    return {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "player_id": player_id,
        "family": model_family(player_id),
        "lab": model_lab(player_id),
        "recorded_at": recorded_at,
        "prediction_locked": True,
        "production_effect": "none",
        "policy_version": policy["policy_version"],
        "policy_sha256": _sha256_path(POLICY_PATH),
        "eligibility": {
            "status": status,
            "prospective_holdout": eligible,
            "game_snapshot": game_snapshot,
        },
        **prediction,
        "source_fingerprints": {
            "core_record_sha256": _canonical_sha256(inputs["core_record"]),
            "game_like_record_sha256": _canonical_sha256(
                inputs["game_like_record"]
            ),
            "stability_record_sha256": _canonical_sha256(
                inputs["stability_record"]
            ),
            "blunder_record_sha256": (
                _canonical_sha256(inputs["blunder_record"])
                if inputs["blunder_record"] is not None
                else None
            ),
            "core_positions_sha256": _sha256_path(CORE_POSITIONS_PATH),
            "game_like_positions_sha256": _sha256_path(
                GAME_LIKE_POSITIONS_PATH
            ),
            "blunder_positions_sha256": (
                _sha256_path(BLUNDER_POSITIONS_PATH)
                if BLUNDER_POSITIONS_PATH.exists()
                else None
            ),
        },
    }


def record_shadow_prediction(
    player_id: str,
    *,
    ledger_path: Path = SHADOW_LEDGER_PATH,
    recorded_at: str | None = None,
    game_snapshot: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Append one locked record; development configurations are never re-enrolled."""
    _, freeze = _load_policy_and_freeze()
    if player_id in set(freeze["development_configuration_ids"]):
        return None, False

    if ledger_path.exists():
        ledger = _load_json(ledger_path)
    else:
        ledger = {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "production_effect": "none",
            "exclusions": {},
            "entries": {},
        }
    if ledger.get("schema_version") != SHADOW_SCHEMA_VERSION:
        raise ValueError(f"unexpected shadow ledger schema: {ledger_path}")
    entries = ledger.setdefault("entries", {})
    if player_id in entries:
        return entries[player_id], False

    record = build_shadow_record(
        player_id,
        recorded_at=recorded_at,
        game_snapshot=game_snapshot,
    )
    exclusion = ledger.setdefault("exclusions", {}).get(player_id)
    if exclusion is not None:
        record["eligibility"] = {
            **record["eligibility"],
            "status": "manual-save-opt-out",
            "prospective_holdout": False,
            "exclusion": exclusion,
        }
    entries[player_id] = record
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n")
    temporary.replace(ledger_path)
    return record, True


def record_nonprospective_exclusion(
    player_id: str,
    *,
    reason: str,
    ledger_path: Path = SHADOW_LEDGER_PATH,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Persist an explicit opt-out before a saved manual game is played."""
    if ledger_path.exists():
        ledger = _load_json(ledger_path)
    else:
        ledger = {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "production_effect": "none",
            "exclusions": {},
            "entries": {},
        }
    if ledger.get("schema_version") != SHADOW_SCHEMA_VERSION:
        raise ValueError(f"unexpected shadow ledger schema: {ledger_path}")
    if player_id in ledger.setdefault("entries", {}):
        return ledger["entries"][player_id].get("eligibility", {}).get(
            "exclusion",
            {},
        )
    exclusions = ledger.setdefault("exclusions", {})
    if player_id in exclusions:
        return exclusions[player_id]
    exclusion = {
        "recorded_at": recorded_at or datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "production_effect": "none",
    }
    exclusions[player_id] = exclusion
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n")
    temporary.replace(ledger_path)
    return exclusion


def ensure_shadow_lock_before_saved_game(
    player_id: str,
    *,
    game_snapshot: dict[str, Any],
    ledger_path: Path = SHADOW_LEDGER_PATH,
) -> ShadowGameGuard:
    """Fail closed for a future configuration's first saved rated game."""
    try:
        _, freeze = _load_policy_and_freeze()
        if player_id in set(freeze["development_configuration_ids"]):
            return ShadowGameGuard(
                True,
                "development-excluded",
                "configuration belongs to the frozen development cohort",
            )

        ledger = (
            _load_json(ledger_path)
            if ledger_path.exists()
            else {
                "schema_version": SHADOW_SCHEMA_VERSION,
                "production_effect": "none",
                "exclusions": {},
                "entries": {},
            }
        )
        if ledger.get("schema_version") != SHADOW_SCHEMA_VERSION:
            raise ValueError(f"unexpected shadow ledger schema: {ledger_path}")
        existing = ledger.get("entries", {}).get(player_id)
        if existing is not None:
            if not existing.get("prediction_locked"):
                raise ValueError("existing shadow entry is not prediction-locked")
            return ShadowGameGuard(
                True,
                existing["eligibility"]["status"],
                "immutable shadow prediction already exists",
                existing,
            )
        if player_id in ledger.get("exclusions", {}):
            return ShadowGameGuard(
                True,
                "manual-save-opt-out",
                "configuration was explicitly excluded from prospective evidence",
            )

        games_played = int(game_snapshot.get("games_played", 0) or 0)
        if games_played > 0:
            return ShadowGameGuard(
                True,
                "games-already-observed",
                "configuration predates prospective shadow enrollment",
            )

        record, _ = record_shadow_prediction(
            player_id,
            ledger_path=ledger_path,
            game_snapshot=game_snapshot,
        )
        if record is None or not record.get("prediction_locked"):
            raise ValueError("shadow prediction was not locked")
        return ShadowGameGuard(
            True,
            record["eligibility"]["status"],
            "immutable shadow prediction locked before first saved game",
            record,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return ShadowGameGuard(
            False,
            "shadow-lock-failed",
            str(error),
        )


def affected_prospective_player_ids(
    *,
    policy_path: Path = POLICY_PATH,
    ledger_path: Path = SHADOW_LEDGER_PATH,
) -> set[str]:
    """Return affected prospective IDs without inspecting any game outcome."""
    policy = _load_json(policy_path)
    ledger = _load_json(ledger_path)
    reference = policy["comparison"]["reference_candidate"]
    challenger = policy["comparison"]["challenger_candidate"]
    return {
        player_id
        for player_id, record in ledger.get("entries", {}).items()
        if record.get("eligibility", {}).get("prospective_holdout")
        and not math.isclose(
            float(record["candidates"][reference]),
            float(record["candidates"][challenger]),
            abs_tol=1e-9,
        )
    }


def shadow_priority_settings(
    *,
    policy_path: Path = POLICY_PATH,
) -> tuple[int, float, float]:
    """Return the fixed maturity gate and collection-priority multiplier."""
    policy = _load_json(policy_path)
    target = policy["primary_target"]
    collection = policy["collection_control"]
    return (
        int(target["minimum_games"]),
        float(target["maximum_games_rd"]),
        float(collection["affected_immature_priority_multiplier"]),
    )
