#!/usr/bin/env python3
"""Verify the manifest, stable-ID panels, results, and legacy mappings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "position_benchmark"


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def resolve_manifest_path(value: str) -> Path:
    return ROOT / value


def verify_panel(
    name: str,
    config: dict[str, Any],
    *,
    require_complete_results: bool,
) -> list[str]:
    issues: list[str] = []
    positions_path = resolve_manifest_path(config["positions"])
    results_path = resolve_manifest_path(config["results"])
    if not positions_path.exists():
        return [f"{name}: missing positions file {positions_path}"]
    if not results_path.exists():
        return [f"{name}: missing results file {results_path}"]

    positions_data = load_json(positions_path)
    positions = positions_data.get("positions", [])
    expected_count = int(config["position_count"])
    if len(positions) != expected_count:
        issues.append(f"{name}: {len(positions)} positions != manifest {expected_count}")
    if positions_data.get("metadata", {}).get("position_count") != expected_count:
        issues.append(f"{name}: panel metadata count does not match manifest")

    position_ids = [position.get("position_id") for position in positions]
    if any(not position_id for position_id in position_ids):
        issues.append(f"{name}: position without position_id")
    if len(position_ids) != len(set(position_ids)):
        issues.append(f"{name}: duplicate position_id")
    fens = [position.get("fen") for position in positions]
    if len(fens) != len(set(fens)):
        issues.append(f"{name}: duplicate FEN")

    required_position_fields = {
        "position_id",
        "panel",
        "panel_index",
        "type",
        "fen",
        "eval_before",
        "best_move",
        "move_history",
    }
    for index, position in enumerate(positions):
        missing = required_position_fields - position.keys()
        if missing:
            issues.append(f"{name}: position {index} missing {sorted(missing)}")
        if position.get("panel_index") != index:
            issues.append(f"{name}: {position.get('position_id')} panel_index != {index}")

    results = load_json(results_path)
    required_result_fields = {
        "position_id",
        "result_schema_version",
        "position_idx",
        "panel",
        "fen",
        "model_move",
        "best_move",
        "cpl",
        "is_legal",
        "is_best",
    }
    for player_id, player_data in results.items():
        rows = player_data.get("results", [])
        if require_complete_results and len(rows) != expected_count:
            issues.append(f"{name}/{player_id}: {len(rows)} rows != {expected_count}")
        seen: set[str] = set()
        for row in rows:
            missing = required_result_fields - row.keys()
            if missing:
                issues.append(f"{name}/{player_id}: row missing {sorted(missing)}")
                continue
            index = row["position_idx"]
            if not isinstance(index, int) or not 0 <= index < len(positions):
                issues.append(f"{name}/{player_id}: invalid index {index}")
                continue
            position = positions[index]
            if row["position_id"] != position["position_id"]:
                issues.append(f"{name}/{player_id}: ID mismatch at index {index}")
            if row["result_schema_version"] != 2:
                issues.append(f"{name}/{player_id}: unexpected result schema version")
            if row["fen"] != position["fen"]:
                issues.append(f"{name}/{player_id}: FEN mismatch at {row['position_id']}")
            if row["position_id"] in seen:
                issues.append(f"{name}/{player_id}: duplicate {row['position_id']}")
            seen.add(row["position_id"])

        if not rows:
            continue
        summary = player_data.get("summary", {})
        legal_pct = 100.0 * sum(bool(row["is_legal"]) for row in rows) / len(rows)
        best_pct = 100.0 * sum(bool(row["is_best"]) for row in rows) / len(rows)
        avg_cpl = sum(float(row["cpl"]) for row in rows) / len(rows)
        for key, computed, tolerance in (
            ("legal_pct", legal_pct, 0.01),
            ("best_pct", best_pct, 0.01),
            ("avg_cpl", avg_cpl, 0.1),
        ):
            if key not in summary or abs(float(summary[key]) - computed) > tolerance:
                issues.append(f"{name}/{player_id}: summary {key} mismatch")

    print(f"  {name}: {len(positions)} positions, {len(results)} model result sets")
    return issues


def verify_legacy_mapping(manifest: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    legacy_path = resolve_manifest_path(manifest["legacy"]["combined_position_registry"])
    legacy_positions = load_json(legacy_path)["positions"]
    for panel_name in ("core", "blunder"):
        panel_path = resolve_manifest_path(manifest["panels"][panel_name]["positions"])
        for position in load_json(panel_path)["positions"]:
            legacy_index = position.get("legacy_position_idx")
            if not isinstance(legacy_index, int) or not 0 <= legacy_index < len(legacy_positions):
                issues.append(f"{panel_name}: invalid legacy index for {position.get('position_id')}")
                continue
            legacy = legacy_positions[legacy_index]
            if position.get("fen") != legacy.get("fen"):
                issues.append(f"{panel_name}: legacy FEN mismatch for {position.get('position_id')}")
            if position.get("best_move") != legacy.get("best_move"):
                issues.append(f"{panel_name}: legacy best-move mismatch for {position.get('position_id')}")
    return issues


def verify_automatic_acquisition(manifest: dict[str, Any]) -> list[str]:
    """Verify the deterministic core + supplemental scheduler contract."""
    issues: list[str] = []
    config = manifest.get("automatic_acquisition", {})
    expected_order = ["core", "game_like", "continuation_stability"]
    if config.get("enabled") is not True:
        issues.append("automatic_acquisition: must remain enabled")
    if config.get("policy_version") != "production-supplements-depth30-v2":
        issues.append("automatic_acquisition: policy version mismatch")
    if config.get("panel_order") != expected_order:
        issues.append("automatic_acquisition: panel order mismatch")
    if config.get("completion_policy") != "current-readiness-v2":
        issues.append("automatic_acquisition: readiness policy mismatch")
    if (
        config.get("legacy_static_retry_policy")
        != "additive-does-not-invalidate-ready-rows"
    ):
        issues.append("automatic_acquisition: static retry compatibility mismatch")
    if config.get("resume_policy") != "first-missing-panel":
        issues.append("automatic_acquisition: resume policy mismatch")
    if config.get("defer_games_after_acquisition") is not True:
        issues.append("automatic_acquisition: acquired models must defer games")

    for panel_name in expected_order:
        if manifest["panels"].get(panel_name, {}).get("acquisition") != "automatic":
            issues.append(f"automatic_acquisition: {panel_name} is not automatic")

    excluded = set(config.get("excluded_panels", []))
    expected_excluded = set(manifest["panels"]) - set(expected_order)
    if excluded != expected_excluded:
        issues.append("automatic_acquisition: excluded panel set mismatch")
    for panel_name in excluded:
        if manifest["panels"].get(panel_name, {}).get("acquisition") == "automatic":
            issues.append(f"automatic_acquisition: excluded {panel_name} is automatic")

    continuation = manifest["panels"]["continuation_stability"]
    if continuation.get("stability_probe_version") != "stratified-depth30-v3":
        issues.append("automatic_acquisition: continuation probe version mismatch")
    if int(continuation.get("stockfish_score_depth", 0)) != 30:
        issues.append("automatic_acquisition: continuation depth must remain 30")
    if (
        continuation.get("catastrophe_event_policy")
        != "first-1000cp-event-per-start-v1"
    ):
        issues.append("automatic_acquisition: catastrophe event policy mismatch")
    if continuation.get("catastrophe_rate_denominator") != "scored_model_moves":
        issues.append("automatic_acquisition: catastrophe denominator mismatch")
    expected_calls = (
        int(continuation.get("default_starting_positions", 0))
        * ((int(continuation.get("probe_plies", 0)) + 1) // 2)
    )
    if int(continuation.get("planned_base_first_attempt_calls", 0)) != expected_calls:
        issues.append("automatic_acquisition: continuation call count mismatch")
    print(f"  automatic_acquisition: {', '.join(expected_order)}")
    return issues


def verify_prospective_stability_cap(
    manifest: dict[str, Any],
) -> list[str]:
    """Verify that shadow predictions stay prospective and research-only."""
    issues: list[str] = []
    config = manifest.get("prospective_stability_cap_validation", {})
    if config.get("production_effect") != "none":
        issues.append("stability_cap_shadow: production_effect must remain none")
    if (
        config.get("recording_timing")
        != "after-current-automatic-suite-before-first-game"
    ):
        issues.append("stability_cap_shadow: recording timing mismatch")

    required_paths = (
        "development_freeze",
        "policy",
        "shadow_ledger",
        "game_only_target",
        "evaluation",
        "report",
    )
    resolved = {}
    for key in required_paths:
        value = config.get(key)
        if not isinstance(value, str) or not value:
            issues.append(f"stability_cap_shadow: missing {key} manifest path")
            continue
        path = resolve_manifest_path(value)
        resolved[key] = path
        if not path.exists():
            issues.append(f"stability_cap_shadow: missing {key} file {path}")
    if issues:
        return issues

    policy = load_json(resolved["policy"])
    ledger = load_json(resolved["shadow_ledger"])
    if policy.get("production_effect") != "none":
        issues.append("stability_cap_shadow: policy can affect production")
    if ledger.get("production_effect") != "none":
        issues.append("stability_cap_shadow: ledger can affect production")
    if ledger.get("schema_version") != "depth30-cap-shadow-v1":
        issues.append("stability_cap_shadow: ledger schema mismatch")
    comparison = policy.get("comparison", {})
    if comparison.get("reference_candidate") != config.get(
        "reference_candidate"
    ):
        issues.append("stability_cap_shadow: reference candidate mismatch")
    if comparison.get("challenger_candidate") != config.get(
        "challenger_candidate"
    ):
        issues.append("stability_cap_shadow: challenger candidate mismatch")

    freeze_config = policy.get("development_freeze", {})
    if freeze_config.get("path") != config.get("development_freeze"):
        issues.append("stability_cap_shadow: development freeze path mismatch")
    actual_freeze_hash = hashlib.sha256(
        resolved["development_freeze"].read_bytes()
    ).hexdigest()
    if freeze_config.get("sha256") != actual_freeze_hash:
        issues.append("stability_cap_shadow: development freeze hash mismatch")
    if any(
        entry.get("production_effect") != "none"
        or entry.get("prediction_locked") is not True
        for entry in ledger.get("entries", {}).values()
    ):
        issues.append("stability_cap_shadow: unlocked or production-active entry")
    policy_hash = hashlib.sha256(resolved["policy"].read_bytes()).hexdigest()
    if any(
        entry.get("policy_version") != policy.get("policy_version")
        or entry.get("policy_sha256") != policy_hash
        for entry in ledger.get("entries", {}).values()
    ):
        issues.append("stability_cap_shadow: entry policy fingerprint mismatch")
    print(
        "  stability_cap_shadow: "
        f"{len(ledger.get('entries', {}))} locked prospective record(s)"
    )
    return issues


def verify_stability_results(manifest: dict[str, Any]) -> list[str]:
    """Verify replayable production continuation rows against the current contract."""
    issues: list[str] = []
    config = manifest["panels"]["continuation_stability"]
    results_path = resolve_manifest_path(config["results"])
    if not results_path.exists():
        return [f"stability: missing results file {results_path}"]

    expected_depth = int(config["stockfish_score_depth"])
    expected_indices = config["selected_position_indices"]
    results = load_json(results_path)
    replayable = 0
    for player_id, record in results.items():
        rows = record.get("results") or []
        if not rows:
            continue
        replayable += 1
        summary = record.get("summary", {})
        if summary.get("stability_probe_version") != config.get(
            "stability_probe_version"
        ):
            issues.append(f"stability/{player_id}: probe version mismatch")
        if summary.get("position_selection_policy") != config.get("selection_policy"):
            issues.append(f"stability/{player_id}: selection policy mismatch")
        if summary.get("selected_position_indices") != expected_indices:
            issues.append(f"stability/{player_id}: summary indices mismatch")
        if int(summary.get("score_depth", 0)) != expected_depth:
            issues.append(f"stability/{player_id}: summary depth mismatch")
        if any(int(row.get("score_depth", 0)) != expected_depth for row in rows):
            issues.append(f"stability/{player_id}: result row depth mismatch")

    print(
        f"  stability: {len(results)} model result sets, "
        f"{replayable} replayable sets checked against the current contract"
    )
    return issues


def verify_sequence_candidate(manifest: dict[str, Any]) -> list[str]:
    """Verify the isolated long-sequence research contract and saved rows."""
    issues: list[str] = []
    config = manifest["panels"].get("continuation_sequence_candidate", {})
    if config.get("production_effect") != "none":
        issues.append("protocol_sequence: production_effect must remain none")
    results_path = resolve_manifest_path(config.get("results", ""))
    if not results_path.exists():
        return issues + [f"protocol_sequence: missing results file {results_path}"]

    expected_indices = [0, 12, 24, 36]
    results = load_json(results_path)
    for player_id, record in results.items():
        summary = record.get("summary", {})
        rows = record.get("results", [])
        if summary.get("stability_probe_version") != config.get("stability_probe_version"):
            issues.append(f"protocol_sequence/{player_id}: probe version mismatch")
        if summary.get("position_selection_policy") != config.get("selection_policy"):
            issues.append(f"protocol_sequence/{player_id}: selection policy mismatch")
        if summary.get("selected_position_indices") != expected_indices:
            issues.append(f"protocol_sequence/{player_id}: summary indices mismatch")
        if [row.get("position_idx") for row in rows] != expected_indices:
            issues.append(f"protocol_sequence/{player_id}: row indices mismatch")
        if int(summary.get("attempted_positions", 0)) != len(expected_indices):
            issues.append(f"protocol_sequence/{player_id}: incomplete positions")
        if int(summary.get("probe_plies", 0)) != int(config.get("probe_plies", 0)):
            issues.append(f"protocol_sequence/{player_id}: probe plies mismatch")
        expected_depth = int(config.get("stockfish_score_depth", 0))
        if int(summary.get("score_depth", 0)) != expected_depth:
            issues.append(f"protocol_sequence/{player_id}: summary depth mismatch")
        if any(int(row.get("score_depth", 0)) != expected_depth for row in rows):
            issues.append(f"protocol_sequence/{player_id}: result row depth mismatch")
        if int(summary.get("api_errors", 0)) != 0:
            issues.append(f"protocol_sequence/{player_id}: contains API errors")
        if any(
            row.get("conditional_retry_protocol_version")
            != config.get("conditional_retry_protocol_version")
            for row in rows
        ):
            issues.append(f"protocol_sequence/{player_id}: retry stamp mismatch")

    print(f"  protocol_sequence: {len(results)} model result sets")
    return issues


def verify_failure_transfer_screen(manifest: dict[str, Any]) -> list[str]:
    """Verify matrix leakage guards and any saved cross-model screen results."""
    issues: list[str] = []
    config = manifest["panels"].get("failure_transfer_screen", {})
    if config.get("production_effect") != "none":
        issues.append("failure_transfer: production_effect must remain none")
    expected_depth = int(config.get("stockfish_depth", 0))
    if expected_depth != 30:
        issues.append("failure_transfer: Stockfish depth must remain 30")
    matrix_path = resolve_manifest_path(config.get("matrix", ""))
    if not matrix_path.exists():
        return issues + [f"failure_transfer: missing matrix {matrix_path}"]
    matrix = load_json(matrix_path)
    metadata = matrix.get("metadata", {})
    if metadata.get("screen_version") != config.get("screen_version"):
        issues.append("failure_transfer: screen version mismatch")
    if int(metadata.get("selected_failure_count", 0)) != int(
        config.get("source_failure_count", 0)
    ):
        issues.append("failure_transfer: source failure count mismatch")
    if int(metadata.get("matched_control_count", 0)) != int(
        config.get("matched_control_count", 0)
    ):
        issues.append("failure_transfer: control count mismatch")

    failures = {
        row["candidate_id"]: row for row in matrix.get("selected_failures", [])
    }
    controls = {
        row["candidate_id"]: row for row in matrix.get("matched_controls", [])
    }
    for target_player, position_ids in matrix.get("test_matrix", {}).items():
        target_family = next(
            (
                family
                for family in ("luna", "terra", "sol")
                if target_player.startswith(f"gpt-5.6-{family}")
            ),
            None,
        )
        if target_family is None:
            issues.append(f"failure_transfer: unexpected target {target_player}")
            continue
        if len(position_ids) != int(config.get("positions_per_target", 0)):
            issues.append(f"failure_transfer/{target_player}: position count mismatch")
        for position_id in position_ids:
            source = failures.get(position_id) or controls.get(position_id)
            if source is None:
                issues.append(f"failure_transfer/{target_player}: unknown {position_id}")
            elif source.get("base_model") == target_family:
                issues.append(f"failure_transfer/{target_player}: source-family leakage")

        target_path_value = metadata.get("target_files", {}).get(target_family, "")
        target_path = resolve_manifest_path(target_path_value)
        if not target_path.exists():
            issues.append(f"failure_transfer/{target_player}: missing target file")
            continue
        target_data = load_json(target_path)
        if int(target_data.get("metadata", {}).get("stockfish_setup_depth", 0)) != expected_depth:
            issues.append(f"failure_transfer/{target_player}: target metadata depth mismatch")
        if any(
            int(row.get("stockfish_setup_depth", 0)) != expected_depth
            for row in target_data.get("positions", [])
        ):
            issues.append(f"failure_transfer/{target_player}: target row depth mismatch")

    results_path = resolve_manifest_path(config.get("results", ""))
    if results_path.exists():
        results = load_json(results_path)
        if set(results) != set(matrix.get("test_matrix", {})):
            issues.append("failure_transfer: incomplete or unexpected result set")
        for player_id, record in results.items():
            expected = matrix.get("test_matrix", {}).get(player_id)
            rows = record.get("results", [])
            if expected is None:
                issues.append(f"failure_transfer: unexpected result player {player_id}")
                continue
            if [row.get("position_id") for row in rows] != expected:
                issues.append(f"failure_transfer/{player_id}: result IDs mismatch")
            if int(record.get("summary", {}).get("positions_skipped", 0)) != 0:
                issues.append(f"failure_transfer/{player_id}: skipped positions")
            if int(record.get("summary", {}).get("stockfish_depth", 0)) != expected_depth:
                issues.append(f"failure_transfer/{player_id}: summary depth mismatch")
            if any(int(row.get("stockfish_depth", 0)) != expected_depth for row in rows):
                issues.append(f"failure_transfer/{player_id}: result row depth mismatch")
            if any(
                row.get("conditional_retry_protocol_version")
                != config.get("conditional_retry_protocol_version")
                for row in rows
            ):
                issues.append(f"failure_transfer/{player_id}: retry stamp mismatch")
        print(f"  failure_transfer: {len(results)} model result sets")
    else:
        print("  failure_transfer: frozen matrix, no results yet")
    return issues


def verify_failure_transfer_shortlist(manifest: dict[str, Any]) -> list[str]:
    """Verify that the post-screen shortlist stays isolated and depth-30 stamped."""
    issues: list[str] = []
    config = manifest["panels"].get("failure_transfer_positive_shortlist", {})
    if config.get("production_effect") != "none":
        issues.append("failure_transfer_shortlist: production_effect must remain none")
    if config.get("required_next_gate") != "held-out-non-gpt56-model-families":
        issues.append("failure_transfer_shortlist: held-out gate mismatch")
    expected_depth = int(config.get("stockfish_depth", 0))
    if expected_depth != 30:
        issues.append("failure_transfer_shortlist: Stockfish depth must remain 30")
    shortlist_path = resolve_manifest_path(config.get("positions", ""))
    if not shortlist_path.exists():
        return issues + [f"failure_transfer_shortlist: missing file {shortlist_path}"]

    shortlist = load_json(shortlist_path)
    metadata = shortlist.get("metadata", {})
    positions = shortlist.get("positions", [])
    expected_count = int(config.get("failure_position_count", 0)) + int(
        config.get("matched_control_count", 0)
    )
    if len(positions) != expected_count:
        issues.append("failure_transfer_shortlist: position count mismatch")
    if int(metadata.get("stockfish_depth", 0)) != expected_depth:
        issues.append("failure_transfer_shortlist: metadata depth mismatch")
    if any(int(row.get("stockfish_setup_depth", 0)) != expected_depth for row in positions):
        issues.append("failure_transfer_shortlist: row depth mismatch")
    if metadata.get("selection_uses_gpt56_transfer_results") is not True:
        issues.append("failure_transfer_shortlist: selection provenance missing")
    return issues


def main() -> None:
    manifest = load_json(BASE / "benchmark_manifest.json")
    print("POSITION BENCHMARK LAYOUT VERIFICATION")
    issues: list[str] = []
    issues.extend(verify_automatic_acquisition(manifest))
    issues.extend(verify_prospective_stability_cap(manifest))
    issues.extend(verify_panel("core", manifest["panels"]["core"], require_complete_results=True))
    issues.extend(
        verify_panel("game_like", manifest["panels"]["game_like"], require_complete_results=True)
    )
    issues.extend(
        verify_panel("blunder", manifest["panels"]["blunder"], require_complete_results=False)
    )
    issues.extend(verify_legacy_mapping(manifest))
    issues.extend(verify_sequence_candidate(manifest))
    issues.extend(verify_failure_transfer_screen(manifest))
    issues.extend(verify_failure_transfer_shortlist(manifest))

    issues.extend(verify_stability_results(manifest))

    if issues:
        print(f"FAILED: {len(issues)} issue(s)")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
