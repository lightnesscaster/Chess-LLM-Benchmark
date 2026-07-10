#!/usr/bin/env python3
"""Verify the manifest, stable-ID panels, results, and legacy mappings."""

from __future__ import annotations

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


def main() -> None:
    manifest = load_json(BASE / "benchmark_manifest.json")
    print("POSITION BENCHMARK LAYOUT VERIFICATION")
    issues: list[str] = []
    issues.extend(verify_panel("core", manifest["panels"]["core"], require_complete_results=True))
    issues.extend(
        verify_panel("game_like", manifest["panels"]["game_like"], require_complete_results=True)
    )
    issues.extend(
        verify_panel("blunder", manifest["panels"]["blunder"], require_complete_results=False)
    )
    issues.extend(verify_legacy_mapping(manifest))

    stability_path = resolve_manifest_path(manifest["panels"]["continuation_stability"]["results"])
    if not stability_path.exists():
        issues.append(f"stability: missing results file {stability_path}")
    else:
        print(f"  stability: {len(load_json(stability_path))} model result sets")

    if issues:
        print(f"FAILED: {len(issues)} issue(s)")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
