#!/usr/bin/env python3
"""Create the stable-ID position benchmark layout from the legacy artifacts."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from position_benchmark.layout import (  # noqa: E402
    BLUNDER_POSITIONS_PATH,
    BLUNDER_RESULTS_PATH,
    CORE_POSITIONS_PATH,
    CORE_RESULTS_PATH,
    GAME_LIKE_POSITIONS_PATH,
    GAME_LIKE_RESULTS_PATH,
    LEGACY_COMBINED_POSITIONS_PATH,
    STABILITY_RESULTS_PATH,
    RESULT_SCHEMA_VERSION,
)
from position_benchmark.predictions import (  # noqa: E402
    CURRENT_BENCHMARK_VERSION,
    result_row_is_current,
)
from position_benchmark.retry_protocol import attach_conditional_retry_summary  # noqa: E402
from position_benchmark.token_accounting import refresh_result_token_usage  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OLD_COMBINED_POSITIONS = LEGACY_COMBINED_POSITIONS_PATH
OLD_CORE_RESULTS = ROOT / "position_benchmark" / "legacy" / "combined_results_75.json"
OLD_GAME_LIKE_POSITIONS = GAME_LIKE_POSITIONS_PATH
OLD_GAME_LIKE_RESULTS = ROOT / "position_benchmark" / "legacy" / "game_like_results_pre_stable_ids.json"
OLD_STABILITY_RESULTS = STABILITY_RESULTS_PATH


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def panel_position(
    source: dict[str, Any],
    *,
    panel: str,
    local_index: int,
    legacy_index: int | None,
) -> dict[str, Any]:
    position = deepcopy(source)
    position["position_id"] = f"{panel}-{local_index + 1:03d}"
    position["panel"] = panel
    position["panel_index"] = local_index
    if legacy_index is not None:
        position["legacy_position_idx"] = legacy_index
    return position


def build_panel(
    source_positions: list[dict[str, Any]],
    selected: list[tuple[int | None, dict[str, Any]]],
    *,
    panel: str,
    description: str,
    required: bool,
) -> dict[str, Any]:
    del source_positions
    positions = [
        panel_position(
            position,
            panel=panel,
            local_index=local_index,
            legacy_index=legacy_index,
        )
        for local_index, (legacy_index, position) in enumerate(selected)
    ]
    return {
        "metadata": {
            "panel": panel,
            "description": description,
            "required": required,
            "position_count": len(positions),
            "index_space": "panel-local",
            "stable_key": "position_id",
            "benchmark_version": CURRENT_BENCHMARK_VERSION,
        },
        "positions": positions,
    }


def summarize_rows(player_id: str, rows: list[dict[str, Any]], panel: str) -> dict[str, Any]:
    total = len(rows)
    legal = [row for row in rows if row.get("is_legal", True)]
    cpls = [float(row.get("cpl", 0.0)) for row in rows]
    legal_cpls = [float(row.get("cpl", 0.0)) for row in legal]
    sorted_cpls = sorted(cpls)
    if not sorted_cpls:
        median_cpl = 10000.0
    elif total % 2:
        median_cpl = sorted_cpls[total // 2]
    else:
        median_cpl = (sorted_cpls[total // 2 - 1] + sorted_cpls[total // 2]) / 2.0

    current_rows = sum(1 for row in rows if result_row_is_current(row))
    summary: dict[str, Any] = {
        "player_id": player_id,
        "panel": panel,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "total_positions": total,
        "positions_attempted": total,
        "positions_skipped": 0,
        "legal_moves": len(legal),
        "legal_pct": 100.0 * len(legal) / total if total else 0.0,
        "best_moves": sum(1 for row in rows if row.get("is_best", False)),
        "best_pct": (
            100.0 * sum(1 for row in rows if row.get("is_best", False)) / total
            if total
            else 0.0
        ),
        "avg_cpl": sum(cpls) / total if total else 10000.0,
        "avg_cpl_legal": sum(legal_cpls) / len(legal_cpls) if legal_cpls else 10000.0,
        "median_cpl": median_cpl,
        "current_result_rows": current_rows,
        "legacy_or_mixed_result_rows": total - current_rows,
    }
    if panel == "optional-blunder":
        avoided = sum(1 for row in rows if row.get("avoided_blunder", True))
        summary["avoided_blunders"] = avoided
        summary["avoided_pct"] = 100.0 * avoided / total if total else 0.0
    if rows and current_rows == total:
        summary["position_benchmark_version"] = CURRENT_BENCHMARK_VERSION
        summary["prompt_history_replay"] = True
        summary["stockfish_depth"] = min(
            int(row.get("stockfish_depth", row.get("reevaluated_depth", 0))) for row in rows
        )
    else:
        summary["position_benchmark_version"] = "mixed-or-legacy"
        summary["prompt_history_replay"] = False
    attach_conditional_retry_summary(summary, rows)
    return summary


def migrate_row(
    row: dict[str, Any],
    *,
    position: dict[str, Any],
    local_index: int,
    legacy_index: int | None,
) -> dict[str, Any]:
    migrated = deepcopy(row)
    migrated["position_id"] = position["position_id"]
    migrated["result_schema_version"] = RESULT_SCHEMA_VERSION
    migrated["position_idx"] = local_index
    migrated["panel"] = position["panel"]
    if legacy_index is not None:
        migrated["legacy_position_idx"] = legacy_index
    return migrated


def migrate_results(
    source_results: dict[str, Any],
    panel_data: dict[str, Any],
    *,
    source_index_for_local: list[int],
    panel: str,
) -> dict[str, Any]:
    positions = panel_data["positions"]
    source_to_local = {source_index: local for local, source_index in enumerate(source_index_for_local)}
    migrated: dict[str, Any] = {}
    skipped_mismatched_fens = 0
    for player_id, player_data in source_results.items():
        rows: list[dict[str, Any]] = []
        for row in player_data.get("results", []):
            source_index = row.get("position_idx")
            if source_index not in source_to_local:
                continue
            local_index = source_to_local[source_index]
            position = positions[local_index]
            if row.get("fen") and row["fen"] != position.get("fen"):
                skipped_mismatched_fens += 1
                continue
            rows.append(
                migrate_row(
                    row,
                    position=position,
                    local_index=local_index,
                    legacy_index=source_index,
                )
            )
        if not rows:
            continue
        rows.sort(key=lambda item: item["position_idx"])
        summary = summarize_rows(player_id, rows, panel)
        summary["positions_attempted"] = len(source_index_for_local)
        summary["positions_skipped"] = len(source_index_for_local) - len(rows)
        if summary["positions_skipped"]:
            summary["migration_excluded_rows"] = summary["positions_skipped"]
        old_summary = player_data.get("summary") or {}
        for key in (
            "note",
            "subagent_reasoning_effort",
            "run_estimated_api_cost",
            "run_actual_api_cost",
            "run_prompt_tokens",
            "run_completion_tokens",
        ):
            if key in old_summary:
                summary[key] = deepcopy(old_summary[key])
        migrated[player_id] = {
            "summary": summary,
            "results": rows,
        }
        # The source may contain several panels. Copying its whole-file totals
        # into each split panel makes every panel claim the same spend.
        refresh_result_token_usage(migrated[player_id])
    if skipped_mismatched_fens:
        print(f"Skipped {skipped_mismatched_fens} FEN-mismatched rows from {panel}")
    return migrated


def verify_fens(results: dict[str, Any], panel_data: dict[str, Any]) -> None:
    positions = panel_data["positions"]
    for player_id, player_data in results.items():
        seen: set[str] = set()
        for row in player_data.get("results", []):
            index = row["position_idx"]
            position = positions[index]
            if row["position_id"] != position["position_id"]:
                raise ValueError(f"{player_id}: position_id mismatch at {index}")
            if row.get("fen") and row["fen"] != position.get("fen"):
                raise ValueError(f"{player_id}: FEN mismatch at {row['position_id']}")
            if row["position_id"] in seen:
                raise ValueError(f"{player_id}: duplicate {row['position_id']}")
            seen.add(row["position_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-positions", type=Path, default=OLD_COMBINED_POSITIONS)
    parser.add_argument("--core-results", type=Path, default=OLD_CORE_RESULTS)
    parser.add_argument("--game-like-positions", type=Path, default=OLD_GAME_LIKE_POSITIONS)
    parser.add_argument("--game-like-results", type=Path, default=OLD_GAME_LIKE_RESULTS)
    parser.add_argument("--stability-results", type=Path, default=OLD_STABILITY_RESULTS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite organized outputs; unsafe after new panel results have been added",
    )
    args = parser.parse_args()

    organized_outputs = (
        CORE_POSITIONS_PATH,
        BLUNDER_POSITIONS_PATH,
        GAME_LIKE_POSITIONS_PATH,
        CORE_RESULTS_PATH,
        BLUNDER_RESULTS_PATH,
        GAME_LIKE_RESULTS_PATH,
    )
    existing_outputs = [path for path in organized_outputs if path.exists()]
    if existing_outputs and not args.force:
        joined = ", ".join(str(path) for path in existing_outputs)
        raise SystemExit(
            "Refusing to overwrite organized benchmark artifacts. "
            f"Existing: {joined}. Pass --force only for a deliberate remigration."
        )

    combined_data = load_json(args.combined_positions)
    combined_positions = combined_data["positions"]
    equal_selected = [
        (index, position)
        for index, position in enumerate(combined_positions)
        if position.get("type") == "equal"
    ]
    blunder_selected = [
        (index, position)
        for index, position in enumerate(combined_positions)
        if position.get("type") == "blunder"
    ]
    if len(equal_selected) != 50 or len(blunder_selected) != 25:
        raise ValueError("Legacy combined panel must contain exactly 50 equal and 25 blunder rows")

    core_panel = build_panel(
        combined_positions,
        equal_selected,
        panel="core-equal",
        description="Required production core: 50 history-replayed equal positions",
        required=True,
    )
    blunder_panel = build_panel(
        combined_positions,
        blunder_selected,
        panel="optional-blunder",
        description="Optional historical downside panel; not part of the production core",
        required=False,
    )

    game_like_source = load_json(args.game_like_positions)
    game_like_source_positions = game_like_source["positions"]
    if len(game_like_source_positions) != 48:
        raise ValueError("Game-like panel must contain exactly 48 rows")
    game_like_panel = build_panel(
        game_like_source_positions,
        [(None, position) for position in game_like_source_positions],
        panel="game-like",
        description="Optional 48-position non-opening downside panel",
        required=False,
    )
    game_like_panel["metadata"]["categories"] = deepcopy(
        game_like_source.get("metadata", {}).get("targets", {})
    )

    source_results = load_json(args.core_results)
    core_results = migrate_results(
        source_results,
        core_panel,
        source_index_for_local=[index for index, _ in equal_selected],
        panel="core-equal",
    )
    blunder_results = migrate_results(
        source_results,
        blunder_panel,
        source_index_for_local=[index for index, _ in blunder_selected],
        panel="optional-blunder",
    )
    game_like_source_results = load_json(args.game_like_results)
    game_like_results = migrate_results(
        game_like_source_results,
        game_like_panel,
        source_index_for_local=list(range(48)),
        panel="game-like",
    )

    verify_fens(core_results, core_panel)
    verify_fens(blunder_results, blunder_panel)
    verify_fens(game_like_results, game_like_panel)

    write_json(CORE_POSITIONS_PATH, core_panel)
    write_json(BLUNDER_POSITIONS_PATH, blunder_panel)
    write_json(GAME_LIKE_POSITIONS_PATH, game_like_panel)
    write_json(CORE_RESULTS_PATH, core_results)
    write_json(BLUNDER_RESULTS_PATH, blunder_results)
    write_json(GAME_LIKE_RESULTS_PATH, game_like_results)
    write_json(STABILITY_RESULTS_PATH, load_json(args.stability_results))

    legacy_data = deepcopy(combined_data)
    legacy_data["metadata"] = {
        **legacy_data.get("metadata", {}),
        "status": "legacy-index-registry",
        "active_production_input": False,
        "replacement": "position_benchmark/panels/core_equal_50.json",
    }
    write_json(LEGACY_COMBINED_POSITIONS_PATH, legacy_data)

    print(f"Core panel: {len(core_panel['positions'])} positions, {len(core_results)} models")
    print(f"Blunder panel: {len(blunder_panel['positions'])} positions, {len(blunder_results)} models")
    print(f"Game-like panel: {len(game_like_panel['positions'])} positions, {len(game_like_results)} models")


if __name__ == "__main__":
    main()
