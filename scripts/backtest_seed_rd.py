#!/usr/bin/env python3
"""Compare seed RDs without model calls or production writes.

Supply a private snapshot directory containing results_snapshot.json. --freeze
copies local benchmark/config inputs once; subsequent runs verify their hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from position_benchmark.predictions import predict_rating_from_model_data_with_supplement
from rating.reasoning_prediction import release_cohort_for_model_id
from rating.seed_rd_backtest import replay, summarize


CANDIDATES = (166, 250, 300, 350)
INPUTS = (
    "config/benchmark.yaml", "data/model_publish_dates.json",
    "position_benchmark/panels/core_equal_50.json",
    "position_benchmark/panels/game_like_48.json",
    "position_benchmark/panels/optional_blunder_25.json",
    "position_benchmark/results/core.json",
    "position_benchmark/results/game_like.json",
    "position_benchmark/results/optional_blunder.json",
    "position_benchmark/results/stability.json",
    "position_benchmark/results/stability_cap_shadow.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(directory: Path) -> None:
    """Save immutable local inputs; never refetch games or overwrite a snapshot."""
    if (directory / "inputs").exists() or (directory / "manifest.json").exists():
        raise ValueError("Snapshot already exists; omit --freeze to rerun")
    for relative in INPUTS:
        source, target = ROOT / relative, directory / "inputs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    paths = [directory / "results_snapshot.json"] + [directory / "inputs" / p for p in INPUTS]
    manifest = {str(p.relative_to(directory)): digest(p) for p in paths}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_seeds(directory: Path) -> tuple[dict, dict, dict, dict]:
    """Load fixed means and prospective locks from verified frozen artifacts."""
    def read(relative: str) -> Any:
        return json.loads((directory / "inputs" / relative).read_text())

    metadata = read("data/model_publish_dates.json")
    model_ids = {p: m["model_id"] for p, m in metadata.items() if m.get("model_id")}
    locks = read("position_benchmark/results/stability_cap_shadow.json")["entries"]
    prospective = {}
    for player, entry in locks.items():
        eligibility = entry.get("eligibility", {})
        if (entry.get("prediction_locked") is True
                and eligibility.get("prospective_holdout") is True
                and eligibility.get("game_snapshot", {}).get("games_played") == 0):
            prospective[player] = {"rating": entry["production_prediction"],
                                   "available_at": entry["recorded_at"],
                                   "family": entry["family"], "lab": entry["lab"]}
    core = read("position_benchmark/results/core.json")
    game_like = read("position_benchmark/results/game_like.json")
    blunder = read("position_benchmark/results/optional_blunder.json")
    stability = read("position_benchmark/results/stability.json")
    positions = read("position_benchmark/panels/core_equal_50.json")["positions"]
    game_positions = read("position_benchmark/panels/game_like_48.json")["positions"]
    blunder_positions = read("position_benchmark/panels/optional_blunder_25.json")["positions"]
    retrospective = {}
    for player, data in core.items():
        rating = predict_rating_from_model_data_with_supplement(
            data, positions, blunder_model_data=blunder.get(player),
            blunder_positions=blunder_positions, game_like_model_data=game_like.get(player),
            game_like_positions=game_positions, stability_probe_model_data=stability.get(player),
            require_ready=True,
        )
        if rating is not None:
            model_id = model_ids.get(player, player)
            # Prefer the shadow registry's model-family grouping when available.
            retrospective[player] = {
                "rating": rating, "available_at": "1970-01-01T00:00:00Z",
                "family": locks.get(player, {}).get("family", release_cohort_for_model_id(model_id)),
                "lab": locks.get(player, {}).get("lab", model_id.split("/")[0]),
            }
    config = yaml.safe_load((directory / "inputs/config/benchmark.yaml").read_text())
    return prospective, retrospective, config, model_ids


def evaluate(directory: Path) -> dict:
    """Verify snapshot integrity and return all predeclared candidate metrics."""
    manifest = json.loads((directory / "manifest.json").read_text())
    expected_keys = {"results_snapshot.json"} | {f"inputs/{p}" for p in INPUTS}
    if set(manifest) != expected_keys:
        raise ValueError("Unexpected or missing snapshot manifest entries")
    for relative, expected in manifest.items():
        if digest(directory / relative) != expected:
            raise ValueError(f"Snapshot checksum mismatch: {relative}")
    results = json.loads((directory / "results_snapshot.json").read_text())
    prospective, retrospective, config, model_ids = load_seeds(directory)
    report = {
        "input_hashes": manifest,
        "implementation_hashes": {p: digest(ROOT / p) for p in (
            "rating/seed_rd_backtest.py", "scripts/backtest_seed_rd.py",
            "rating/glicko2.py", "rating/anchor_history.py", "rating/rating_store.py",
            "rating/reasoning_prediction.py", "position_benchmark/predictions.py",
        )},
        "candidates": CANDIDATES, "runs": {},
    }
    for label, seeds in (("prospective", prospective), ("retrospective", retrospective)):
        report["runs"][label] = {}
        for rd in CANDIDATES:
            result = replay(results, seeds, config["engines"], model_ids, rd)
            rows = result["observations"]
            no_argon = [r for r in rows if "argon" not in r["player_id"]]
            metrics = {
                "all": summarize(rows),
                "first_ten": summarize([r for r in rows if r["prior_games"] < 10]),
                "without_argon": summarize(no_argon),
                "without_argon_first_ten": summarize([r for r in no_argon if r["prior_games"] < 10]),
                "by_player": {p: summarize([r for r in rows if r["player_id"] == p])
                              for p in sorted({r["player_id"] for r in rows})},
                "rejected": result["rejected"],
                "argon_medium_replay": result["ratings"].get("gemini-4-argon (medium)"),
            }
            report["runs"][label][str(rd)] = metrics
            # Private diagnostics preserve paired forecasts for further audit.
            (directory / f"{label}_rd{rd}.json").write_text(json.dumps(result, indent=2) + "\n")
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run the frozen, read-only RD comparison and print aggregate metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.snapshot_dir)
    report = evaluate(args.snapshot_dir)
    for label, candidates in report["runs"].items():
        print(label)
        for rd, metrics in candidates.items():
            print(rd, json.dumps({k: metrics[k] for k in
                                 ("all", "first_ten", "without_argon", "without_argon_first_ten")}))


if __name__ == "__main__":
    main()
