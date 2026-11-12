"""Acquire and verify a configured player's production position benchmark."""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import subprocess
import sys
import time
from typing import Any
import urllib.request

from firebase_client import get_firestore_client
from game.checkpoints import CheckpointError, CheckpointStore
from model_metadata import load_publish_dates
from position_benchmark.acquisition import (
    AcquisitionPanel, AcquisitionPolicy, ROOT, load_acquisition_policy,
    load_acquisition_state, missing_panels, panel_readiness,
)
from position_benchmark.evaluation import SCORING_VERSION
from position_benchmark.panel_records import merge_player_record
from position_benchmark.predictions import collect_equal_position_coverage, resolve_result_position_index
from position_benchmark.run_benchmark import load_player_configs
from position_benchmark.scoring_provenance import validate_scoring_record
from position_benchmark.sync import WEB_URL, render_job_receipt, sync_panel
from rating.benchmark_recalculation import recalculate_pending, state_reference
from rating.glicko2 import PlayerRating
from rating.rating_store import RatingStore, benchmark_predictions_from_inputs, load_position_benchmark_inputs
from scripts.rescore_position_results import _EngineEvaluator, rescore_record
from utils import REASONING_SUFFIXES, REASONING_SUFFIX_RE


def firestore_panel(panel: AcquisitionPanel) -> str:
    return "stability" if panel.runner == "continuation" else panel.name


def player_record(panel: AcquisitionPanel, player: str) -> dict:
    if not panel.results_path.exists():
        return {}
    return json.loads(panel.results_path.read_text()).get(player, {})


def local_ready(panel: AcquisitionPanel, player: str) -> bool:
    return panel_readiness(panel, player).is_ready


def publish_saved(panel: AcquisitionPanel, player: str) -> None:
    sync_panel(player, firestore_panel(panel), player_record(panel, player))


def prepare_static_resume(panel: AcquisitionPanel, player: str, scorer: _EngineEvaluator) -> bool:
    """Retain current prompt rows and rescore their saved moves if the score epoch differs."""
    record = player_record(panel, player)
    rows = record.get("results", [])
    if not rows:
        return False
    positions = json.loads(panel.positions_path.read_text())["positions"]
    coverage = collect_equal_position_coverage(rows, positions, min_stockfish_depth=panel.score_depth)
    if coverage.unique_valid != len(rows) or (
        coverage.stale_rows or coverage.mismatched_fen or coverage.duplicate_rows
    ):
        return False
    # The static runner merges by panel-local index. Normalize legacy indices only
    # after coverage proves that all retained rows map to unique current positions.
    normalized = False
    for row in rows:
        index = resolve_result_position_index(row, positions)
        if row.get("position_idx") != index or row.get("position_id") != positions[index]["position_id"]:
            row.update(position_idx=index, position_id=positions[index]["position_id"])
            normalized = True
    try:
        validate_scoring_record(
            record, required_version=SCORING_VERSION, required_profile=scorer.scoring_profile,
        )
    except ValueError:
        record = rescore_record(record, positions, scorer, panel.score_depth)
        normalized = True
    if normalized:
        merge_player_record(panel.results_path, player, record)
    return True


def run_panel(panel: AcquisitionPanel, player: str, *, expected_profile: dict | None = None) -> None:
    """Run exactly one canonical panel using the configured player's API and effort."""
    args = ["--players", player, "--positions", str(panel.positions_path), "--output", str(panel.results_path)]
    # Use the same binary as both canonical runners. Compare its actual UCI/options/
    # NNUE profile before making model calls that production might decline to publish.
    with closing(_EngineEvaluator("stockfish", panel.score_depth, None)) as scorer:
        if expected_profile and scorer.scoring_profile != expected_profile:
            raise RuntimeError("Local Stockfish scoring profile differs from the required production profile")
        if panel.runner == "continuation":
            module = "scripts.run_stability_probe"
            args += ["--score-depth", str(panel.score_depth), "--limit", str(panel.position_count),
                     "--probe-plies", str(panel.probe_plies), "--random-seed", str(panel.random_seed)]
        else:
            module = "position_benchmark.run_benchmark"
            args += ["--depth", str(panel.score_depth)]
            if prepare_static_resume(panel, player, scorer):
                if local_ready(panel, player):
                    publish_saved(panel, player)
                    return
                args.append("--retry-missing")
    subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, check=True)


def published_family_results(db, player: str, *, collection_name: str = "benchmark_results") -> dict[str, dict]:
    """Fetch published records for one player and its explicit effort siblings."""
    base = REASONING_SUFFIX_RE.sub("", player).rstrip()
    family_ids = {base, player, *(f"{base} ({effort})" for effort in REASONING_SUFFIXES)}
    # RatingStore identifies effort siblings by the published model ID, which
    # can also link different display names for the same model.
    metadata = load_publish_dates()
    model_id = metadata.get(player, {}).get("model_id")
    if model_id:
        family_ids.update(pid for pid, info in metadata.items() if info.get("model_id") == model_id)
    collection = db.collection(collection_name)
    refs = [collection.document(pid) for pid in sorted(family_ids)]
    records = {doc.id: doc.to_dict() for doc in db.get_all(refs) if doc.exists}
    return {pid: records[pid] for pid in sorted(records)}


def firestore_snapshot(player: str, policy: AcquisitionPolicy) -> dict[str, Any]:
    """Read the published family measurements and the current rating revision."""
    db = get_firestore_client()
    records = published_family_results(db, player)
    inputs = load_position_benchmark_inputs(False, remote_results=records, published_only=True)
    if inputs is None:
        raise RuntimeError("Production position inputs are unavailable")
    state = load_acquisition_state(policy, inputs=inputs)
    measured = state.get(player, set())
    published = [panel.name for panel in policy.panels if panel.name in measured]
    revision = state_reference(db).get().to_dict() or {}
    ratings = published_family_results(db, player, collection_name="ratings")
    predictions = benchmark_predictions_from_inputs(inputs)
    persisted = ratings.get(player, {}).get("rating")
    expected_live = persisted
    if persisted is not None and ratings[player].get("games_played", 0) == 0:
        family = {pid: PlayerRating.from_dict({"player_id": pid, **data}) for pid, data in ratings.items()}
        expected_live = RatingStore(snapshot=(family, predictions)).get(player).rating
    receipt = revision.get("receipt") or {}
    return {"player": player, "policy": policy.version, "published": published,
            "pending": [p.name for p in missing_panels(player, policy, state)],
            "source": state.deferred_sources.get(player), "rating": persisted,
            "position_prediction": predictions.get(player), "expected_live_rating": expected_live,
            "generation": revision.get("generation"), "applied_generation": revision.get("applied_generation"),
            "scoring_version": revision.get("scoring_version"), "scoring_profile": revision.get("scoring_profile"),
            "receipt": {"status": receipt.get("status"), "generation": receipt.get("generation")}}


def production_snapshot(player: str, policy: AcquisitionPolicy) -> dict[str, Any]:
    try:
        get_firestore_client()
    except FileNotFoundError:
        code = (
            "import json; from position_benchmark.run_production import firestore_snapshot, load_acquisition_policy; "
            f"print('POSITION_STATUS '+json.dumps(firestore_snapshot({player!r}, load_acquisition_policy())),flush=True)"
        )
        result = render_job_receipt(code, "POSITION_STATUS")
        if result.get("player") != player or result.get("policy") != policy.version:
            raise RuntimeError("Production status does not match the requested player and acquisition policy")
        return result
    return firestore_snapshot(player, policy)


def resume_rating_update() -> None:
    """Finish a previously queued benchmark replay after an interrupted publication."""
    try:
        db = get_firestore_client()
    except FileNotFoundError:
        code = (
            "import json; from firebase_client import get_firestore_client; "
            "from rating.benchmark_recalculation import recalculate_pending; "
            "print('RATING_UPDATE '+json.dumps(recalculate_pending(get_firestore_client())),flush=True)"
        )
        render_job_receipt(code, "RATING_UPDATE")
    else:
        recalculate_pending(db)


def live_rating(player: str) -> int | None:
    with urllib.request.urlopen(WEB_URL + "/api/leaderboard", timeout=30) as response:
        return next((row["rating"] for row in json.load(response) if row["player_id"] == player), None)


def run(player: str, *, full_audit: bool = False) -> dict[str, Any]:
    """Resume a targeted acquisition and confirm publication, rating replay and live rating."""
    lock_id = hashlib.sha256(player.encode()).hexdigest()
    checkpoint = ROOT / "data" / "checkpoints" / "position-benchmark" / lock_id
    try:
        with CheckpointStore(checkpoint):
            return _run(player, full_audit=full_audit)
    except CheckpointError as exc:
        raise RuntimeError(f"A targeted position benchmark is already running for {player}") from exc


def _run(player: str, *, full_audit: bool) -> dict[str, Any]:
    config = load_player_configs().get(player)
    if not config or not config.get("model_name") or config.get("unavailable"):
        raise ValueError(f"Expected an available configured model player ID, got {player!r}")
    policy = load_acquisition_policy()
    snapshot = production_snapshot(player, policy)
    # A Render status job cannot see partial measurements saved only on this machine.
    finish_local_supplements = any(player_record(p, player) for p in policy.panels[1:])
    for panel in policy.panels:
        if panel.name in snapshot["published"]:
            continue
        current_local = local_ready(panel, player)
        if current_local and panel.runner == "positions" and snapshot.get("scoring_profile"):
            try:
                validate_scoring_record(
                    player_record(panel, player), required_version=SCORING_VERSION,
                    required_profile=snapshot["scoring_profile"],
                )
            except ValueError:
                current_local = False
        if current_local:
            publish_saved(panel, player)
        elif full_audit or panel.name in snapshot["pending"] or (panel.name != "core" and finish_local_supplements):
            if snapshot.get("scoring_version") not in {None, SCORING_VERSION}:
                raise RuntimeError("Local scorer version differs from the required production version")
            run_panel(panel, player, expected_profile=snapshot.get("scoring_profile"))
        else:
            continue
        snapshot = production_snapshot(player, policy)
        if panel.name not in snapshot["published"]:
            raise RuntimeError(f"{panel.name} publication could not be verified for {player}; inspect saved results before retrying")

    if (snapshot["generation"] or 0) > (snapshot["applied_generation"] or 0):
        resume_rating_update()
        snapshot = production_snapshot(player, policy)
    measured = [panel.name for panel in policy.panels if panel.name in snapshot["published"]]
    deferred = [panel.name for panel in policy.panels if panel.name not in measured]
    if deferred and (full_audit or "core" in deferred or not snapshot["source"] or set(deferred) & set(snapshot["pending"])):
        raise RuntimeError(f"Incomplete production measurements for {player}: {deferred}")
    generation = snapshot["generation"]
    receipt = snapshot["receipt"]
    if not generation or generation != snapshot["applied_generation"] or (
        receipt.get("status") != "applied" or receipt.get("generation") != generation
    ):
        raise RuntimeError(f"Automatic rating-update receipt is pending or unavailable for {player}")
    if snapshot["rating"] is None:
        raise RuntimeError(f"Missing production rating for {player}")
    # The site's workers cache the applied revision for ten seconds.
    for attempt in range(4):
        live = live_rating(player)
        if live == round(snapshot["expected_live_rating"]):
            return {"player": player, "policy": policy.version, "measured": measured, "deferred": deferred,
                    "source": snapshot["source"] if deferred else None, "rating": snapshot["rating"],
                    "position_prediction": snapshot["position_prediction"], "live_rating": live,
                    "applied_generation": generation}
        if attempt < 3:
            time.sleep(5)
    raise RuntimeError(f"Live leaderboard rating does not match the published rating for {player}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", required=True, help="Configured player ID including reasoning effort")
    parser.add_argument("--full-audit", action="store_true", help="Measure every missing panel, including policy deferrals")
    args = parser.parse_args()
    print(json.dumps(run(args.player, full_audit=args.full_audit), indent=2))


if __name__ == "__main__":
    main()
