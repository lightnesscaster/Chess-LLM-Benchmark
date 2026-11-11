"""Prepare and explicitly publish an audited, atomic position-scoring correction.

``prepare`` only reads production and writes a durable local audit. ``apply`` first
backs that audit up to Firestore, then guards the complete input snapshot in one
transaction before publishing panels, replayed ratings and their shared revision.
Neither importing this module nor preparing a plan publishes production changes.
"""

from __future__ import annotations

import base64
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from time import time

from game.models import GameResult
from position_benchmark.layout import (
    BLUNDER_POSITIONS_PATH, BLUNDER_RESULTS_PATH,
    CORE_POSITIONS_PATH, CORE_RESULTS_PATH, GAME_LIKE_POSITIONS_PATH,
    GAME_LIKE_RESULTS_PATH, STABILITY_RESULTS_PATH,
)
from position_benchmark.predictions import (
    benchmark_result_readiness, resolve_result_position_index, stability_probe_readiness,
)
from rating.benchmark_recalculation import _calculate, state_reference
from rating.glicko2 import PlayerRating
from rating.rating_store import (
    BENCHMARK_CALIBRATION_VERSION, BENCHMARK_SEED_RD,
    PositionBenchmarkInputs, benchmark_predictions_from_inputs,
)

BACKUP_COLLECTION = "position_scoring_backups"
AUDIT_COLLECTION = "position_scoring_migrations"
MAX_WRITES = 499
MAX_BYTES = 8 * 1024 * 1024
MAX_DOCUMENT_BYTES = 900 * 1024
SCORING_FIELDS = frozenset({
    "summary", "best_move", "best_move_san", "eval_before", "eval_best", "eval_model",
    "cpl", "is_best", "move_rank", "top3", "top5", "reciprocal_rank", "preceding_opponent_cpl",
    "rescoring", "rank_metadata_source",
})
STATISTICS = ("games_played", "wins", "losses", "draws")


class ConcurrentMigration(RuntimeError):
    """Production changed since preparation; prepare a fresh plan without model calls."""


def _encode(value):
    if isinstance(value, datetime):
        # Firestore timestamps can carry nanoseconds beyond datetime precision.
        return {"__scoring_timestamp__": value.isoformat(),
                "nanosecond": getattr(value, "nanosecond", value.microsecond * 1000)}
    if isinstance(value, bytes):
        return {"__scoring_bytes__": base64.b64encode(value).decode()}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value):
    if isinstance(value, dict):
        if "__scoring_timestamp__" in value:
            from google.api_core.datetime_helpers import DatetimeWithNanoseconds
            parsed = datetime.fromisoformat(value["__scoring_timestamp__"])
            return DatetimeWithNanoseconds(
                parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second,
                nanosecond=value["nanosecond"], tzinfo=parsed.tzinfo,
            )
        if "__scoring_bytes__" in value:
            return base64.b64decode(value["__scoring_bytes__"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _json(value) -> str:
    return json.dumps(_encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def evidence_fingerprint(record: dict) -> str:
    """Hash response, prompt/history, usage and game evidence, excluding derived scores."""
    def evidence(value):
        if isinstance(value, dict):
            return {key: evidence(item) for key, item in value.items()
                    if key not in SCORING_FIELDS and not key.startswith("scoring_")}
        if isinstance(value, list):
            return [evidence(item) for item in value]
        return value
    return _hash(evidence(record))



def _same_evidence(source: dict, candidate: dict, panel: str) -> bool:
    """Permit registry relabeling only after resolving both ready static panels.

    Legacy combined-registry indices and canonical per-panel indices can label
    the same saved response. FEN and current prompt/depth readiness must agree
    before those registry labels can be omitted from the evidence comparison.
    """
    if evidence_fingerprint(source) == evidence_fingerprint(candidate):
        return True
    if panel not in {"core", "game_like", "blunder"} or not _ready(source, panel) or not _ready(candidate, panel):
        return False
    positions = _positions(panel)
    registry_fields = {"position_id", "position_idx", "legacy_position_idx", "panel", "result_schema_version"}

    def normalized(record):
        aligned = {}
        for row in record.get("results", []):
            index = resolve_result_position_index(row, positions)
            if index is None or row.get("fen") != positions[index].get("fen") or index in aligned:
                return None
            aligned[index] = {key: copy.deepcopy(value) for key, value in row.items() if key not in registry_fields}
        result = {key: copy.deepcopy(value) for key, value in record.items() if key != "results"}
        result["results"] = [{"resolved_position": index, "response": aligned[index]} for index in sorted(aligned)]
        return result

    normalized_source, normalized_candidate = normalized(source), normalized(candidate)
    return (normalized_source is not None and normalized_candidate is not None
            and evidence_fingerprint(normalized_source) == evidence_fingerprint(normalized_candidate))


def _snapshot_collection(db, collection: str, transaction=None) -> dict:
    return {doc.id: _encode(doc.to_dict())
            for doc in db.collection(collection).stream(transaction=transaction)}


def _positions(panel: str) -> list:
    path = {"core": CORE_POSITIONS_PATH, "game_like": GAME_LIKE_POSITIONS_PATH,
            "blunder": BLUNDER_POSITIONS_PATH}[panel]
    return json.loads(path.read_text())["positions"]


def _baseline_panels() -> tuple[dict, dict, dict]:
    return tuple(json.loads(path.read_text()) if path.exists() else {}
                 for path in (CORE_RESULTS_PATH, GAME_LIKE_RESULTS_PATH, STABILITY_RESULTS_PATH))


def _baseline_blunder() -> dict:
    return json.loads(BLUNDER_RESULTS_PATH.read_text()) if BLUNDER_RESULTS_PATH.exists() else {}


def _ready(record: dict, panel: str) -> bool:
    if panel == "stability":
        return stability_probe_readiness(record).is_ready
    positions = _positions(panel)
    return benchmark_result_readiness(
        record, positions, min_equal_positions=len(positions),
        position_type="blunder" if panel == "blunder" else "equal",
    ).is_ready


def _validate_corrected(player: str, panel: str, record: dict) -> None:
    from position_benchmark.evaluation import SCORING_VERSION
    from position_benchmark.scoring_provenance import validate_scoring_record
    from position_benchmark.sync import validate_panel
    if panel == "blunder":
        if record.get("summary", {}).get("player_id") != player or not _ready(record, panel):
            raise ValueError("Optional blunder panel is incomplete or has a mismatched player ID")
    else:
        validate_panel(player, panel, record)
    validate_scoring_record(record, required_version=SCORING_VERSION)
    # A legal selected reference move must never have a positive loss.
    for row in record.get("results", []):
        scores = (row.get("model_move_scores", []) + row.get("opponent_move_scores", [])) if panel == "stability" else [row]
        for score in scores:
            if score.get("is_best") and float(score.get("cpl", 0)) != 0:
                raise ValueError(f"Nonzero best-move CPL in {player}/{panel}")


def _core(record: dict) -> dict:
    return {key: copy.deepcopy(value) for key, value in record.items() if key != "supplements"}


def _inputs(core: dict, game_like: dict, stability: dict, blunder: dict | None = None) -> PositionBenchmarkInputs:
    return PositionBenchmarkInputs(
        core_results=core, core_positions=_positions("core"),
        blunder_results=blunder, blunder_positions=_positions("blunder") if blunder else None,
        game_like_results=game_like, game_like_positions=_positions("game_like"),
        stability_results=stability,
    )


def _merge_baseline(core: dict, game_like: dict, stability: dict, remote: dict) -> tuple[dict, dict, dict]:
    """Reproduce the legacy reader's fallback using only captured input dictionaries."""
    merged = copy.deepcopy(core)
    for player, record in remote.items():
        if _ready(record, "core"):
            merged[player] = copy.deepcopy(record)
        elif player in merged:
            merged[player]["supplements"] = copy.deepcopy(record.get("supplements", {}))
    return merged, copy.deepcopy(game_like), copy.deepcopy(stability)


def _validate_ratings(before: dict, after: dict) -> None:
    for player, data in after.items():
        rating = PlayerRating.from_dict(data)
        if rating.player_id != player or not all(math.isfinite(value) for value in (
            rating.rating, rating.rating_deviation, rating.volatility, rating.games_rd,
        )):
            raise ValueError(f"Invalid recalculated rating: {player}")
        if player in before and any(data.get(field, 0) != before[player].get(field, 0) for field in STATISTICS):
            raise ValueError(f"Game statistics changed for {player}; production snapshot is inconsistent")


def _writes(plan: dict) -> list[tuple[str, str, dict]]:
    receipt = plan["receipt"]
    state = {**plan["snapshot"]["state"], "generation": receipt["generation"],
             "applied_generation": receipt["generation"], "receipt": receipt,
             "scoring_version": receipt["scoring_version"], "scoring_profile": plan["scoring_profile"]}
    if "calibration_version" in receipt:
        state["calibration_version"] = receipt["calibration_version"]
    writes = [("benchmark_results", player, plan["benchmark_after"][player]) for player in plan["changed_models"]]
    writes.extend(("ratings", player, data) for player, data in sorted(plan["ratings_after"].items()))
    writes.extend([
        ("benchmark_rating_updates", "current", state),
        (AUDIT_COLLECTION, plan["migration_id"], {**receipt, "backup_id": plan["migration_id"]}),
    ])
    return writes


def _validate_budget(plan: dict) -> None:
    writes = _writes(plan)
    sizes = [len(_json(data).encode()) + len(collection) + len(player) + 1024
             for collection, player, data in writes]
    if len(writes) > MAX_WRITES or sum(sizes) > MAX_BYTES or max(sizes, default=0) > MAX_DOCUMENT_BYTES:
        raise ValueError("Scoring migration exceeds conservative atomic Firestore write/byte budget")


def _plan_hash(plan: dict) -> str:
    body = {key: value for key, value in plan.items() if key not in {"plan_hash", "backup_path"}}
    body = copy.deepcopy(body)
    body.get("receipt", {}).pop("plan_hash", None)
    return _hash(body)


def _durable_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def prepare(db, candidate_core: dict, candidate_game_like: dict, candidate_stability: dict,
            backup_dir: Path, *, candidate_blunder: dict | None = None) -> dict:
    """Build a reviewable, durable plan from saved corrected records; perform no writes to db."""
    snapshot = {name: _snapshot_collection(db, name) for name in ("benchmark_results", "ratings", "results")}
    snapshot["state"] = _encode(state_reference(db).get().to_dict() or {})
    baseline = _baseline_panels()
    baseline_blunder = _baseline_blunder()
    before_inputs = _inputs(*_merge_baseline(*baseline, snapshot["benchmark_results"]),
                            blunder=baseline_blunder)
    predictions_before = benchmark_predictions_from_inputs(before_inputs)
    after = copy.deepcopy(snapshot["benchmark_results"])
    effective_blunder = {**copy.deepcopy(baseline_blunder), **copy.deepcopy(candidate_blunder or {})}
    candidates = {"core": candidate_core, "game_like": candidate_game_like,
                  "stability": candidate_stability, "blunder": effective_blunder}
    provenance = None
    baseline_by_panel = {**dict(zip(("core", "game_like", "stability"), baseline)),
                         "blunder": baseline_blunder}
    corrected_counts = {}
    for panel, records in candidates.items():
        corrected_counts[panel] = 0
        for player, record in records.items():
            summary = record.get("summary", {})
            if not summary.get("scoring_version"):
                continue  # Legacy prompt rows are retained as historical evidence.
            _validate_corrected(player, panel, record)
            identity = (summary["scoring_version"], summary.get("scoring_profile"))
            if not identity[1] or (provenance is not None and identity != provenance):
                raise ValueError("Corrected panels have inconsistent scoring provenance")
            provenance = identity
            existing = after.get(player, {})
            previous = _core(existing) if panel == "core" else existing.get("supplements", {}).get(panel, {})
            source = previous if _ready(previous, panel) else baseline_by_panel[panel].get(player, {})
            if not _ready(source, panel):
                raise ValueError(f"No saved source evidence for {player}/{panel}")
            if not _same_evidence(_core(source), _core(record), panel):
                raise ValueError(f"Raw response evidence differs for {player}/{panel}")
            if panel == "core":
                after[player] = {**_core(record), **({"supplements": copy.deepcopy(existing["supplements"])}
                                                   if "supplements" in existing else {})}
            else:
                after.setdefault(player, {}).setdefault("supplements", {})[panel] = copy.deepcopy(record)
            corrected_counts[panel] += 1
    if provenance is None:
        raise ValueError("No corrected production panels supplied")
    # Construct candidate predictions without the loader's implicit local fallback.
    # Every eligible local model must be corrected, including cores absent in cloud.
    for panel, records in candidates.items():
        for player, rows in records.items():
            if _ready(rows, panel):
                summary = rows.get("summary", {})
                if (summary.get("scoring_version"), summary.get("scoring_profile")) != provenance:
                    raise ValueError(f"Ready candidate lacks corrected scoring provenance: {player}/{panel}")
    candidate_inputs_core = copy.deepcopy(candidate_core)
    candidate_inputs_core.update(copy.deepcopy(after))
    for player, record in candidate_inputs_core.items():
        panels = {"core": _core(record), **record.get("supplements", {})}
        for panel, rows in panels.items():
            if panel not in candidates or not _ready(rows, panel):
                continue
            summary = rows.get("summary", {})
            if (summary.get("scoring_version"), summary.get("scoring_profile")) != provenance:
                raise ValueError(f"Ready panel lacks corrected scoring provenance: {player}/{panel}")
            _validate_corrected(player, panel, rows)
    predictions_after = benchmark_predictions_from_inputs(_inputs(
        candidate_inputs_core, candidate_game_like, candidate_stability, blunder=effective_blunder,
    ))
    # Invalid saved games must stop publication, never silently disappear from replay.
    games = [GameResult.from_json(_decode(data)) for data in snapshot["results"].values()]
    ratings_after = {**copy.deepcopy(snapshot["ratings"]), **_encode(_calculate(games, predictions_after))}
    _validate_ratings(snapshot["ratings"], ratings_after)
    changed_models = sorted(player for player, record in after.items()
                            if record != snapshot["benchmark_results"].get(player))
    generation = max(snapshot["state"].get("generation", 0), snapshot["state"].get("applied_generation", 0)) + 1
    migration_id = "scoring-" + _hash({"snapshot": snapshot, "after": after, "ratings": ratings_after})[:24]
    receipt = {
        "status": "applied", "migration_id": migration_id, "generation": generation,
        "scoring_version": provenance[0], "scoring_profile_hash": _hash(provenance[1]),
        "models": len(ratings_after), "games": len(games), "changed_benchmark_models": len(changed_models),
        "corrected_panels": corrected_counts, "seed_rd": BENCHMARK_SEED_RD, "prepared_at": time(),
        "calibration_version": BENCHMARK_CALIBRATION_VERSION,
        "before_hash": _hash(snapshot["ratings"]), "after_hash": _hash(ratings_after),
        "benchmark_before_hash": _hash(snapshot["benchmark_results"]), "benchmark_after_hash": _hash(after),
        "results_hash": _hash(snapshot["results"]), "predictions_before_hash": _hash(predictions_before),
        "predictions_after_hash": _hash(predictions_after),
    }
    plan = {
        "schema_version": 1, "migration_id": migration_id, "snapshot": snapshot,
        "baseline_hashes": [_hash(panel) for panel in baseline],
        "baseline_blunder": baseline_blunder,
        "candidate_hashes": {key: _hash(value) for key, value in candidates.items()},
        "scoring_profile": provenance[1], "benchmark_after": after,
        "ratings_after": ratings_after, "predictions_before": predictions_before,
        "predictions_after": predictions_after, "changed_models": changed_models, "receipt": receipt,
    }
    plan["plan_hash"] = _plan_hash(plan)
    receipt["plan_hash"] = plan["plan_hash"]
    _validate_budget(plan)
    plan["backup_path"] = str((Path(backup_dir) / f"{migration_id}.json").resolve())
    _durable_write(Path(plan["backup_path"]), plan)
    return plan


def _backup_firestore(db, plan: dict) -> None:
    """Store the complete audit in bounded chunks before touching production collections."""
    data = _json(plan)
    chunks = [data[start:start + 200_000] for start in range(0, len(data), 200_000)]
    root = db.collection(BACKUP_COLLECTION).document(plan["migration_id"])
    entries = [(root.collection("chunks").document(f"{index:06d}"), {"data": chunk})
               for index, chunk in enumerate(chunks)]
    for start in range(0, len(entries), 20):
        batch = db.batch()
        for reference, value in entries[start:start + 20]:
            batch.set(reference, value)
        batch.commit()
    batch = db.batch()
    batch.set(root, {"plan_hash": plan["plan_hash"], "audit_hash": _hash(plan),
                     "chunks": len(chunks), "bytes": len(data.encode()), "status": "backed_up"})
    batch.commit()
    # Confirm durability and completeness before the production transaction.
    manifest = root.get().to_dict() or {}
    restored = "".join(doc.to_dict()["data"] for doc in sorted(root.collection("chunks").stream(), key=lambda doc: doc.id))
    if manifest.get("plan_hash") != plan["plan_hash"] or restored != data:
        raise RuntimeError("Firestore scoring backup verification failed")


def apply(db, plan: dict) -> dict:
    """Publish a prepared plan exactly once, rejecting any intervening production changes."""
    from firebase_admin import firestore
    if plan.get("schema_version") != 1 or _plan_hash(plan) != plan.get("plan_hash"):
        raise ValueError("Scoring migration plan hash mismatch")
    _validate_budget(plan)
    # A copied/transported plan is durably saved on this host as well.
    _durable_write(Path(plan["backup_path"]), plan)
    audit_ref = db.collection(AUDIT_COLLECTION).document(plan["migration_id"])
    already = audit_ref.get().to_dict() or {}
    if already:
        if already.get("plan_hash") != plan["plan_hash"]:
            raise ValueError("Migration receipt hash mismatch")
        return {key: value for key, value in already.items() if key != "backup_id"}
    _backup_firestore(db, plan)

    @firestore.transactional
    def commit(transaction):
        existing = audit_ref.get(transaction=transaction).to_dict() or {}
        if existing:
            if existing.get("plan_hash") != plan["plan_hash"]:
                raise ValueError("Migration receipt hash mismatch")
            return {key: value for key, value in existing.items() if key != "backup_id"}
        current = {name: _snapshot_collection(db, name, transaction)
                   for name in ("benchmark_results", "ratings", "results")}
        current["state"] = _encode(state_reference(db).get(transaction=transaction).to_dict() or {})
        if current != plan["snapshot"]:
            raise ConcurrentMigration("Production benchmark, ratings, games or revision changed; prepare again")
        for collection, player, value in _writes(plan):
            transaction.set(db.collection(collection).document(player), _decode(value))
        return plan["receipt"]

    return commit(db.transaction())
