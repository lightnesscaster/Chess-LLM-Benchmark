"""Apply newly published position priors without losing concurrent rated games."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import threading
from time import monotonic, time
from types import SimpleNamespace

from game.models import GameResult
from rating.glicko2 import PlayerRating
from rating.rating_store import (
    BENCHMARK_SEED_RD, benchmark_predictions_from_inputs, load_position_benchmark_inputs,
)

STATE_COLLECTION = "benchmark_rating_updates"
STATE_ID = "current"
ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
_revision_cache = (0.0, 0)
_revision_lock = threading.Lock()


class ConcurrentRatingUpdate(RuntimeError):
    """A newer game or benchmark requires a fresh calculation."""


def state_reference(db):
    return db.collection(STATE_COLLECTION).document(STATE_ID)


def queue_recalculation(transaction, db) -> None:
    """Increment the durable revision inside the panel's publication transaction."""
    ref = state_reference(db)
    snapshot = ref.get(transaction=transaction)
    state = snapshot.to_dict() or {}
    transaction.set(ref, {
        **state, "generation": state.get("generation", 0) + 1, "requested_at": time(),
    })


def _calculate(results: list[GameResult], predictions: dict) -> dict:
    """Run the existing rating replay in a local, isolated validation store."""
    from cli import recalculate_ratings

    with tempfile.TemporaryDirectory(prefix="benchmark-ratings-") as directory:
        output = Path(directory) / "ratings.json"
        args = SimpleNamespace(
            config=str(ROOT / "config/benchmark.yaml"), verbose=False,
            validation_output=output, validation_seed_rd=BENCHMARK_SEED_RD,
            validation_disable_benchmark_seeds=False, suppress_result_table=True,
        )
        status = asyncio.run(recalculate_ratings(
            args, results_snapshot=results, benchmark_predictions=predictions,
        ))
        if status:
            raise RuntimeError("Benchmark rating recalculation failed; published panels remain saved.")
        return json.loads(output.read_text())


def _fingerprint(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def recalculate_pending(db, *, max_attempts: int = 3) -> dict:
    """Complete a pending revision, retrying snapshots raced by normal publication."""
    from firebase_admin import firestore

    for _ in range(max_attempts):
        state = state_reference(db).get().to_dict() or {}
        generation = state.get("generation", 0)
        if generation <= state.get("applied_generation", 0):
            return state.get("receipt", {"status": "not_needed", "generation": generation})
        before = {doc.id: doc.to_dict() for doc in db.collection("ratings").stream()}
        records = {doc.id: doc.to_dict() for doc in db.collection("benchmark_results").stream()}
        results = []
        for doc in db.collection("results").stream():
            try:
                results.append(GameResult.from_json(doc.to_dict()))
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping invalid saved game %s during rating replay", doc.id)
        inputs = load_position_benchmark_inputs(False, remote_results=records)
        if inputs is None:
            raise RuntimeError("Production position inputs are unavailable; rating update is still pending.")
        predictions = benchmark_predictions_from_inputs(inputs)
        after = _calculate(results, predictions)
        # Preserve unrelated registered configurations with no game/benchmark yet.
        after = {**before, **after}
        if len(after) > 498:
            raise RuntimeError("Rating table exceeds the atomic publication write budget.")
        for pid, data in after.items():
            rating = PlayerRating.from_dict(data)
            if rating.player_id != pid or not all(math.isfinite(value) for value in (
                rating.rating, rating.rating_deviation, rating.volatility, rating.games_rd,
            )):
                raise RuntimeError(f"Invalid recalculated rating for {pid}")
            if pid in before and any(data.get(field, 0) != before[pid].get(field, 0)
                                     for field in ("games_played", "wins", "losses", "draws")):
                # Usually a new game landed while the input snapshot was being read.
                break
        else:
            refs = [db.collection("ratings").document(pid) for pid in sorted(after)]
            receipt = {
                "status": "applied", "generation": generation, "models": len(after),
                "games": len(results), "applied_at": time(), "seed_rd": BENCHMARK_SEED_RD,
                "before_hash": _fingerprint(before), "after_hash": _fingerprint(after),
            }

            @firestore.transactional
            def commit(transaction):
                current_state = state_reference(db).get(transaction=transaction).to_dict() or {}
                if current_state.get("generation", 0) != generation:
                    raise ConcurrentRatingUpdate("A newer benchmark was published")
                if current_state.get("applied_generation", 0) >= generation:
                    return current_state["receipt"]
                current = {doc.id: doc.to_dict() for doc in db.get_all(refs, transaction=transaction) if doc.exists}
                if current != before:
                    raise ConcurrentRatingUpdate("A rated game was published during recalculation")
                for ref in refs:
                    transaction.set(ref, after[ref.id])
                transaction.set(db.collection("rating_recalculations").document(f"benchmark-{generation}"), {
                    **receipt, "previous_ratings": before,
                })
                transaction.set(state_reference(db), {
                    **current_state, "applied_generation": generation, "receipt": receipt,
                })
                return receipt

            try:
                return commit(db.transaction())
            except ConcurrentRatingUpdate:
                continue
    raise RuntimeError("Ratings kept changing; retry the saved panel sync. No model calls are needed.")


def applied_generation() -> int:
    """Observe cross-process rating updates on Render without per-request queries."""
    global _revision_cache
    if not os.environ.get("RENDER"):
        return 0
    now = monotonic()
    if now - _revision_cache[0] < 10 or not _revision_lock.acquire(blocking=False):
        return _revision_cache[1]
    try:
        from firebase_client import get_firestore_client
        state = state_reference(get_firestore_client()).get().to_dict() or {}
        _revision_cache = (now, int(state.get("applied_generation", 0)))
    except Exception as error:
        logger.warning("Could not refresh benchmark rating revision: %s", error)
        _revision_cache = (now, _revision_cache[1])
    finally:
        _revision_lock.release()
    return _revision_cache[1]
