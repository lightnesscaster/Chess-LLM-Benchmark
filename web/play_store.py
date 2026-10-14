"""Authoritative active human games, independent of response-cookie ordering."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from contextlib import contextmanager
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache

from firebase_client import get_firestore_client
from web.approved_players import normalize_email
from web.play_service import _board_from_state


ACTIVE_PLAY_COLLECTION = "active_play_games"
TURN_LEASE_SECONDS = 1500  # Above the longest configured model timeout (1200s).
logger = logging.getLogger(__name__)


class PlayConflict(ValueError):
    """A request targets an old position or overlaps an unfinished turn."""


@dataclass(frozen=True)
class TurnClaim:
    owner: str
    nonce: str
    state: dict | None


class ActivePlayStore:
    """Keep one active game per account and serialize expensive turn requests."""

    def __init__(self, db=None, *, journal_dir: Path | None = None):
        self._db = db if db is not None else get_firestore_client()
        self._journal_dir = journal_dir or Path(os.environ.get(
            "PLAY_TURN_JOURNAL_DIR", "/var/data/play-turns" if os.environ.get("RENDER") else "data/play_turns",
        ))

    def _reference(self, email: str):
        normalized = normalize_email(email)
        if not normalized:
            raise ValueError("A verified player account is required.")
        key = hashlib.sha256(normalized.encode()).hexdigest()
        return self._db.collection(ACTIVE_PLAY_COLLECTION).document(key)

    def load(self, email: str, legacy: dict | None = None) -> dict | None:
        """Import an existing signed-cookie game once; never overwrite newer data."""
        from firebase_admin import firestore

        self._drain_journal(email)
        reference = self._reference(email)

        @firestore.transactional
        def read(transaction):
            snapshot = reference.get(transaction=transaction)
            if snapshot.exists:
                return copy.deepcopy((snapshot.to_dict() or {}).get("state"))
            if legacy is not None:
                uuid.UUID(legacy["game_id"])
                _board_from_state(legacy)
                transaction.set(reference, {"state": copy.deepcopy(legacy), "updated_at": time.time()})
                return copy.deepcopy(legacy)
            return None

        return read(self._db.transaction())

    def begin(self, email: str, expected: dict, *, starting: bool = False) -> TurnClaim:
        """Claim a specific displayed board before any model call or correction."""
        from firebase_admin import firestore

        self._drain_journal(email)
        self._owner_directory(email).mkdir(parents=True, exist_ok=True, mode=0o700)
        reference = self._reference(email)
        nonce = str(uuid.uuid4())

        @firestore.transactional
        def claim(transaction):
            snapshot = reference.get(transaction=transaction)
            record = snapshot.to_dict() or {} if snapshot.exists else {}
            state = record.get("state")
            pending = record.get("pending") or {}
            if pending.get("expires_at", 0) > time.time():
                raise PlayConflict("Another turn is still processing. Wait for it to finish, then refresh this game.")
            if starting and state and state.get("status") == "finished" and not state.get("rating_result"):
                raise PlayConflict("The finished game's rating is still pending. Refresh to retry saving it before starting another game.")
            if not starting:
                if state is None:
                    raise PlayConflict("Start a new game first.")
                if (
                    expected.get("game_id") != state.get("game_id")
                    or type(expected.get("expected_ply")) is not int
                    or expected["expected_ply"] != len(state["moves"])
                    or expected.get("expected_fen") != _board_from_state(state).fen()
                ):
                    raise PlayConflict("This move targets an older position. Refresh the page and check the saved position before moving again.")
            transaction.set(reference, {
                **record, "pending": {"nonce": nonce, "expires_at": time.time()+TURN_LEASE_SECONDS},
            })
            return TurnClaim(email, nonce, copy.deepcopy(state))

        return claim(self._db.transaction())

    def finish(self, claim: TurnClaim, state: dict | None) -> None:
        """Save the outcome only if this request still owns the turn."""
        directory = self._owner_directory(claim.owner)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = directory / f"{claim.nonce}.json"
        temporary = directory / f"{claim.nonce}.{uuid.uuid4()}.tmp"
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w") as handle:
                json.dump({"owner": claim.owner, "nonce": claim.nonce, "state": state}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)  # An outcome is immutable for its claim.
            except FileExistsError:
                saved = json.loads(path.read_text())
                if saved != {"owner": claim.owner, "nonce": claim.nonce, "state": state}:
                    raise PlayConflict("A completed turn is waiting to be saved. Refresh the game.")
        finally:
            temporary.unlink(missing_ok=True)
        self._commit(claim, state)
        path.unlink(missing_ok=True)

    def _owner_directory(self, email: str) -> Path:
        return self._journal_dir / hashlib.sha256(normalize_email(email).encode()).hexdigest()

    def _drain_journal(self, email: str) -> None:
        for path in self._owner_directory(email).glob("*.json"):
            try:
                record = json.loads(path.read_text())
            except FileNotFoundError:
                continue  # Another worker has just delivered the same turn.
            if normalize_email(record["owner"]) != normalize_email(email):
                raise ValueError("Saved turn owner does not match this account.")
            claim = TurnClaim(record["owner"], record["nonce"], None)
            try:
                self._commit(claim, record["state"])
            except PlayConflict:
                # Keep late superseded outcomes for diagnosis, never apply them.
                try:
                    path.rename(path.with_suffix(".superseded"))
                except FileNotFoundError:
                    pass
            else:
                path.unlink(missing_ok=True)

    def _commit(self, claim: TurnClaim, state: dict | None) -> None:
        from firebase_admin import firestore

        reference = self._reference(claim.owner)

        @firestore.transactional
        def commit(transaction):
            snapshot = reference.get(transaction=transaction)
            record = snapshot.to_dict() or {} if snapshot.exists else {}
            if (record.get("pending") or {}).get("nonce") != claim.nonce:
                if record.get("state") == state:
                    return  # The commit succeeded, but its acknowledgement was lost.
                raise PlayConflict("This turn was superseded. Refresh the game to see the saved position.")
            transaction.set(reference, {"state": copy.deepcopy(state), "updated_at": time.time()})

        commit(self._db.transaction())

    def renew(self, claim: TurnClaim) -> None:
        """Keep the whole turn exclusive across provider retries and strike retries."""
        from firebase_admin import firestore
        reference = self._reference(claim.owner)

        @firestore.transactional
        def heartbeat(transaction):
            snapshot = reference.get(transaction=transaction)
            record = snapshot.to_dict() or {} if snapshot.exists else {}
            if (record.get("pending") or {}).get("nonce") != claim.nonce:
                raise PlayConflict("This turn no longer owns the game.")
            transaction.set(reference, {
                **record, "pending": {"nonce": claim.nonce, "expires_at": time.time()+TURN_LEASE_SECONDS},
            })

        heartbeat(self._db.transaction())

    @contextmanager
    def keep_alive(self, claim: TurnClaim):
        stop = threading.Event()
        def heartbeat():
            while not stop.wait(60):
                try:
                    self.renew(claim)
                except Exception:
                    logger.warning("Could not renew active human turn")
        thread = threading.Thread(target=heartbeat, name="play-turn-lease", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=2)

    def annotate_rating(self, email: str, game_id: str, result: dict) -> None:
        """Attach an idempotent publication receipt without replacing a newer game."""
        from firebase_admin import firestore

        reference = self._reference(email)

        @firestore.transactional
        def annotate(transaction):
            snapshot = reference.get(transaction=transaction)
            record = snapshot.to_dict() or {} if snapshot.exists else {}
            state = record.get("state")
            if state and state.get("game_id") == game_id and state.get("status") == "finished":
                transaction.set(reference, {**record, "state": {**state, "rating_result": result}})

        annotate(self._db.transaction())


@lru_cache(maxsize=1)
def get_active_play_store() -> ActivePlayStore:
    return ActivePlayStore()
