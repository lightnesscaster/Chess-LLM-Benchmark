"""Ephemeral, public game snapshots with best-effort background publication."""

import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import chess

HEARTBEAT_INTERVAL = 15
STALE_AFTER = 90
ACTIVE_TTL = 30 * 60
TERMINAL_TTL = 10 * 60
STORAGE_TIMEOUT = 3
CACHE_TTL = 2
LIVE_GAMES_COLLECTION = "live_games"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "live_games"
logger = logging.getLogger(__name__)

STRING_FIELDS = (
    "game_id", "white", "black", "fen", "turn", "status", "started_at",
    "updated_at", "result",
)


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def sanitize_snapshot(data: dict) -> dict:
    """Allow only the public schema, including primitive values in move lists."""
    if not isinstance(data, dict):
        raise ValueError("Invalid live snapshot")
    result = {key: data[key] for key in STRING_FIELDS if isinstance(data.get(key), str)}
    if len(result) != len(STRING_FIELDS):
        raise ValueError("Incomplete live snapshot")
    if result["status"] not in ("active", "finished", "error", "interrupted"):
        raise ValueError("Invalid live status")
    if (result["turn"] not in ("white", "black")
            or result["result"] not in ("*", "1-0", "0-1", "1/2-1/2")):
        raise ValueError("Invalid live result or turn")
    for key in ("moves", "san_moves"):
        values = data.get(key)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError("Invalid live moves")
        result[key] = list(values)
    for key in ("heartbeat_at", "expires_at"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("Invalid live timestamp")
        result[key] = value
    for key in ("illegal_moves_white", "illegal_moves_black"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Invalid live illegal counter")
        result[key] = value
    winner = data.get("winner")
    if winner not in ("white", "black", "draw", None):
        raise ValueError("Invalid live winner")
    result["winner"] = winner
    termination = data.get("termination")
    if termination is not None and not isinstance(termination, str):
        raise ValueError("Invalid live termination")
    result["termination"] = termination
    return result


def build_snapshot(
    game_id: str,
    white: str,
    black: str,
    board: chess.Board,
    *,
    started_at: float,
    now: float | None = None,
    status: str = "active",
    winner: str | None = None,
    termination: str | None = None,
    illegal_moves_white: int = 0,
    illegal_moves_black: int = 0,
) -> dict:
    """Replay the actual board history to retain castling and promotion SAN."""
    now = time.time() if now is None else now
    replay = board.root()
    moves, san_moves = [], []
    for move in board.move_stack:
        moves.append(move.uci())
        san_moves.append(replay.san(move))
        replay.push(move)
    return sanitize_snapshot({
        "game_id": game_id, "white": white, "black": black, "fen": board.fen(),
        "moves": moves, "san_moves": san_moves,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "status": status, "started_at": _iso(started_at), "updated_at": _iso(now),
        "heartbeat_at": now,
        "expires_at": now + (ACTIVE_TTL if status in ("active", "interrupted") else TERMINAL_TTL),
        "winner": winner, "termination": termination,
        "result": {"white": "1-0", "black": "0-1", "draw": "1/2-1/2"}.get(winner, "*"),
        "illegal_moves_white": illegal_moves_white,
        "illegal_moves_black": illegal_moves_black,
    })


class LiveGameStore:
    """Atomic local or per-ID Firestore storage; initialization performs no I/O."""

    def __init__(
        self, path: str | Path = DEFAULT_PATH, use_firestore: bool | None = None,
    ) -> None:
        self.path = Path(path)
        self._use_firestore = use_firestore
        self._cache: list[dict] | None = None
        self._cache_time = 0.0
        self._lock = threading.Lock()

    def _firestore_enabled(self) -> bool:
        if self._use_firestore is not None:
            return self._use_firestore
        return (
            os.environ.get("FIREBASE_ENABLED", "").lower() in ("1", "true", "yes")
            or bool(os.environ.get("FIREBASE_CREDENTIALS_JSON"))
            or bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
            or (Path(__file__).resolve().parent.parent / "firebase-key.json").exists()
        )

    def _collection(self):
        from firebase_client import get_firestore_client
        return get_firestore_client().collection(LIVE_GAMES_COLLECTION)

    def write(self, snapshot: dict) -> None:
        """Store only public fields; propagate storage failures to the publisher."""
        data = sanitize_snapshot(snapshot)
        if self._firestore_enabled():
            self._collection().document(data["game_id"]).set(data, timeout=STORAGE_TIMEOUT, retry=None)
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            name = hashlib.sha256(data["game_id"].encode()).hexdigest() + ".json"
            temporary = None
            try:
                with tempfile.NamedTemporaryFile("w", dir=self.path, suffix=".tmp", delete=False) as output:
                    temporary = output.name
                    json.dump(data, output, ensure_ascii=False)
                os.replace(temporary, self.path / name)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
        with self._lock:
            self._cache = None

    def _read(self, now: float) -> list[dict]:
        if self._firestore_enabled():
            from google.cloud.firestore_v1.base_query import FieldFilter
            query = self._collection().where(filter=FieldFilter("expires_at", ">", now))
            return [doc.to_dict() for doc in query.stream(timeout=STORAGE_TIMEOUT, retry=None)]
        try:
            files = list(self.path.iterdir())
        except FileNotFoundError:
            return []
        records = []
        for path in files:
            if path.suffix != ".json":
                continue
            try:
                records.append(json.loads(path.read_text()))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("Ignoring invalid live snapshot")
        return records

    def list_games(self, now: float | None = None) -> list[dict]:
        """Return fresh public views; expired records disappear and stale ones interrupt."""
        now = time.time() if now is None else now
        with self._lock:
            if self._cache is None or time.monotonic() - self._cache_time >= CACHE_TTL:
                records = self._read(now)
                self._cache = records
                self._cache_time = time.monotonic()
            records = self._cache
            games = []
            for record in records:
                try:
                    data = sanitize_snapshot(record)
                except ValueError:
                    continue
                if data["expires_at"] <= now:
                    continue
                if data["status"] == "active" and now - data["heartbeat_at"] > STALE_AFTER:
                    data["status"] = "interrupted"
                games.append(data)
        games.sort(
            key=lambda game: (game["status"] == "active", game["started_at"], game["game_id"]),
            reverse=True,
        )
        return games


@lru_cache(maxsize=1)
def get_live_game_store() -> LiveGameStore:
    """Share the short-lived read cache within this process."""
    return LiveGameStore()


class LiveGamePublisher:
    """One coalescing daemon writer and one heartbeat task for a game.

    Daemon storage work deliberately avoids asyncio's default executor: a hung
    filesystem or SDK cannot prevent event-loop shutdown or change game results.
    The writer remains serialized even if the bounded close times out, so an
    older slow write cannot overwrite a newer terminal state.
    """

    def __init__(self, store: LiveGameStore) -> None:
        self.store = store
        self._condition = threading.Condition()
        self._pending: dict | None = None
        self._latest: dict | None = None
        self._closing = False
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._heartbeat_task: asyncio.Task | None = None

    def start(self) -> None:
        self._thread.start()
        self._heartbeat_task = asyncio.create_task(self._heartbeat())

    def submit(self, snapshot: dict) -> None:
        data = sanitize_snapshot(snapshot)
        with self._condition:
            if self._closing:
                return
            self._latest = data
            self._pending = data
            self._condition.notify()

    def finish(self, status: str, termination: str | None = None) -> None:
        """Mark an aborted run without exposing the exception or private details."""
        if self._latest is None or self._latest["status"] != "active":
            return
        data = copy.deepcopy(self._latest)
        now = time.time()
        data.update(status=status, termination=termination, updated_at=_iso(now), heartbeat_at=now,
                    expires_at=now + (ACTIVE_TTL if status == "interrupted" else TERMINAL_TTL))
        self.submit(data)

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            if self._latest is None or self._latest["status"] != "active":
                continue
            data = copy.deepcopy(self._latest)
            now = time.time()
            data.update(heartbeat_at=now, expires_at=now + ACTIVE_TTL)
            self.submit(data)

    def _write_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closing)
                if self._pending is None:
                    return
                data, self._pending = self._pending, None
            try:
                self.store.write(data)
            except Exception:
                logger.warning("Live snapshot publication unavailable")

    async def close(self, timeout: float = 1.0) -> None:
        """Stop heartbeats and allow at most timeout seconds for the final write."""
        try:
            if self._heartbeat_task is not None:
                self._heartbeat_task.cancel()
                # Gather turns only the child's cancellation into a result.
                # A new cancellation of this caller still propagates, including
                # when close runs while handling an earlier inference cancel.
                await asyncio.gather(self._heartbeat_task, return_exceptions=True)
        finally:
            with self._condition:
                self._closing = True
                self._condition.notify()
        deadline = time.monotonic() + timeout
        while self._thread.is_alive() and time.monotonic() < deadline:
            await asyncio.sleep(min(0.01, max(0, deadline - time.monotonic())))
