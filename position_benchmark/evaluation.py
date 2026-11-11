"""Versioned, history-preserving Stockfish scoring for position benchmarks."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

import chess
import chess.engine

from position_benchmark.scoring import ILLEGAL_MOVE_EVAL, illegal_move_cpl


SCORING_VERSION = "root-reference-v1"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "position_scoring.sqlite3"


def scoring_profile_identity(profile: Mapping[str, Any]) -> dict:
    """Return compact provenance for either a full or already compact profile.

    The digest covers the complete engine/options/NNUE profile. Keeping that
    profile once in an artifact preserves audit detail without repeating it on
    every scored move. This conversion never changes cached analysis keys.
    """
    if set(profile) == {"sha256", "engine", "depth"}:
        return dict(profile)
    encoded = json.dumps(dict(profile), sort_keys=True, separators=(",", ":"))
    identity = profile["engine"]
    return {
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "engine": identity.get("name", "unknown") if isinstance(identity, Mapping) else str(identity),
        "depth": profile["depth"],
    }


def _eval_to_cp(info: Mapping[str, Any], perspective: chess.Color) -> float:
    """Use the benchmark's existing centipawn/mate scale from the root side."""
    if info.get("score") is None:
        raise ValueError("Engine analysis did not return a score")
    score = info["score"].pov(perspective)
    if score.is_mate():
        mate = score.mate()
        return 10000 - mate * 10 if mate > 0 else -10000 - mate * 10
    cp = score.score()
    return cp if cp is not None else 0.0


class ConsistentEvaluator:
    """Compare a reference and selected move at the same full-history root.

    Real UCI engines use the persistent production cache by default. Lightweight
    engine doubles share an in-memory cache on that engine unless given a path.
    Each uncached search receives an independent game token and a cleared hash.
    """

    def __init__(
        self,
        engine: chess.engine.SimpleEngine,
        depth: int = 30,
        cache_path: str | Path | None = None,
    ) -> None:
        if depth < 1:
            raise ValueError("Scoring depth must be positive")
        self.engine = engine
        self.depth = depth
        options = getattr(engine, "options", {})
        self._options = dict(options) if isinstance(options, Mapping) else {}
        required = {
            "Threads": 1, "Hash": 64, "Skill Level": 20,
            "UCI_LimitStrength": False, "SyzygyPath": "", "UCI_AnalyseMode": True,
        }
        self._configuration = {name: value for name, value in required.items() if name in self._options}
        if self._configuration:
            engine.configure(self._configuration)
        self.engine_profile = self._profile()
        self.scoring_profile = scoring_profile_identity(self.engine_profile)
        self.cache_path = Path(cache_path) if cache_path is not None else (
            DEFAULT_CACHE_PATH if isinstance(engine, chess.engine.SimpleEngine) else None
        )
        state = getattr(engine, "_consistent_evaluation_state", None)
        if not isinstance(state, dict):
            state = {"cache": {}, "lock": threading.RLock()}
            setattr(engine, "_consistent_evaluation_state", state)
        self._memory = state["cache"]
        self._lock = state["lock"]
        parameters = inspect.signature(engine.analyse).parameters.values()
        self._supports_game = any(p.name == "game" or p.kind == p.VAR_KEYWORD for p in parameters)
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connection() as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS position_scores "
                    "(cache_key TEXT PRIMARY KEY, result TEXT NOT NULL)"
                )

    def _profile(self) -> dict:
        identity = getattr(self.engine, "id", {})
        if not isinstance(identity, Mapping):
            identity = {"name": type(self.engine).__qualname__}
        effective = {name: option.default for name, option in self._options.items() if option.type != "button"}
        protocol = getattr(self.engine, "protocol", None)
        configured = getattr(protocol, "target_config", {})
        if isinstance(configured, Mapping):
            effective.update(configured)
        effective.update(self._configuration)
        # python-chess manages these options itself for single-PV analysis.
        for name, value in {"MultiPV": 1, "Ponder": False, "UCI_AnalyseMode": True}.items():
            if name in effective:
                effective[name] = value
        nnue = {}
        for name, value in effective.items():
            if "evalfile" in name.lower():
                path = Path(str(value)).expanduser()
                nnue[name] = {"file": str(value)}
                if path.is_file():
                    nnue[name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "scoring_version": SCORING_VERSION,
            "depth": self.depth,
            "engine": dict(identity),
            "options": effective,
            "nnue": nnue,
        }

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.cache_path), timeout=60)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _key(self, board: chess.Board, move: chess.Move | None) -> str:
        payload = {
            "profile": self.engine_profile,
            "root_fen": board.root().fen(en_passant="fen"),
            "moves": [played.uci() for played in board.move_stack],
            "fen": board.fen(en_passant="fen"),
            "chess960": board.chess960,
            "variant": type(board).__qualname__,
            "move": move.uci() if move is not None else None,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _search(self, board: chess.Board, move: chess.Move | None = None) -> dict:
        key = self._key(board, move)
        cached = self._memory.get(key)
        if cached is not None:
            return dict(cached)
        if self.cache_path is not None:
            with self._connection() as connection:
                row = connection.execute("SELECT result FROM position_scores WHERE cache_key = ?", (key,)).fetchone()
            if row is not None:
                self._memory[key] = json.loads(row[0])
                return dict(self._memory[key])
        if self._configuration:
            self.engine.configure(self._configuration)
        if "Clear Hash" in self._options:
            self.engine.configure({"Clear Hash": None})
        kwargs = {"root_moves": [move]} if move is not None else {}
        if self._supports_game:
            kwargs["game"] = object()
        info = self.engine.analyse(board.copy(stack=True), chess.engine.Limit(depth=self.depth), **kwargs)
        pv = info.get("pv", [])
        best = pv[0] if pv else None
        if move is None and best is None and any(board.legal_moves):
            raise ValueError("Engine analysis did not return a best move")
        value = {"eval": _eval_to_cp(info, board.turn), "best_move": best.uci() if best else None}
        if self.cache_path is not None:
            with self._connection() as connection:
                # The first completed writer owns this immutable cache entry.
                connection.execute("INSERT OR IGNORE INTO position_scores VALUES (?, ?)", (key, json.dumps(value)))
                row = connection.execute("SELECT result FROM position_scores WHERE cache_key = ?", (key,)).fetchone()
                value = json.loads(row[0])
        self._memory[key] = value
        return dict(value)

    def score(self, board: chess.Board, move_or_none: chess.Move | None) -> dict:
        """Return same-root CPL and scoring provenance without mutating board."""
        with self._lock:
            reference = self._search(board)
            best_uci = reference["best_move"]
            best = chess.Move.from_uci(best_uci) if best_uci is not None else None
            legal = move_or_none is not None and move_or_none in board.legal_moves
            is_best = bool(legal and move_or_none == best)
            eval_best = reference["eval"]
            if not legal:
                eval_model = ILLEGAL_MOVE_EVAL
                cpl = illegal_move_cpl(eval_best)
            else:
                eval_model = eval_best if is_best else self._search(board, move_or_none)["eval"]
                cpl = max(0, eval_best - eval_model)
            return {
                "best_move": best_uci,
                "best_move_san": board.san(best) if best is not None else None,
                "eval_before": eval_best,
                "eval_best": eval_best,
                "eval_model": eval_model,
                "cpl": cpl,
                "is_best": is_best,
                "scoring_version": SCORING_VERSION,
                "scoring_profile": self.scoring_profile,
            }
