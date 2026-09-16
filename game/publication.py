"""Publish saved manual games atomically; safe to retry the same game ID."""

import base64
import copy
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request
import uuid

import chess
import chess.pgn

from game.models import GameResult
from rating.glicko2 import Glicko2System, PlayerRating


SERVICE_ID = "srv-d4mlb0m3jp1c73a172v0"
WEB_URL = "https://chessbenchllm.onrender.com"
COMPLETED_TERMINATIONS = {
    "resignation", "forfeit_illegal_move", "max_moves", "checkmate",
    "stalemate", "insufficient_material", "seventyfive_moves",
    "fivefold_repetition", "fifty_moves", "threefold_repetition",
}


def load_production_ratings(store) -> None:
    """Refresh in-memory game history and prompt ratings, without overwriting local data.

    The public leaderboard rounds rating/RD; these snapshots are only for game
    eligibility and prompts. Publication still uses exact transactional ratings.
    """
    with urllib.request.urlopen(WEB_URL + "/api/leaderboard", timeout=30) as response:
        rows = json.load(response)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Production leaderboard is empty or invalid")
    ratings = [PlayerRating(
        player_id=row["player_id"], rating=row["rating"],
        rating_deviation=row["rating_deviation"], games_rd=row["rating_deviation"],
        games_played=row["games_played"], wins=row["wins"],
        losses=row["losses"], draws=row["draws"], is_frozen=row.get("is_frozen", False),
    ) for row in rows]
    for rating in ratings:
        store.set(rating, auto_save=False)


def validate_result(result: GameResult, pgn: str) -> None:
    """Reject incomplete, mismatched, or non-benchmark records before writes."""
    uuid.UUID(result.game_id)
    if result.game_type != "benchmark" or result.termination not in COMPLETED_TERMINATIONS:
        raise ValueError("Only completed benchmark games can be published")
    if result.white_id == result.black_id:
        raise ValueError("Players must be distinct")
    stream = io.StringIO(pgn)
    game = chess.pgn.read_game(stream)
    if game is None or game.errors or chess.pgn.read_game(stream) is not None:
        raise ValueError("Expected exactly one legal PGN")
    expected = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2"}.get(result.winner)
    for key, value in {"White": result.white_id, "Black": result.black_id,
                       "Result": expected, "Termination": result.termination}.items():
        if value is None or game.headers.get(key) != value:
            raise ValueError(f"PGN {key} does not match result")
    moves = list(game.mainline_moves())
    if len(moves) != result.moves:
        raise ValueError("PGN plies do not match result")
    board = game.end().board()
    if result.termination in {"resignation", "forfeit_illegal_move"}:
        winner = "black" if board.turn == chess.WHITE else "white"
        if result.winner != winner or board.is_game_over():
            raise ValueError("Resignation/forfeit does not match side to move")
        if result.termination == "forfeit_illegal_move":
            strikes = result.illegal_moves_white if board.turn else result.illegal_moves_black
            if strikes < 2:
                raise ValueError("Forfeit requires two illegal-move strikes")
    elif result.termination == "max_moves":
        if result.winner != "draw":
            raise ValueError("Move-limit result must be a draw")
    else:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.result() != expected or outcome.termination.name.lower() != result.termination:
            raise ValueError("Natural termination does not match final board")


def commit_result(tx, db, result: GameResult, pgn: str, anchors: set, ghosts: set) -> dict:
    """Transaction body: all reads precede all writes; duplicate IDs never rerate."""
    validate_result(result, pgn)
    refs = [db.collection(collection).document(key) for collection, key in (
        ("results", result.game_id), ("games", result.game_id),
        ("ratings", result.white_id), ("ratings", result.black_id),
    )]
    existing, replay, white, black = [next(tx.get(ref)) for ref in refs]
    data = result.to_json()
    data["pgn_path"] = f"firestore://games/{result.game_id}"
    if existing.exists:
        old = GameResult.from_json(existing.to_dict()).to_json()
        old["pgn_path"] = data["pgn_path"]
        if old != data or not replay.exists or replay.to_dict().get("pgn") != pgn:
            raise ValueError("Existing game ID conflict; refusing to overwrite")
        return {"game_id": result.game_id, "already_published": True}
    if replay.exists:
        raise ValueError("Orphan replay conflict; manual review required")
    if not white.exists or not black.exists:
        raise ValueError("Missing production player rating; register/seed players first")
    players = [PlayerRating.from_dict(doc.to_dict()) for doc in (white, black)]
    score = {"white": 1.0, "black": 0.0, "draw": 0.5}[result.winner]
    updated = []
    for player, opponent, points in ((players[0], players[1], score), (players[1], players[0], 1 - score)):
        if player.player_id in anchors or opponent.player_id in ghosts:
            new = copy.deepcopy(player)
            new.games_played += 1
            new.wins += int(points == 1)
            new.losses += int(points == 0)
            new.draws += int(points == 0.5)
        else:
            new = Glicko2System().update_rating(player, [opponent], [points])
        updated.append(new)
    tx.set(refs[0], data)
    tx.set(refs[1], {"game_id": result.game_id, "pgn": pgn})
    for ref, player in zip(refs[2:], updated):
        tx.set(ref, player.to_dict())
    return {"game_id": result.game_id, "already_published": False,
            "ratings": {p.player_id: p.rating for p in updated}}


def publish_in_production(payload: dict) -> dict:
    """Run on the authenticated server; finish freeze/cache steps on retries too."""
    import yaml
    from firebase_admin import firestore
    from firebase_client import get_firestore_client
    from game.freeze_checker import FreezeChecker
    from game.pgn_logger import PGNLogger
    from game.stats_collector import StatsCollector
    from rating.rating_store import RatingStore
    from utils import is_reasoning_model

    result = GameResult.from_json(payload["result"])
    config = yaml.safe_load(Path("config/benchmark.yaml").read_text())
    engines = config.get("engines", [])
    anchors = {e["player_id"] for e in engines if e.get("anchor", True)}
    ghosts = {e["player_id"] for e in engines if e.get("ghost")}
    db = get_firestore_client()
    receipt = firestore.transactional(commit_result)(
        db.transaction(), db, result, payload["pgn"], anchors, ghosts,
    )
    store = RatingStore(use_firestore=True, anchor_ids=anchors, ghost_ids=ghosts)
    stats = StatsCollector()
    stats.add_results(PGNLogger(use_firestore=True).load_all_results())
    checker = FreezeChecker(store, stats,
                            reasoning_ids={p for p in store.get_all() if is_reasoning_model(p)},
                            engine_ids={e["player_id"] for e in engines})
    frozen = {}
    for pid in (result.white_id, result.black_id):
        if pid not in anchors:
            rating = store.get(pid)
            frozen[pid] = checker.is_frozen(pid, rating.rating_deviation)

    @firestore.transactional
    def update_freeze(tx):
        refs = {pid: db.collection("ratings").document(pid) for pid in frozen}
        current = {pid: next(tx.get(ref)).to_dict() for pid, ref in refs.items()}
        for pid, data in current.items():
            rating = store.get(pid)
            if data["rating"] != rating.rating or data["games_played"] != rating.games_played:
                raise ValueError("Ratings changed during freeze check; retry publication")
        for pid, ref in refs.items():
            tx.update(ref, {"is_frozen": frozen[pid]})

    update_freeze(db.transaction())
    req = urllib.request.Request(WEB_URL + "/api/invalidate-cache", data=b"",
                                 headers={"X-Cache-Token": os.environ["CACHE_INVALIDATE_TOKEN"]}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("Cache invalidation failed")
    receipt["frozen"] = frozen
    print("PUBLICATION_COMPLETE " + json.dumps(receipt), flush=True)
    return receipt


def render_json(arguments: list[str]):
    """Call the installed authenticated Render CLI without logging job payloads."""
    process = subprocess.run(["render", *arguments, "--output", "json", "--confirm"],
                             capture_output=True, text=True, timeout=90)
    if process.returncode:
        raise RuntimeError("Render command failed; check Render authentication/service access")
    return json.loads(process.stdout)


def publish_saved_game(result_path: str | Path, timeout: float = 900) -> dict:
    """Submit a durable saved game, wait for publication, and retain a retry receipt."""
    result_path = Path(result_path).resolve()
    result = GameResult.from_json(json.loads(result_path.read_text()))
    pgn = Path(result.pgn_path).read_text()
    validate_result(result, pgn)
    receipt_path = result_path.with_suffix(".publication")
    state = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if state.get("status") == "published":
        return state
    payload = base64.b64encode(json.dumps({"result": result.to_json(), "pgn": pgn}).encode()).decode()
    if not state.get("job_id") or state.get("status") == "failed":
        # Ship the checked-in worker with the job, so a local CLI upgrade does
        # not require a web deployment. No credentials are copied to the client.
        source = base64.b64encode(Path(__file__).read_bytes()).decode()
        code = (f"import base64,json; ns={{'__name__':'publication_worker'}}; "
                f"exec(compile(base64.b64decode({source!r}),'<publication>','exec'),ns); "
                f"ns['publish_in_production'](json.loads(base64.b64decode({payload!r})))")
        job = render_json(["jobs", "create", SERVICE_ID, "--start-command", "python -c " + shlex.quote(code)])
        state = {"game_id": result.game_id, "job_id": job["id"], "status": "pending"}
        receipt_path.write_text(json.dumps(state, indent=2))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # A success marker is printed only after the transaction, freeze check,
        # and cache invalidation. Job success alone is not enough.
        process = subprocess.run(["render", "logs", "--resources", state["job_id"],
                                  "--limit", "10", "--direction", "backward", "--output", "json", "--confirm"],
                                 capture_output=True, text=True, timeout=90)
        if process.returncode:
            raise RuntimeError(f"Cannot read publication job {state['job_id']}; retry saved result")
        text = process.stdout
        decoder = json.JSONDecoder()
        while text.strip():
            entry, end = decoder.raw_decode(text.lstrip())
            text = text.lstrip()[end:]
            message = entry.get("message", "")
            if message.startswith("PUBLICATION_COMPLETE "):
                receipt = json.loads(message.removeprefix("PUBLICATION_COMPLETE "))
                if receipt.get("game_id") != result.game_id:
                    raise ValueError("Publication receipt game ID mismatch")
                state.update(status="published", receipt=receipt)
                receipt_path.write_text(json.dumps(state, indent=2))
                return state
        jobs = render_json(["jobs", "list", SERVICE_ID])
        job = next((j for j in jobs if j.get("id") == state["job_id"]), {})
        if job.get("status") in {"failed", "canceled", "cancelled"}:
            state["status"] = "failed"
            receipt_path.write_text(json.dumps(state, indent=2))
            raise RuntimeError(f"Publication failed: {state['job_id']}; retry saved result")
        time.sleep(10)
    raise TimeoutError(f"Publication pending: {state['job_id']}; retry saved result to keep waiting")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Publish/retry one saved manual game and apply ratings")
    parser.add_argument("result_path")
    args = parser.parse_args()
    print(json.dumps(publish_saved_game(args.result_path), indent=2))
