"""Engine-vs-engine matches for calibrating engine anchors against lichess bots.

Each "-lichess" engine reproduces a lichess bot's classical conditions: the same
engine build and UCI options, its speed scaled to the bot's stated nodes/second
(probed locally at startup), and each game's clock drawn from the bot's exported
classical time controls. Each opening (8 random gm2001.bin plies) is played twice
with colours swapped. Results are appended to a JSONL file; nothing is published.

Usage:
    python scripts/calibrate_engine_anchor.py <engine_a> <engine_b> <n_openings> <seed> <out.jsonl>
    python scripts/calibrate_engine_anchor.py --summary <out.jsonl> <engine_b_rating>

See docs/anchor_calibration/2026-09-30-eubos/README.md for the eubos calibration.
"""

import json
import math
import random
import sys
import time
from pathlib import Path

import chess
import chess.polyglot

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from engines.maia_engine import MaiaEngine  # noqa: E402
from engines.uci_engine import UCIEngine  # noqa: E402

CACHE = REPO / "data" / "engines_cache"
TIME_CONTROLS = REPO / "docs/anchor_calibration/2026-09-30-eubos/data/lichess_classical_time_controls.json"

# Bot conditions as stated in each lichess bio on 2026-09-30.
LICHESS_EMULATIONS = {
    "eubos-lichess": dict(bot="eubos", rating=2357, path=CACHE / "eubos-4.3/eubos.sh",
                          options={"Threads": 1, "Hash": 512}, reference_nps=500_000),
    "baby-lichess": dict(bot="baby_eubos", rating=2075, path=CACHE / "eubos-4.3/eubos.sh",
                         options={"Threads": 1, "Hash": 64}, reference_nps=25_000),
    "cheng-lichess": dict(bot="Cheng-4", rating=2510, path=CACHE / "cheng-4.39/cheng4",
                          options={"Threads": 1, "Hash": 1024, "OwnBook": False},
                          reference_nps=1_500_000),
}


def make_engine(name):
    if name in LICHESS_EMULATIONS:
        spec = LICHESS_EMULATIONS[name]
        mix = json.loads(TIME_CONTROLS.read_text())["bots"][spec["bot"]]["time_controls"]
        return UCIEngine(name, spec["rating"], str(spec["path"]), options=spec["options"],
                         time_controls=mix, reference_nps=spec["reference_nps"])
    if name == "eubos-local-15+10":  # the anchor setup used for games before 2026-09-30
        return UCIEngine(name, 2346, str(CACHE / "eubos-4.5/eubos.sh"), initial_time=900, increment=10)
    if name.startswith("eubos-") and name.endswith("-fast"):  # e.g. eubos-4.2-fast: version checks
        version = name[len("eubos-"):-len("-fast")]
        return UCIEngine(name, 0, str(CACHE / f"eubos-{version}/eubos.sh"), initial_time=60,
                         increment=0.6, options={"Threads": 1, "Hash": 256})
    if name == "maia-1900":
        return MaiaEngine("maia-1900", 1816, lc0_path="/opt/homebrew/bin/lc0",
                          weights_path=str(REPO / "maia-1900.pb.gz"), nodes=1)
    raise ValueError(f"unknown engine {name}")


def random_opening(rng):
    board = chess.Board()
    with chess.polyglot.open_reader(REPO / "data/openings/gm2001.bin") as reader:
        for _ in range(8):
            entries = list(reader.find_all(board))
            if not entries:
                break
            board.push(rng.choice(entries).move)
    return board


def play(white, black, start):
    board = start.copy()
    for engine in (white, black):
        if hasattr(engine, "reset_clock"):
            engine.reset_clock()
    while not board.is_game_over(claim_draw=True) and board.ply() < 400:
        board.push((white if board.turn == chess.WHITE else black).select_move(board))
    outcome = board.outcome(claim_draw=True)
    result = outcome.result() if outcome else "1/2-1/2"
    return result, (outcome.termination.name if outcome else "MAX_PLIES"), board.ply()


def run(a_name, b_name, n_openings, seed, out_path):
    rng = random.Random(seed)
    random.seed(seed)  # time-control draws
    a, b = make_engine(a_name), make_engine(b_name)
    with open(out_path, "a") as out:
        for _ in range(n_openings):
            start = random_opening(rng)
            for a_white in (True, False):
                began = time.time()
                white, black = (a, b) if a_white else (b, a)
                result, termination, plies = play(white, black, start)
                score = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
                row = dict(a=a_name, b=b_name, seed=seed, opening=start.fen(), a_white=a_white,
                           tcs={e.player_id: getattr(e, "game_time_control", None) for e in (a, b)},
                           result=result, termination=termination, plies=plies,
                           a_score=score if a_white else 1 - score,
                           seconds=round(time.time() - began))
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(json.dumps(row), flush=True)
    a.close()
    b.close()


def summary(path, b_rating):
    scores = [json.loads(line)["a_score"] for line in open(path) if line.strip()]
    n, p = len(scores), sum(scores) / len(scores)
    se = math.sqrt(max(p * (1 - p), 1e-9) / n)

    def perf(q):
        q = min(max(q, 0.005), 0.995)
        return b_rating - 400 * math.log10(1 / q - 1)

    print(f"{n} games, score {p:.3f} -> performance {perf(p):.0f} "
          f"[~95% {perf(p - 1.96 * se):.0f}..{perf(p + 1.96 * se):.0f}]")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2], float(sys.argv[3]))
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
