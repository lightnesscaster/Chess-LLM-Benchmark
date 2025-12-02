"""
Flask web application for LLM Chess Benchmark.

Displays leaderboard and game library with PGN viewer.
"""

import atexit
import logging
import math
import os
import re
import sys
import threading
from pathlib import Path

import chess
import chess.engine
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, abort, request

from rating.rating_store import RatingStore
from rating.leaderboard import Leaderboard
from game.pgn_logger import PGNLogger
from game.stats_collector import StatsCollector

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "benchmark.yaml"
RATINGS_PATH = DATA_DIR / "ratings.json"

# Stockfish configuration
# NOTE: This uses module-level globals and is designed for single-worker
# deployments (Flask development server). For production with multiple workers
# (gunicorn, uwsgi), each worker process will create its own Stockfish instance.
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
_stockfish_engine = None
_stockfish_lock = threading.Lock()
_stockfish_pid = None  # Track PID to detect Flask reloads/worker restarts


def get_stockfish_engine():
    """Get or create the Stockfish engine instance."""
    global _stockfish_engine, _stockfish_pid
    current_pid = os.getpid()

    # Reset engine if we're in a new process (Flask reload or worker restart)
    if _stockfish_pid is not None and _stockfish_pid != current_pid:
        _stockfish_engine = None

    _stockfish_pid = current_pid

    if _stockfish_engine is None:
        try:
            _stockfish_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        except Exception as e:
            app.logger.error(f"Failed to initialize Stockfish: {e}")
            return None
    return _stockfish_engine


def cleanup_stockfish_engine():
    """Close Stockfish engine on shutdown."""
    global _stockfish_engine
    if _stockfish_engine is not None:
        try:
            _stockfish_engine.quit()
        except Exception:
            pass
        _stockfish_engine = None


atexit.register(cleanup_stockfish_engine)


def analyze_position(fen: str, depth: int = 18, num_lines: int = 3) -> dict | None:
    """
    Analyze a chess position with Stockfish.

    Args:
        fen: FEN string of the position
        depth: Search depth
        num_lines: Number of principal variations to return

    Returns:
        Dictionary with evaluation and lines, or None if analysis fails
    """
    engine = get_stockfish_engine()
    if engine is None:
        return None

    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    # Use lock to prevent race conditions with MultiPV configuration
    with _stockfish_lock:
        try:
            # Configure MultiPV for multiple lines
            engine.configure({"MultiPV": num_lines})

            # Run analysis with time limit to prevent blocking other requests
            info = engine.analyse(
                board,
                chess.engine.Limit(depth=depth, time=5.0),
                multipv=num_lines
            )

            lines = []
            for pv_info in info:
                score = pv_info.get("score")
                pv = pv_info.get("pv", [])

                if score is None:
                    continue

                # Convert score to centipawns from white's perspective
                if score.is_mate():
                    mate_in = score.white().mate()
                    score_cp = None
                    # Positive = white wins, negative = black wins
                    score_text = f"M{mate_in}" if mate_in > 0 else f"-M{abs(mate_in)}"
                else:
                    score_cp = score.white().score()
                    score_text = f"{score_cp / 100:+.2f}"

                # Convert PV moves to SAN
                pv_san = []
                temp_board = board.copy()
                for move in pv[:10]:  # Limit to 10 moves
                    try:
                        pv_san.append(temp_board.san(move))
                        temp_board.push(move)
                    except Exception as e:
                        app.logger.warning(f"Failed to convert PV move {move}: {e}")
                        break

                lines.append({
                    "score_cp": score_cp,
                    "score_text": score_text,
                    "mate": score.white().mate() if score.is_mate() else None,
                    "pv": pv_san,
                    "pv_uci": [move.uci() for move in pv[:10]],
                })

            if not lines:
                return None

            return {
                "fen": fen,
                "depth": depth,
                "lines": lines,
                "turn": "white" if board.turn else "black",
            }
        except Exception as e:
            app.logger.error(f"Stockfish analysis failed: {e}")
            return None


def get_anchors_from_config() -> dict:
    """Load anchor IDs and ratings from config file."""
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        return {
            engine["player_id"]: engine["rating"]
            for engine in config.get("engines", [])
        }
    except Exception as e:
        app.logger.warning(f"Could not load anchors from config: {e}")
        return {}

# Game ID validation pattern (UUID format or alphanumeric with hyphens/underscores)
VALID_GAME_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


def is_valid_game_id(game_id: str) -> bool:
    """Validate game_id to prevent path traversal attacks."""
    if not game_id or len(game_id) > 100:
        return False
    return bool(VALID_GAME_ID_PATTERN.match(game_id))


def get_leaderboard_data(min_games: int = 1) -> list:
    """Get leaderboard data from rating store."""
    try:
        anchors = get_anchors_from_config()
        anchor_ids = set(anchors.keys())
        rating_store = RatingStore(path=str(RATINGS_PATH), anchor_ids=anchor_ids)

        # Ensure anchors exist in the store (preserves existing game stats if present)
        for anchor_id, rating in anchors.items():
            if not rating_store.has_player(anchor_id):
                rating_store.set_anchor(anchor_id, rating, auto_save=False)

        pgn_logger = PGNLogger()
        stats_collector = StatsCollector()
        stats_collector.add_results(pgn_logger.load_all_results())

        leaderboard = Leaderboard(rating_store, stats_collector)
        return leaderboard.get_leaderboard(min_games=min_games)
    except Exception as e:
        app.logger.error(f"Error loading leaderboard: {e}")
        return []


def get_all_games() -> list:
    """Get all games with metadata."""
    try:
        pgn_logger = PGNLogger()
        results = pgn_logger.load_all_results()

        # Sort by date descending (most recent first)
        results.sort(key=lambda r: r.created_at or "", reverse=True)

        games = []
        for result in results:
            games.append({
                "game_id": result.game_id,
                "white": result.white_id,
                "black": result.black_id,
                "winner": result.winner,
                "termination": result.termination,
                "moves": math.ceil((result.moves or 0) / 2),
                "illegal_moves_white": result.illegal_moves_white or 0,
                "illegal_moves_black": result.illegal_moves_black or 0,
                "created_at": result.created_at,
            })

        return games
    except Exception as e:
        app.logger.error(f"Error loading games: {e}")
        return []


def get_game(game_id: str) -> dict | None:
    """Get a single game with PGN."""
    try:
        pgn_logger = PGNLogger()
        result = pgn_logger.load_result(game_id)

        if not result:
            return None

        pgn = pgn_logger.load_pgn(game_id)
        if not pgn:
            pgn = "[PGN file not found]"

        return {
            "game_id": result.game_id,
            "white": result.white_id,
            "black": result.black_id,
            "winner": result.winner,
            "termination": result.termination,
            "moves": math.ceil((result.moves or 0) / 2),
            "illegal_moves_white": result.illegal_moves_white or 0,
            "illegal_moves_black": result.illegal_moves_black or 0,
            "created_at": result.created_at,
            "pgn": pgn,
        }
    except Exception as e:
        app.logger.error(f"Error loading game {game_id}: {e}")
        return None


@app.after_request
def set_security_headers(response):
    """Add security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


@app.route("/")
def index():
    """Redirect to leaderboard."""
    return render_template("leaderboard.html", leaderboard=get_leaderboard_data())


@app.route("/leaderboard")
def leaderboard():
    """Show leaderboard page."""
    min_games = 1
    return render_template("leaderboard.html", leaderboard=get_leaderboard_data(min_games))


@app.route("/games")
def games():
    """Show game library."""
    all_games = get_all_games()
    model_filter = request.args.get('model', '').strip()
    if len(model_filter) > 200:
        abort(400)

    # Collect all unique models for the dropdown
    all_models = sorted(set(g['white'] for g in all_games) | set(g['black'] for g in all_games))

    if model_filter:
        filtered_games = [g for g in all_games if g['white'] == model_filter or g['black'] == model_filter]
    else:
        filtered_games = all_games

    return render_template("games.html", filtered_games=filtered_games, all_models=all_models, model_filter=model_filter)


@app.route("/methodology")
def methodology():
    """Show methodology page."""
    return render_template("methodology.html")


@app.route("/game/<game_id>")
def game(game_id: str):
    """Show individual game viewer."""
    if not is_valid_game_id(game_id):
        abort(400)  # Bad request for invalid game_id
    game_data = get_game(game_id)
    if not game_data:
        abort(404)
    return render_template("game.html", game=game_data)


# API endpoints for dynamic updates
@app.route("/api/leaderboard")
def api_leaderboard():
    """Get leaderboard as JSON."""
    return jsonify(get_leaderboard_data())


@app.route("/api/games")
def api_games():
    """Get all games as JSON."""
    return jsonify(get_all_games())


@app.route("/api/game/<game_id>")
def api_game(game_id: str):
    """Get single game as JSON."""
    if not is_valid_game_id(game_id):
        abort(400)
    game_data = get_game(game_id)
    if not game_data:
        abort(404)
    return jsonify(game_data)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Analyze a chess position with Stockfish."""
    data = request.get_json()
    if not data or "fen" not in data:
        abort(400, description="Missing 'fen' parameter")

    fen = data["fen"]
    if not isinstance(fen, str) or len(fen) > 200:
        abort(400, description="Invalid FEN")

    depth = data.get("depth", 18)
    num_lines = data.get("lines", 3)

    # Validate parameters
    if not isinstance(depth, int) or depth < 1 or depth > 30:
        depth = 18
    if not isinstance(num_lines, int) or num_lines < 1 or num_lines > 5:
        num_lines = 3

    result = analyze_position(fen, depth=depth, num_lines=num_lines)
    if result is None:
        abort(503, description="Analysis unavailable - Stockfish may not be installed")

    return jsonify(result)


if __name__ == "__main__":
    debug_mode = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true', 'yes')
    app.run(debug=debug_mode, port=5000)
