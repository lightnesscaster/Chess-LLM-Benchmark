"""
Flask web application for LLM Chess Benchmark.

Displays leaderboard and game library with PGN viewer.
"""

import logging
import os
import re
import sys
from pathlib import Path

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
                "moves": result.moves or 0,
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
            "moves": result.moves or 0,
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
        model_filter = model_filter[:200]

    # Collect all unique models for the dropdown
    all_models = sorted(set(g['white'] for g in all_games) | set(g['black'] for g in all_games))

    if model_filter:
        filtered_games = [g for g in all_games if g['white'] == model_filter or g['black'] == model_filter]
    else:
        filtered_games = all_games

    return render_template("games.html", filtered_games=filtered_games, all_models=all_models, model_filter=model_filter)


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


if __name__ == "__main__":
    debug_mode = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true', 'yes')
    app.run(debug=debug_mode, port=5000)
