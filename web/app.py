"""
Flask web application for LLM Chess Benchmark.

Displays leaderboard and game library with PGN viewer.
"""

import logging
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, abort, request

from rating.rating_store import RatingStore, _CACHE_INVALIDATE_FILE
from rating.leaderboard import Leaderboard
from game.pgn_logger import PGNLogger
from game.stats_collector import StatsCollector
from web.timeline_chart import get_timeline_html
from web.cost_chart import get_cost_chart_html
from utils import is_reasoning_model

app = Flask(__name__)

# Register custom Jinja filter for reasoning model detection
app.jinja_env.filters['is_reasoning'] = is_reasoning_model

# Leaderboard cache
_leaderboard_cache: list = []
_leaderboard_cache_time: float = 0
_leaderboard_lock = threading.Lock()
_leaderboard_refreshing = False
_LEADERBOARD_CACHE_TTL = 86400  # 24 hours

# Games cache
_games_cache: list = []
_games_cache_time: float = 0
_games_lock = threading.Lock()
_games_refreshing = False
_GAMES_CACHE_TTL = 86400  # 24 hours


def _should_invalidate_cache(cache_time: float) -> bool:
    """Check if cache should be invalidated based on signal file."""
    try:
        if not _CACHE_INVALIDATE_FILE.exists():
            return False
        file_mtime = _CACHE_INVALIDATE_FILE.stat().st_mtime
        return file_mtime > cache_time
    except OSError:
        return False

# Configure logging
logging.basicConfig(level=logging.INFO)

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "benchmark.yaml"
RATINGS_PATH = DATA_DIR / "ratings.json"

def get_anchors_from_config() -> dict:
    """Load anchor IDs and ratings from config file.

    Only includes engines with anchor: true (default).
    Engines with anchor: false have updatable ratings and are not anchors.
    """
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        return {
            engine["player_id"]: engine["rating"]
            for engine in config.get("engines", [])
            if engine.get("anchor", True)  # Default to True for backwards compat
        }
    except Exception as e:
        app.logger.warning(f"Could not load anchors from config: {e}")
        return {}


def get_all_engine_ids_from_config() -> set:
    """Load all engine IDs from config (both anchor and non-anchor)."""
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        return {
            engine["player_id"]
            for engine in config.get("engines", [])
        }
    except Exception as e:
        app.logger.warning(f"Could not load engine IDs from config: {e}")
        return set()

# Game ID validation pattern (UUID format or alphanumeric with hyphens/underscores)
VALID_GAME_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


def is_valid_game_id(game_id: str) -> bool:
    """Validate game_id to prevent path traversal attacks."""
    if not game_id or len(game_id) > 100:
        return False
    return bool(VALID_GAME_ID_PATTERN.match(game_id))


def get_leaderboard_data(min_games: int = 1, sort_by: str = "rating") -> list:
    """Get leaderboard data from rating store with thread-safe caching."""
    global _leaderboard_cache, _leaderboard_cache_time, _leaderboard_refreshing

    # Check cache under lock
    with _leaderboard_lock:
        cache_age = time.time() - _leaderboard_cache_time
        cache_valid = _leaderboard_cache and cache_age < _LEADERBOARD_CACHE_TTL
        should_invalidate = _should_invalidate_cache(_leaderboard_cache_time)

        if cache_valid and sort_by == "rating" and not should_invalidate:
            app.logger.debug(f"Using cached leaderboard data ({cache_age:.0f}s old)")
            return list(_leaderboard_cache)

        # Thundering herd prevention: if another thread is refreshing, return stale cache
        if _leaderboard_refreshing and _leaderboard_cache:
            app.logger.debug("Another thread is refreshing leaderboard, returning stale cache")
            return list(_leaderboard_cache)

        # Mark that we're refreshing
        _leaderboard_refreshing = True

    # Fetch data outside lock to avoid blocking other threads
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
        result = leaderboard.get_leaderboard(min_games=min_games, sort_by=sort_by)

        # Mark all engine entries (anchor or not) so the UI can distinguish them from LLMs
        all_engine_ids = get_all_engine_ids_from_config()
        for entry in result:
            entry["is_engine"] = entry["player_id"] in all_engine_ids

        # Update cache under lock
        with _leaderboard_lock:
            if sort_by == "rating":
                _leaderboard_cache = result
                _leaderboard_cache_time = time.time()
            _leaderboard_refreshing = False

        return result
    except Exception as e:
        app.logger.error(f"Error loading leaderboard: {e}")
        with _leaderboard_lock:
            _leaderboard_refreshing = False
            # Return cached data if available, even if expired
            if _leaderboard_cache:
                app.logger.info("Returning stale cached leaderboard data due to error")
                return list(_leaderboard_cache)
        return []


def get_all_games() -> list:
    """Get all games with metadata, with thread-safe caching."""
    global _games_cache, _games_cache_time, _games_refreshing

    # Check cache under lock
    with _games_lock:
        cache_age = time.time() - _games_cache_time
        cache_valid = _games_cache and cache_age < _GAMES_CACHE_TTL
        should_invalidate = _should_invalidate_cache(_games_cache_time)

        if cache_valid and not should_invalidate:
            app.logger.debug(f"Using cached games data ({cache_age:.0f}s old)")
            return list(_games_cache)

        # Thundering herd prevention: if another thread is refreshing, return stale cache
        if _games_refreshing and _games_cache:
            app.logger.debug("Another thread is refreshing games, returning stale cache")
            return list(_games_cache)

        # Mark that we're refreshing
        _games_refreshing = True

    # Fetch data outside lock to avoid blocking other threads
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

        # Update cache under lock
        with _games_lock:
            _games_cache = games
            _games_cache_time = time.time()
            _games_refreshing = False

        return games
    except Exception as e:
        app.logger.error(f"Error loading games: {e}")
        with _games_lock:
            _games_refreshing = False
            # Return cached data if available, even if expired
            if _games_cache:
                app.logger.info("Returning stale cached games data due to error")
                return list(_games_cache)
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
    return render_template("leaderboard.html", leaderboard=get_leaderboard_data(min_games=5))


@app.route("/leaderboard")
def leaderboard():
    """Show leaderboard page."""
    return render_template("leaderboard.html", leaderboard=get_leaderboard_data(min_games=5))


@app.route("/games")
def games():
    """Show game library."""
    GAMES_PER_PAGE = 10

    all_games = get_all_games()
    model_filter = request.args.get('model', '').strip()
    if len(model_filter) > 200:
        abort(400)

    # Validate page parameter
    try:
        page = int(request.args.get('page', 1))
        if page < 1:
            page = 1
    except (ValueError, TypeError):
        page = 1

    # Collect all unique models for the dropdown
    all_models = sorted(set(g['white'] for g in all_games) | set(g['black'] for g in all_games))

    if model_filter:
        filtered_games = [g for g in all_games if g['white'] == model_filter or g['black'] == model_filter]
    else:
        filtered_games = all_games

    # Calculate pagination
    total_games = len(filtered_games)
    total_pages = max(1, math.ceil(total_games / GAMES_PER_PAGE))

    # Clamp page to valid range
    if page > total_pages:
        page = total_pages

    # Slice games for current page
    start_idx = (page - 1) * GAMES_PER_PAGE
    end_idx = start_idx + GAMES_PER_PAGE
    paginated_games = filtered_games[start_idx:end_idx]

    return render_template(
        "games.html",
        filtered_games=paginated_games,
        all_models=all_models,
        model_filter=model_filter,
        page=page,
        total_pages=total_pages,
        total_games=total_games,
    )


@app.route("/methodology")
def methodology():
    """Show methodology page."""
    return render_template("methodology.html")


@app.route("/timeline")
def timeline():
    """Show timeline visualization."""
    leaderboard_data = get_leaderboard_data(min_games=5)
    chart_html = get_timeline_html(leaderboard_data)
    return render_template("timeline.html", chart_html=chart_html)


@app.route("/cost")
def cost():
    """Show cost vs rating visualization."""
    leaderboard_data = get_leaderboard_data(min_games=5)
    chart_html = get_cost_chart_html(leaderboard_data)
    return render_template("cost.html", chart_html=chart_html)


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
@app.route("/api/invalidate-cache", methods=["POST"])
def api_invalidate_cache():
    """Invalidate the leaderboard cache for ALL workers by touching signal file."""
    from rating.rating_store import invalidate_cache as touch_signal_file

    # Require secret token (header only, not query param to avoid log leakage)
    expected_token = os.environ.get("CACHE_INVALIDATE_TOKEN")
    if not expected_token:
        app.logger.error("CACHE_INVALIDATE_TOKEN not configured")
        abort(500)
    provided_token = request.headers.get("X-Cache-Token")
    if provided_token != expected_token:
        abort(403)

    # Touch the signal file - all workers share this filesystem and will see it
    touch_signal_file()
    app.logger.info("Cache invalidation signal file touched - all workers will refresh")
    return jsonify({"status": "ok", "message": "Cache invalidated for all workers"})


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
