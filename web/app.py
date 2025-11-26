"""
Flask web application for LLM Chess Benchmark.

Displays leaderboard and game library with PGN viewer.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, abort

from rating.rating_store import RatingStore
from rating.leaderboard import Leaderboard
from game.pgn_logger import PGNLogger
from game.stats_collector import StatsCollector

app = Flask(__name__)

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
RATINGS_PATH = DATA_DIR / "ratings.json"


def get_leaderboard_data(min_games: int = 1) -> list:
    """Get leaderboard data from rating store."""
    rating_store = RatingStore(path=str(RATINGS_PATH))
    pgn_logger = PGNLogger()
    stats_collector = StatsCollector()
    stats_collector.add_results(pgn_logger.load_all_results())

    leaderboard = Leaderboard(rating_store, stats_collector)
    return leaderboard.get_leaderboard(min_games=min_games)


def get_all_games() -> list:
    """Get all games with metadata."""
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
            "moves": result.moves,
            "illegal_moves_white": result.illegal_moves_white,
            "illegal_moves_black": result.illegal_moves_black,
            "created_at": result.created_at,
        })

    return games


def get_game(game_id: str) -> dict | None:
    """Get a single game with PGN."""
    pgn_logger = PGNLogger()
    result = pgn_logger.load_result(game_id)

    if not result:
        return None

    pgn = pgn_logger.load_pgn(game_id)

    return {
        "game_id": result.game_id,
        "white": result.white_id,
        "black": result.black_id,
        "winner": result.winner,
        "termination": result.termination,
        "moves": result.moves,
        "illegal_moves_white": result.illegal_moves_white,
        "illegal_moves_black": result.illegal_moves_black,
        "created_at": result.created_at,
        "pgn": pgn,
    }


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
    return render_template("games.html", games=get_all_games())


@app.route("/game/<game_id>")
def game(game_id: str):
    """Show individual game viewer."""
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
    game_data = get_game(game_id)
    if not game_data:
        abort(404)
    return jsonify(game_data)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
