#!/usr/bin/env python3
"""
CLI for the Chess LLM Benchmark.

Commands:
- run: Run the benchmark
- leaderboard: Show current leaderboard
- manual: Run a manual game
"""

import argparse
import asyncio
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import requests
import yaml

from engines.stockfish_engine import StockfishEngine
from engines.maia_engine import MaiaEngine
from engines.random_engine import RandomEngine
from engines.uci_engine import UCIEngine
from llm.openrouter_client import OpenRouterPlayer
from game.game_runner import GameRunner
from game.pgn_logger import PGNLogger
from game.stats_collector import StatsCollector
from game.match_scheduler import MatchScheduler
from utils import is_reasoning_model
from rating.glicko2 import Glicko2System, PlayerRating
from rating.rating_store import RatingStore, invalidate_cache
from rating.leaderboard import Leaderboard

# Starting ratings based on model type
REASONING_START_RATING = 1200
NON_REASONING_START_RATING = 400


def invalidate_remote_cache():
    """Invalidate cache on remote web server and locally."""
    # Local file-based invalidation (for local dev)
    invalidate_cache()

    # Remote API invalidation (for production)
    web_url = os.environ.get("WEB_APP_URL")
    if not web_url:
        return

    token = os.environ.get("CACHE_INVALIDATE_TOKEN")
    try:
        headers = {"X-Cache-Token": token} if token else {}
        resp = requests.post(f"{web_url}/api/invalidate-cache", headers=headers, timeout=10)
        if resp.status_code == 200:
            print(f"Remote cache invalidated: {web_url}")
        else:
            print(f"Warning: Failed to invalidate remote cache (HTTP {resp.status_code})")
    except requests.RequestException as e:
        print(f"Warning: Failed to invalidate remote cache: {e}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_engines(config: dict) -> dict:
    """Create engine players from config."""
    engines = {}

    for i, engine_cfg in enumerate(config.get("engines", [])):
        if "player_id" not in engine_cfg:
            raise ValueError(f"Engine {i+1} missing required 'player_id' field")
        if "rating" not in engine_cfg:
            raise ValueError(f"Engine '{engine_cfg['player_id']}' missing required 'rating' field")

        engine_type = engine_cfg.get("type", "stockfish")
        player_id = engine_cfg["player_id"]
        rating = engine_cfg["rating"]

        if engine_type == "stockfish":
            engines[player_id] = StockfishEngine(
                player_id=player_id,
                rating=rating,
                engine_path=engine_cfg.get("path", "stockfish"),
                skill_level=engine_cfg.get("skill_level", 20),
                move_time=engine_cfg.get("move_time"),
                nodes=engine_cfg.get("nodes"),
                depth=engine_cfg.get("depth"),
            )
        elif engine_type == "maia":
            engines[player_id] = MaiaEngine(
                player_id=player_id,
                rating=rating,
                lc0_path=engine_cfg.get("lc0_path", "lc0"),
                weights_path=engine_cfg.get("weights_path"),
                nodes=engine_cfg.get("nodes", 1),
            )
        elif engine_type == "random":
            engines[player_id] = RandomEngine(
                player_id=player_id,
                rating=rating,
                seed=engine_cfg.get("seed"),
            )
        elif engine_type == "uci":
            if "path" not in engine_cfg:
                raise ValueError(f"UCI engine '{player_id}' missing required 'path' field")
            engines[player_id] = UCIEngine(
                player_id=player_id,
                rating=rating,
                engine_path=engine_cfg["path"],
                move_time=engine_cfg.get("move_time"),
                nodes=engine_cfg.get("nodes"),
                depth=engine_cfg.get("depth"),
                initial_time=engine_cfg.get("initial_time"),
                increment=engine_cfg.get("increment"),
            )

    return engines


def create_llm_players(config: dict, api_key: str = None) -> tuple[dict, set]:
    """Create LLM players from config.

    Returns:
        Tuple of (players dict, set of reasoning model player IDs)
    """
    players = {}
    reasoning_ids = set()

    for llm_cfg in config.get("llms", []):
        player_id = llm_cfg["player_id"]
        model_name = llm_cfg["model_name"]
        reasoning_effort = llm_cfg.get("reasoning_effort")
        reasoning = llm_cfg.get("reasoning")  # None = not set, True = enable, False = disable

        # Validate no conflicting reasoning settings
        if reasoning is False and reasoning_effort is not None:
            raise ValueError(
                f"Model '{player_id}': reasoning=false conflicts with reasoning_effort={reasoning_effort}"
            )

        # Append reasoning effort to player_id if set and not already included
        if reasoning_effort and f"({reasoning_effort})" not in player_id:
            player_id = f"{player_id} ({reasoning_effort})"

        players[player_id] = OpenRouterPlayer(
            player_id=player_id,
            model_name=model_name,
            api_key=api_key,
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 0),
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            provider_order=llm_cfg.get("provider_order"),
            timeout=llm_cfg.get("timeout", 300),
        )

        # Track reasoning models:
        # 1. Has reasoning_effort set, OR
        # 2. Has reasoning=True, OR
        # 3. Matches naming convention (unless reasoning is explicitly False)
        if reasoning_effort is not None or reasoning is True:
            reasoning_ids.add(player_id)
        elif reasoning is not False and is_reasoning_model(player_id):
            reasoning_ids.add(player_id)

    return players, reasoning_ids


async def run_benchmark(args):
    """Run the benchmark."""
    # Load config
    config = load_config(args.config)

    # Get API key
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OpenRouter API key required. Set OPENROUTER_API_KEY or use --api-key")
        return 1

    # Create players
    engines = create_engines(config)
    llm_players, reasoning_ids = create_llm_players(config, api_key)
    all_players = {**engines, **llm_players}

    # Set up rating store with anchors
    anchor_ids = set(engines.keys())
    rating_store = RatingStore(path="data/ratings.json", anchor_ids=anchor_ids)

    # Initialize anchor ratings
    for engine_id, engine in engines.items():
        rating_store.set_anchor(engine_id, engine.rating)

    # Create components
    glicko = Glicko2System()
    pgn_logger = PGNLogger()
    stats_collector = StatsCollector()

    # Create scheduler
    scheduler = MatchScheduler(
        players=all_players,
        rating_store=rating_store,
        glicko=glicko,
        pgn_logger=pgn_logger,
        stats_collector=stats_collector,
        max_concurrent=config.get("benchmark", {}).get("max_concurrent", 4),
        max_moves=config.get("benchmark", {}).get("max_moves", 200),
        verbose=args.verbose,
        reasoning_ids=reasoning_ids,
    )

    try:
        # Run benchmark
        results = await scheduler.run_benchmark(
            llm_ids=list(llm_players.keys()),
            anchor_ids=list(engines.keys()),
            games_vs_anchor_per_color=config.get("benchmark", {}).get("games_vs_anchor_per_color", 10),
            games_vs_llm_per_color=config.get("benchmark", {}).get("games_vs_llm_per_color", 5),
            rating_threshold=config.get("benchmark", {}).get("rating_threshold"),
        )

        # Show leaderboard
        print("\n" + "=" * 85)
        print("FINAL LEADERBOARD")
        print("=" * 85)
        leaderboard = Leaderboard(rating_store, stats_collector)
        print(leaderboard.format_table(min_games=1))

    finally:
        # Clean up
        for engine in engines.values():
            engine.close()
        for llm in llm_players.values():
            await llm.close()

    # Invalidate web cache so leaderboard refreshes
    invalidate_remote_cache()

    return 0


async def show_leaderboard(args):
    """Show current leaderboard."""
    rating_store = RatingStore(path="data/ratings.json")

    # Load results for stats
    pgn_logger = PGNLogger()
    stats_collector = StatsCollector()
    stats_collector.add_results(pgn_logger.load_all_results())

    leaderboard = Leaderboard(rating_store, stats_collector)
    print(leaderboard.format_table(min_games=args.min_games, sort_by=args.sort))

    return 0


async def recalculate_ratings(args):
    """Recalculate ratings from stored game results."""
    from datetime import datetime

    # Load config for anchors
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        return 1
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in config file: {e}")
        return 1

    # Build anchor map from config
    anchors = {}
    for i, engine_cfg in enumerate(config.get("engines", [])):
        if "player_id" not in engine_cfg:
            print(f"Error: Engine {i+1} missing required 'player_id' field")
            return 1
        if "rating" not in engine_cfg:
            print(f"Error: Engine '{engine_cfg['player_id']}' missing required 'rating' field")
            return 1
        anchors[engine_cfg["player_id"]] = engine_cfg["rating"]

    if not anchors:
        print("Warning: No anchors (engines) defined in config")

    if args.verbose:
        print(f"Anchors: {anchors}")

    # Load all results
    pgn_logger = PGNLogger()
    results = pgn_logger.load_all_results(verbose=args.verbose)

    if not results:
        print("No game results found in data/results/")
        return 1

    # Sort by creation time for chronological processing (using datetime for robustness)
    invalid_timestamps = []

    def parse_timestamp(r):
        try:
            # Handle ISO format with timezone
            ts = r.created_at
            if ts:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass
        invalid_timestamps.append(r.game_id)
        return datetime.min  # Put invalid timestamps first

    results.sort(key=parse_timestamp)

    if invalid_timestamps and args.verbose:
        print(f"Warning: {len(invalid_timestamps)} result(s) with invalid timestamps (sorted to beginning)")

    if args.verbose:
        print(f"Found {len(results)} game results")

    # Initialize rating store with anchors
    rating_store = RatingStore(path="data/ratings.json", anchor_ids=set(anchors.keys()))

    # Always reset when recalculating to avoid double-counting
    rating_store.reset()
    if args.verbose:
        print("Reset existing ratings")

    # Set anchor ratings (batch save)
    for anchor_id, rating in anchors.items():
        rating_store.set_anchor(anchor_id, rating, auto_save=False)
    rating_store.save()

    glicko = Glicko2System()
    skipped = 0

    # Filter valid games first
    valid_games = []
    for result in results:
        if not result.white_id or not result.black_id:
            if args.verbose:
                print(f"Skipping: missing player ID")
            skipped += 1
            continue

        if result.white_id == result.black_id:
            if args.verbose:
                print(f"Skipping: player vs themselves")
            skipped += 1
            continue

        if result.winner not in ("white", "black", "draw"):
            print(f"Skipping: invalid winner value '{result.winner}'")
            skipped += 1
            continue

        # Determine scores
        if result.winner == "white":
            white_score, black_score = 1.0, 0.0
        elif result.winner == "black":
            white_score, black_score = 0.0, 1.0
        else:
            white_score, black_score = 0.5, 0.5

        valid_games.append({
            'white_id': result.white_id,
            'black_id': result.black_id,
            'white_score': white_score,
            'black_score': black_score,
        })

    if not valid_games:
        print("No valid games to process")
        return 1

    # Rating period configuration
    BATCH_SIZE = 100  # Games per rating period
    random.seed(42)  # Fixed seed for reproducible results

    # Count actual games and W-L-D per player, get all unique player IDs
    all_players = set()
    actual_game_counts = {}
    actual_wld = {}  # {player_id: {'wins': N, 'losses': N, 'draws': N}}
    for game in valid_games:
        all_players.add(game['white_id'])
        all_players.add(game['black_id'])
        actual_game_counts[game['white_id']] = actual_game_counts.get(game['white_id'], 0) + 1
        actual_game_counts[game['black_id']] = actual_game_counts.get(game['black_id'], 0) + 1

        # Initialize W-L-D if needed
        if game['white_id'] not in actual_wld:
            actual_wld[game['white_id']] = {'wins': 0, 'losses': 0, 'draws': 0}
        if game['black_id'] not in actual_wld:
            actual_wld[game['black_id']] = {'wins': 0, 'losses': 0, 'draws': 0}

        # Track W-L-D from scores
        if game['white_score'] == 1.0:
            actual_wld[game['white_id']]['wins'] += 1
            actual_wld[game['black_id']]['losses'] += 1
        elif game['white_score'] == 0.0:
            actual_wld[game['white_id']]['losses'] += 1
            actual_wld[game['black_id']]['wins'] += 1
        else:  # draw
            actual_wld[game['white_id']]['draws'] += 1
            actual_wld[game['black_id']]['draws'] += 1

    # Pre-initialize all non-anchor players with appropriate starting ratings
    reasoning_count = 0
    non_reasoning_count = 0
    for player_id in all_players:
        if not rating_store.is_anchor(player_id):
            if is_reasoning_model(player_id):
                start_rating = REASONING_START_RATING
                reasoning_count += 1
            else:
                start_rating = NON_REASONING_START_RATING
                non_reasoning_count += 1
            rating_store.set(PlayerRating(
                player_id=player_id,
                rating=start_rating,
                rating_deviation=350.0,
                volatility=0.06,
            ), auto_save=False)
    rating_store.save()
    if args.verbose:
        print(f"Initialized {reasoning_count} reasoning models at {REASONING_START_RATING}, "
              f"{non_reasoning_count} non-reasoning models at {NON_REASONING_START_RATING}")

    # Split games into anchor games (calibration) and LLM-only games
    anchor_games = []
    llm_games = []
    for game in valid_games:
        if rating_store.is_anchor(game['white_id']) or rating_store.is_anchor(game['black_id']):
            anchor_games.append(game)
        else:
            llm_games.append(game)

    # Shuffle games within each category for fairness
    random.shuffle(anchor_games)
    random.shuffle(llm_games)

    # Multi-pass convergence settings
    max_passes = 100
    convergence_threshold = 30.0  # Stop when no rating changes by more than this

    print(f"Processing {len(valid_games)} games in rating periods (batch size: {BATCH_SIZE})")
    print(f"  Anchor games: {len(anchor_games)} (calibration phase)")
    print(f"  LLM vs LLM games: {len(llm_games)}")
    print(f"  Max passes: {max_passes}, convergence threshold: {convergence_threshold}")

    def process_batch(batch_games):
        """Process a batch of games as a single rating period."""
        if not batch_games:
            return

        # Snapshot current ratings for this period
        period_ratings = {pid: rating_store.get(pid) for pid in all_players}

        # Collect games per player
        player_games = defaultdict(lambda: {'opponents': [], 'scores': []})

        for game in batch_games:
            white_id, black_id = game['white_id'], game['black_id']

            if not rating_store.is_anchor(white_id):
                player_games[white_id]['opponents'].append(period_ratings[black_id])
                player_games[white_id]['scores'].append(game['white_score'])

            if not rating_store.is_anchor(black_id):
                player_games[black_id]['opponents'].append(period_ratings[white_id])
                player_games[black_id]['scores'].append(game['black_score'])

        # Update each player with their games from this period
        for player_id, games in player_games.items():
            player = period_ratings[player_id]
            new_player = glicko.update_rating(player, games['opponents'], games['scores'])
            rating_store.set(new_player, auto_save=False)

    def run_rating_periods():
        """Run one full pass of rating periods (anchor games first, then LLM games)."""
        # Phase 1: Process ALL anchor games as single rating period (calibration)
        # This ensures all LLMs are calibrated simultaneously against anchors
        process_batch(anchor_games)

        # Phase 2: Process LLM vs LLM games in batches
        for i in range(0, len(llm_games), BATCH_SIZE):
            batch = llm_games[i:i + BATCH_SIZE]
            process_batch(batch)

    # Multi-pass convergence loop
    for pass_num in range(1, max_passes + 1):
        # Store ratings at start of pass to check convergence
        pass_start_ratings = {pid: rating_store.get(pid).rating for pid in all_players}

        # Run all rating periods for this pass
        run_rating_periods()
        rating_store.save()

        # Check convergence
        max_change = 0.0
        for pid in all_players:
            if not rating_store.is_anchor(pid):
                old_rating = pass_start_ratings[pid]
                new_rating = rating_store.get(pid).rating
                change = abs(new_rating - old_rating)
                max_change = max(max_change, change)

        print(f"Pass {pass_num}: max rating change = {max_change:.1f}")

        if pass_num > 1 and max_change < convergence_threshold:
            print(f"Converged after {pass_num} passes (max change {max_change:.1f} < {convergence_threshold})")
            break

    # Fix game counts and W-L-D to actual values (multi-pass inflates them for non-anchors)
    for pid in all_players:
        player = rating_store.get(pid)
        player.games_played = actual_game_counts.get(pid, 0)
        wld = actual_wld.get(pid, {'wins': 0, 'losses': 0, 'draws': 0})
        player.wins = wld['wins']
        player.losses = wld['losses']
        player.draws = wld['draws']
        rating_store.set(player, auto_save=False)
    rating_store.save()

    processed = len(valid_games)
    print(f"\nProcessed {processed} games" + (f" ({skipped} skipped)" if skipped else ""))
    print()

    # Show leaderboard
    stats_collector = StatsCollector()
    stats_collector.add_results(results)
    leaderboard = Leaderboard(rating_store, stats_collector)
    print(leaderboard.format_table(min_games=1))

    # Invalidate web cache so leaderboard refreshes
    invalidate_remote_cache()

    return 0


async def run_manual_game(args):
    """Run manual game(s)."""
    # Validate arguments
    if args.games < 1:
        print("Error: --games must be at least 1")
        return 1

    # Get API key
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OpenRouter API key required. Set OPENROUTER_API_KEY or use --api-key")
        return 1

    # Helper to create engine based on type
    def create_engine(engine_type):
        # Maia engine configurations
        maia_configs = {
            "maia-1100": {
                "weights": "maia-1100.pb.gz",
                "rating": 1628,
            },
            "maia-1900": {
                "weights": "maia-1900.pb.gz",
                "rating": 1816,
            },
        }

        if engine_type in maia_configs:
            config = maia_configs[engine_type]
            weights_path = Path(__file__).parent / config["weights"]
            return MaiaEngine(
                player_id=engine_type,
                rating=config["rating"],
                lc0_path=args.lc0_path,
                weights_path=str(weights_path),
                nodes=1,
            )
        elif engine_type == "random":
            return RandomEngine(
                player_id="random-bot",
                rating=400,
            )
        elif engine_type == "eubos":
            return UCIEngine(
                player_id="eubos",
                rating=2344,
                engine_path="/Volumes/MainStorage/Programming/EubosChess/eubos.sh",
                initial_time=900,  # 15 minutes
                increment=10,      # 10 seconds
            )
        else:
            return StockfishEngine(
                player_id="stockfish-test",
                rating=1500,
                skill_level=args.stockfish_skill,
            )

    # Helper to create LLM player
    def create_llm(model_name, reasoning_effort=None, custom_name=None):
        if custom_name:
            player_id = custom_name
        else:
            player_id = model_name.split("/")[-1]
            if reasoning_effort and f"({reasoning_effort})" not in player_id:
                player_id = f"{player_id} ({reasoning_effort})"
        return OpenRouterPlayer(
            player_id=player_id,
            model_name=model_name,
            api_key=api_key,
            max_tokens=args.max_tokens,
            reasoning=args.reasoning,
            reasoning_effort=reasoning_effort,
        )

    # Track results across games
    results_summary = {"white": 0, "black": 0, "draw": 0}
    total_illegal_white = 0
    total_illegal_black = 0
    api_error_count = 0
    pgn_logger = PGNLogger() if args.save else None

    try:
        for game_num in range(args.games):
            # Alternate colors if playing multiple games
            swap_colors = (game_num % 2 == 1) and args.games > 1

            # Create players for this game
            if swap_colors:
                # Swapped: original black config plays white, original white config plays black
                if args.black_engine:
                    white = create_engine(args.engine_type)
                else:
                    white = create_llm(args.black_model, args.black_reasoning_effort, args.black_name)

                if args.white_engine:
                    black = create_engine(args.engine_type)
                else:
                    black = create_llm(args.white_model, args.white_reasoning_effort, args.white_name)
            else:
                # Normal: original assignments
                if args.white_engine:
                    white = create_engine(args.engine_type)
                else:
                    white = create_llm(args.white_model, args.white_reasoning_effort, args.white_name)

                if args.black_engine:
                    black = create_engine(args.engine_type)
                else:
                    black = create_llm(args.black_model, args.black_reasoning_effort, args.black_name)

            if args.games > 1:
                print(f"\n{'='*50}")
                print(f"Game {game_num + 1}/{args.games}: {white.player_id} vs {black.player_id}")
                print("=" * 50)
            else:
                print(f"Manual game: {white.player_id} vs {black.player_id}")
            print()

            runner = GameRunner(
                white=white,
                black=black,
                max_moves=args.max_moves,
                verbose=True,
            )

            try:
                result, pgn_str = await runner.play_game()

                print()
                print("-" * 50)
                print(f"Result: {result.winner} ({result.termination})")
                print(f"Moves: {result.moves}")
                print(f"Illegal moves - White: {result.illegal_moves_white}, Black: {result.illegal_moves_black}")

                # Don't count or save games that ended due to API errors
                if result.termination == "api_error":
                    print("API error - game not saved or counted")
                    api_error_count += 1
                    continue

                # Track results
                results_summary[result.winner] += 1
                total_illegal_white += result.illegal_moves_white
                total_illegal_black += result.illegal_moves_black

                if args.games == 1:
                    print()
                    print("PGN:")
                    print(pgn_str)

                # Save if requested
                if pgn_logger:
                    saved_result = pgn_logger.save_game(result, pgn_str)
                    print(f"Saved to: {saved_result.pgn_path}")

            finally:
                # Close players after each game
                if hasattr(white, "close"):
                    if asyncio.iscoroutinefunction(white.close):
                        await white.close()
                    else:
                        white.close()
                if hasattr(black, "close"):
                    if asyncio.iscoroutinefunction(black.close):
                        await black.close()
                    else:
                        black.close()

        # Print summary if multiple games
        if args.games > 1:
            games_completed = sum(results_summary.values())
            print()
            print("=" * 50)
            print("SUMMARY")
            print("=" * 50)
            print(f"Games completed: {games_completed}/{args.games}")
            if api_error_count > 0:
                print(f"API errors: {api_error_count} (not saved)")
            print(f"White wins: {results_summary['white']}")
            print(f"Black wins: {results_summary['black']}")
            print(f"Draws: {results_summary['draw']}")
            print(f"Total illegal moves - White: {total_illegal_white}, Black: {total_illegal_black}")

        # Invalidate web cache if games were saved
        if pgn_logger and sum(results_summary.values()) > 0:
            invalidate_remote_cache()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if args.games > 1:
            games_completed = sum(results_summary.values())
            print(f"Games completed: {games_completed}/{args.games}")
            if api_error_count > 0:
                print(f"API errors: {api_error_count} (not saved)")
            print(f"White wins: {results_summary['white']}, Black wins: {results_summary['black']}, Draws: {results_summary['draw']}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Chess LLM Benchmark")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run benchmark command
    run_parser = subparsers.add_parser("run", help="Run the benchmark")
    run_parser.add_argument(
        "--config", "-c",
        default="config/benchmark.yaml",
        help="Path to benchmark config file",
    )
    run_parser.add_argument(
        "--api-key",
        help="OpenRouter API key (or set OPENROUTER_API_KEY)",
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    # Leaderboard command
    lb_parser = subparsers.add_parser("leaderboard", help="Show leaderboard")
    lb_parser.add_argument(
        "--min-games",
        type=int,
        default=1,
        help="Minimum games to include",
    )
    lb_parser.add_argument(
        "--sort",
        choices=["rating", "legal", "cost"],
        default="rating",
        help="Sort by: rating (default), legal (legal move %%), cost ($/game)",
    )

    # Recalculate command
    recalc_parser = subparsers.add_parser("recalculate", help="Recalculate ratings from stored results")
    recalc_parser.add_argument(
        "--config", "-c",
        default="config/benchmark.yaml",
        help="Path to config file (for anchor definitions)",
    )
    recalc_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show each game as it's processed",
    )

    # Manual game command
    manual_parser = subparsers.add_parser("manual", help="Run a manual game")
    manual_parser.add_argument(
        "--white-model",
        default="meta-llama/llama-4-maverick",
        help="White player model (OpenRouter)",
    )
    manual_parser.add_argument(
        "--black-model",
        default="deepseek/deepseek-chat-v3-0324",
        help="Black player model (OpenRouter)",
    )
    manual_parser.add_argument(
        "--white-engine",
        action="store_true",
        help="Use engine as white",
    )
    manual_parser.add_argument(
        "--black-engine",
        action="store_true",
        help="Use engine as black",
    )
    manual_parser.add_argument(
        "--engine-type",
        choices=["stockfish", "maia-1100", "maia-1900", "random", "eubos"],
        default="stockfish",
        help="Engine type to use (stockfish, maia-1100, maia-1900, random, or eubos)",
    )
    manual_parser.add_argument(
        "--stockfish-skill",
        type=int,
        default=5,
        help="Stockfish skill level (0-20)",
    )
    manual_parser.add_argument(
        "--lc0-path",
        default="/opt/homebrew/bin/lc0",
        help="Path to lc0 executable",
    )
    manual_parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Max tokens for LLM response (0 = no limit, recommended for reasoning models)",
    )
    reasoning_group = manual_parser.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--reasoning",
        dest="reasoning",
        action="store_const",
        const=True,
        help="Enable reasoning mode for hybrid models",
    )
    reasoning_group.add_argument(
        "--no-reasoning",
        dest="reasoning",
        action="store_const",
        const=False,
        help="Explicitly disable reasoning mode (for thinking models run without thinking)",
    )
    manual_parser.set_defaults(reasoning=None)  # Explicit three-state default
    manual_parser.add_argument(
        "--white-reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort level for white (minimal, low, medium, high)",
    )
    manual_parser.add_argument(
        "--black-reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort level for black (minimal, low, medium, high)",
    )
    manual_parser.add_argument(
        "--white-name",
        help="Custom display name for white LLM player",
    )
    manual_parser.add_argument(
        "--black-name",
        help="Custom display name for black LLM player",
    )
    manual_parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum moves per game",
    )
    manual_parser.add_argument(
        "--api-key",
        help="OpenRouter API key",
    )
    manual_parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save",
        help="Don't save the game (saves by default)",
    )
    manual_parser.set_defaults(save=True)
    manual_parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of games to play (alternates colors if > 1)",
    )

    args = parser.parse_args()

    if args.command == "run":
        return asyncio.run(run_benchmark(args))
    elif args.command == "leaderboard":
        return asyncio.run(show_leaderboard(args))
    elif args.command == "recalculate":
        return asyncio.run(recalculate_ratings(args))
    elif args.command == "manual":
        return asyncio.run(run_manual_game(args))
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
