#!/usr/bin/env python3
"""
CLI for the Chess LLM Benchmark.

Commands:
- run: Run the benchmark
- leaderboard: Show current leaderboard
- test: Run a test game
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml

from engines.stockfish_engine import StockfishEngine
from engines.maia_engine import MaiaEngine
from llm.openrouter_client import OpenRouterPlayer
from game.game_runner import GameRunner
from game.pgn_logger import PGNLogger
from game.stats_collector import StatsCollector
from game.match_scheduler import MatchScheduler
from rating.glicko2 import Glicko2System
from rating.rating_store import RatingStore
from rating.leaderboard import Leaderboard


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_engines(config: dict) -> dict:
    """Create engine players from config."""
    engines = {}

    for engine_cfg in config.get("engines", []):
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

    return engines


def create_llm_players(config: dict, api_key: str = None) -> dict:
    """Create LLM players from config."""
    players = {}

    for llm_cfg in config.get("llms", []):
        player_id = llm_cfg["player_id"]
        model_name = llm_cfg["model_name"]

        players[player_id] = OpenRouterPlayer(
            player_id=player_id,
            model_name=model_name,
            api_key=api_key,
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 10),
        )

    return players


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
    llm_players = create_llm_players(config, api_key)
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
    )

    try:
        # Run benchmark
        results = await scheduler.run_benchmark(
            llm_ids=list(llm_players.keys()),
            anchor_ids=list(engines.keys()),
            games_vs_anchor_per_color=config.get("benchmark", {}).get("games_vs_anchor_per_color", 10),
            games_vs_llm_per_color=config.get("benchmark", {}).get("games_vs_llm_per_color", 5),
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

    return 0


async def show_leaderboard(args):
    """Show current leaderboard."""
    rating_store = RatingStore(path="data/ratings.json")

    # Load results for stats
    pgn_logger = PGNLogger()
    stats_collector = StatsCollector()
    stats_collector.add_results(pgn_logger.load_all_results())

    leaderboard = Leaderboard(rating_store, stats_collector)
    print(leaderboard.format_table(min_games=args.min_games))

    return 0


async def run_test_game(args):
    """Run a single test game."""
    # Get API key
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OpenRouter API key required. Set OPENROUTER_API_KEY or use --api-key")
        return 1

    # Helper to create engine based on type
    def create_engine(engine_type):
        if engine_type == "maia":
            return MaiaEngine(
                player_id="maia-1100",
                rating=1628,
                lc0_path=args.lc0_path,
                weights_path=args.maia_weights,
                nodes=1,
            )
        else:
            return StockfishEngine(
                player_id="stockfish-test",
                rating=1500,
                skill_level=args.stockfish_skill,
            )

    # Create players
    if args.white_engine:
        white = create_engine(args.engine_type)
    else:
        white = OpenRouterPlayer(
            player_id=args.white_model.split("/")[-1],
            model_name=args.white_model,
            api_key=api_key,
        )

    if args.black_engine:
        black = create_engine(args.engine_type)
    else:
        black = OpenRouterPlayer(
            player_id=args.black_model.split("/")[-1],
            model_name=args.black_model,
            api_key=api_key,
        )

    print(f"Test game: {white.player_id} vs {black.player_id}")
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
        print("=" * 50)
        print(f"Result: {result.winner} ({result.termination})")
        print(f"Moves: {result.moves}")
        print(f"Illegal moves - White: {result.illegal_moves_white}, Black: {result.illegal_moves_black}")
        print()
        print("PGN:")
        print(pgn_str)

        # Save if requested
        if args.save:
            pgn_logger = PGNLogger()
            result = pgn_logger.save_game(result, pgn_str)
            print(f"\nSaved to: {result.pgn_path}")

    finally:
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

    # Test game command
    test_parser = subparsers.add_parser("test", help="Run a test game")
    test_parser.add_argument(
        "--white-model",
        default="meta-llama/llama-4-maverick",
        help="White player model (OpenRouter)",
    )
    test_parser.add_argument(
        "--black-model",
        default="deepseek/deepseek-chat-v3-0324",
        help="Black player model (OpenRouter)",
    )
    test_parser.add_argument(
        "--white-engine",
        action="store_true",
        help="Use engine as white",
    )
    test_parser.add_argument(
        "--black-engine",
        action="store_true",
        help="Use engine as black",
    )
    test_parser.add_argument(
        "--engine-type",
        choices=["stockfish", "maia"],
        default="stockfish",
        help="Engine type to use (stockfish or maia)",
    )
    test_parser.add_argument(
        "--stockfish-skill",
        type=int,
        default=5,
        help="Stockfish skill level (0-20)",
    )
    test_parser.add_argument(
        "--maia-weights",
        default="/Volumes/MainStorage/Programming/create_chess_puzzles/maia-1100.pb",
        help="Path to Maia weights file",
    )
    test_parser.add_argument(
        "--lc0-path",
        default="/opt/homebrew/bin/lc0",
        help="Path to lc0 executable",
    )
    test_parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum moves per game",
    )
    test_parser.add_argument(
        "--api-key",
        help="OpenRouter API key",
    )
    test_parser.add_argument(
        "--save",
        action="store_true",
        help="Save the game",
    )

    args = parser.parse_args()

    if args.command == "run":
        return asyncio.run(run_benchmark(args))
    elif args.command == "leaderboard":
        return asyncio.run(show_leaderboard(args))
    elif args.command == "test":
        return asyncio.run(run_test_game(args))
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
