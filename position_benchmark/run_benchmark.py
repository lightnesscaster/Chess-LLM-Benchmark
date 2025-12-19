"""
Run position benchmark on models.

Tests models on blunder positions and measures CPL (centipawn loss).
"""

import asyncio
import json
import sys
import chess
import chess.engine
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import argparse
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.openrouter_client import OpenRouterPlayer
from llm.prompts import board_to_ascii
from engines.stockfish_engine import StockfishEngine
from engines.maia_engine import MaiaEngine
from engines.random_engine import RandomEngine


@dataclass
class PositionResult:
    """Result for a single position."""
    position_idx: int
    fen: str
    model_move: str
    model_move_san: str
    best_move: str
    best_move_san: str
    blunder_move: str  # What was originally played (the bad move)
    cpl: float  # Centipawn loss of model's move
    is_legal: bool
    is_best: bool
    avoided_blunder: bool  # Did model avoid the original blunder?
    eval_model: float  # Eval after model's move
    eval_best: float  # Eval after best move


def _calculate_median(values: list[float]) -> float:
    """Calculate median, properly handling even-length lists."""
    if not values:
        return 10000
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    else:
        return sorted_vals[n // 2]


def build_position_prompt(fen: str, move_history: list[str], side_to_move: str) -> str:
    """Build prompt for a position."""
    board = chess.Board(fen)
    ascii_board = board_to_ascii(board)

    # Format move history
    if move_history:
        moves_str = []
        for i, move in enumerate(move_history):
            if i % 2 == 0:
                move_num = (i // 2) + 1
                moves_str.append(f"{move_num}. {move}")
            else:
                moves_str.append(move)
        history_section = f"Move history:\n{' '.join(moves_str)}"
    else:
        history_section = "Move history:\n(Game just started - no moves yet)"

    # Get last move info
    if move_history:
        last_move = move_history[-1]
        last_player = "Black" if side_to_move == "White" else "White"
        last_move_section = f"\n>>> {last_player}'s last move: {last_move} <<<\n"
    else:
        last_move_section = ""

    return f"""You are playing chess as {side_to_move}.

{history_section}
{last_move_section}
Current position (FEN):
{fen}

Board:
{ascii_board}

Your task:
- Play exactly ONE legal move for {side_to_move}.
- Use UCI notation only (examples: e2e4, g1f3, e7e8q for promotion).
- Do NOT include any commentary, explanations, or additional text.

Output format:
- Only the move in UCI, e.g.:
b1c3"""


def eval_to_cp(info: chess.engine.InfoDict, perspective: chess.Color) -> float:
    """Convert engine info to centipawns from perspective's view."""
    score = info.get("score")
    if score is None:
        return 0.0

    pov_score = score.pov(perspective)

    if pov_score.is_mate():
        mate_in = pov_score.mate()
        if mate_in > 0:
            return 10000 - mate_in * 10
        else:
            return -10000 - mate_in * 10

    cp = pov_score.score()
    return cp if cp is not None else 0.0


async def test_llm_on_position(
    player: OpenRouterPlayer,
    position: dict,
    engine: chess.engine.SimpleEngine,
    depth: int = 16,
) -> PositionResult:
    """Test an LLM on a single position."""
    fen = position["fen"]
    move_history = position["move_history"]
    side = position["side_to_move"].capitalize()
    best_move_uci = position["best_move"]
    blunder_move_uci = position["blunder_move"]

    board = chess.Board(fen)
    perspective = board.turn

    # Build prompt and get model's move
    prompt = build_position_prompt(fen, move_history, side)

    # Calculate illegal move CPL: half the swing to losing (eval_before + 5000)
    # This represents "half a game loss" since 2 illegals = forfeit
    eval_before = position["eval_before"]
    illegal_cpl = eval_before + 5000

    try:
        model_move_uci = await player.select_move(board, is_retry=False)
    except Exception as e:
        # Model failed to respond
        return PositionResult(
            position_idx=0,
            fen=fen,
            model_move="",
            model_move_san="",
            best_move=best_move_uci,
            best_move_san=position["best_move_san"],
            blunder_move=blunder_move_uci,
            cpl=illegal_cpl,
            is_legal=False,
            is_best=False,
            avoided_blunder=True,  # Technically didn't play the blunder
            eval_model=-5000,  # Halfway to losing
            eval_best=eval_before,
        )

    # Check if move is legal
    try:
        move = chess.Move.from_uci(model_move_uci)
        if move not in board.legal_moves:
            raise ValueError("Illegal move")
        model_move_san = board.san(move)
        is_legal = True
    except:
        return PositionResult(
            position_idx=0,
            fen=fen,
            model_move=model_move_uci,
            model_move_san="",
            best_move=best_move_uci,
            best_move_san=position["best_move_san"],
            blunder_move=blunder_move_uci,
            cpl=illegal_cpl,
            is_legal=False,
            is_best=False,
            avoided_blunder=model_move_uci != blunder_move_uci,
            eval_model=-5000,
            eval_best=eval_before,
        )

    # Evaluate model's move
    board.push(move)
    info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
    eval_model = -eval_to_cp(info_after, not perspective)  # Negate for side that moved
    board.pop()

    # Get best move eval
    eval_best = position["eval_before"]

    # Calculate CPL
    cpl = max(0, eval_best - eval_model)

    return PositionResult(
        position_idx=0,
        fen=fen,
        model_move=model_move_uci,
        model_move_san=model_move_san,
        best_move=best_move_uci,
        best_move_san=position["best_move_san"],
        blunder_move=blunder_move_uci,
        cpl=cpl,
        is_legal=True,
        is_best=model_move_uci == best_move_uci,
        avoided_blunder=model_move_uci != blunder_move_uci,
        eval_model=eval_model,
        eval_best=eval_best,
    )


def test_engine_on_position(
    engine_player,
    position: dict,
    stockfish: chess.engine.SimpleEngine,
    depth: int = 16,
) -> PositionResult:
    """Test an engine on a single position."""
    fen = position["fen"]
    best_move_uci = position["best_move"]
    blunder_move_uci = position["blunder_move"]

    board = chess.Board(fen)
    perspective = board.turn

    # Get engine's move
    try:
        move = engine_player.select_move(board)
        model_move_uci = move.uci()
        model_move_san = board.san(move)
        is_legal = True
    except Exception as e:
        return PositionResult(
            position_idx=0,
            fen=fen,
            model_move="",
            model_move_san="",
            best_move=best_move_uci,
            best_move_san=position["best_move_san"],
            blunder_move=blunder_move_uci,
            cpl=10000,
            is_legal=False,
            is_best=False,
            avoided_blunder=True,
            eval_model=-10000,
            eval_best=position["eval_before"],
        )

    # Evaluate engine's move
    board.push(move)
    info_after = stockfish.analyse(board, chess.engine.Limit(depth=depth))
    eval_model = -eval_to_cp(info_after, not perspective)
    board.pop()

    eval_best = position["eval_before"]
    cpl = max(0, eval_best - eval_model)

    return PositionResult(
        position_idx=0,
        fen=fen,
        model_move=model_move_uci,
        model_move_san=model_move_san,
        best_move=best_move_uci,
        best_move_san=position["best_move_san"],
        blunder_move=blunder_move_uci,
        cpl=cpl,
        is_legal=True,
        is_best=model_move_uci == best_move_uci,
        avoided_blunder=model_move_uci != blunder_move_uci,
        eval_model=eval_model,
        eval_best=eval_best,
    )


async def run_benchmark(
    player_id: str,
    positions: list[dict],
    stockfish: chess.engine.SimpleEngine,
    player_config: dict,
    depth: int = 16,
) -> dict:
    """Run benchmark for a single player."""
    results = []

    is_engine = player_config.get("type") in ["stockfish", "maia", "random", "uci", "survival"]

    if is_engine:
        # Create engine player
        engine_type = player_config.get("type")
        if engine_type == "maia":
            player = MaiaEngine(
                player_id=player_id,
                rating=player_config.get("rating", 1500),
                lc0_path=player_config.get("lc0_path"),
                weights_path=player_config.get("weights_path"),
                nodes=player_config.get("nodes", 1),
            )
        elif engine_type == "random":
            player = RandomEngine(player_id=player_id, rating=400)
        elif engine_type == "uci":
            from engines.uci_engine import UCIEngine
            player = UCIEngine(
                player_id=player_id,
                rating=player_config.get("rating", 2000),
                engine_path=player_config.get("path"),
            )
        elif engine_type == "survival":
            from engines.survival_engine import SurvivalEngine
            player = SurvivalEngine(
                player_id=player_id,
                rating=player_config.get("rating", 1200),
                opening_book_path=player_config.get("opening_book_path"),
                book_draw_threshold=player_config.get("book_draw_threshold", 0.10),
                base_depth=player_config.get("base_depth", 12),
                blunder_threshold=player_config.get("blunder_threshold", 3.0),
            )
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

        for i, pos in enumerate(positions):
            print(f"  [{i+1}/{len(positions)}] ", end="", flush=True)
            result = test_engine_on_position(player, pos, stockfish, depth)
            result.position_idx = i
            results.append(result)
            print(f"CPL: {result.cpl:.0f}" + (" (best!)" if result.is_best else ""))

        # Close engine if needed
        if hasattr(player, 'close'):
            player.close()
    else:
        # Create LLM player
        # Note: use .get("reasoning") without default to allow None (vs explicit False)
        player = OpenRouterPlayer(
            player_id=player_id,
            model_name=player_config.get("model_name"),
            temperature=player_config.get("temperature", 0.0),
            reasoning=player_config.get("reasoning"),
            reasoning_effort=player_config.get("reasoning_effort"),
        )

        # Run positions in parallel (10 at a time)
        max_concurrent = 10
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = [0]  # Use list to allow modification in nested function
        results = [None] * len(positions)  # Pre-allocate to maintain order

        async def test_with_semaphore(idx: int, pos: dict):
            async with semaphore:
                result = await test_llm_on_position(player, pos, stockfish, depth)
                result.position_idx = idx
                results[idx] = result
                completed[0] += 1
                status = "illegal" if not result.is_legal else f"CPL: {result.cpl:.0f}"
                if result.is_best:
                    status += " (best!)"
                print(f"  [{completed[0]}/{len(positions)}] {status}")
                return result

        # Launch all tasks
        tasks = [test_with_semaphore(i, pos) for i, pos in enumerate(positions)]
        await asyncio.gather(*tasks)

    # Calculate summary stats
    legal_results = [r for r in results if r.is_legal]

    # CPL now includes illegal moves (with penalty = eval_before + 5000)
    all_cpls = [r.cpl for r in results]
    legal_cpls = [r.cpl for r in legal_results]

    summary = {
        "player_id": player_id,
        "total_positions": len(positions),
        "legal_moves": len(legal_results),
        "legal_pct": len(legal_results) / len(positions) * 100 if positions else 0,
        "best_moves": sum(1 for r in results if r.is_best),
        "best_pct": sum(1 for r in results if r.is_best) / len(positions) * 100 if positions else 0,
        "avoided_blunders": sum(1 for r in results if r.avoided_blunder),
        "avoided_pct": sum(1 for r in results if r.avoided_blunder) / len(positions) * 100 if positions else 0,
        "avg_cpl": sum(all_cpls) / len(all_cpls) if all_cpls else 10000,  # Includes illegal move penalties
        "avg_cpl_legal": sum(legal_cpls) / len(legal_cpls) if legal_cpls else 10000,  # Only legal moves
        "median_cpl": _calculate_median(all_cpls),
    }

    return {"summary": summary, "results": [asdict(r) for r in results]}


def load_player_configs():
    """Load player configurations from benchmark.yaml."""
    import yaml

    config_path = Path(__file__).parent.parent / "config" / "benchmark.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    players = {}

    # Add engines
    for engine in config.get("engines", []):
        players[engine["player_id"]] = engine

    # Add LLMs
    for llm in config.get("llms", []):
        players[llm["player_id"]] = llm

    return players


async def main():
    parser = argparse.ArgumentParser(description="Run position benchmark")
    parser.add_argument(
        "--positions",
        type=Path,
        default=Path("position_benchmark/blunders.json"),
        help="Path to positions JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("position_benchmark/results.json"),
        help="Output results file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of positions to test (default: 100)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=16,
        help="Stockfish analysis depth",
    )
    parser.add_argument(
        "--players",
        nargs="+",
        help="Specific players to test (default: all cheap + engines)",
    )

    args = parser.parse_args()

    # Load positions
    print(f"Loading positions from {args.positions}...")
    with open(args.positions) as f:
        data = json.load(f)

    positions = data["blunders"][:args.limit]
    print(f"Testing on {len(positions)} positions")

    # Load player configs
    all_players = load_player_configs()

    # Determine which players to test
    if args.players:
        players_to_test = {p: all_players[p] for p in args.players if p in all_players}
    else:
        # Default: engines + cheap LLMs from benchmark.yaml
        # These are models that are known to be cheap (< $0.03/game)
        cheap_llms = [
            "gemini-2.5-flash (no thinking)",
            "gemini-2.0-flash-001",
            "glm-4.6 (no thinking)",
            "deepseek-v3.2 (no thinking)",
            "kimi-k2-0905",
            "deepseek-v3.1-terminus (no thinking)",
            "kimi-k2",
            "qwen3-235b-a22b-2507",
            "deepseek-chat-v3-0324",
            "mistral-medium-3",
            "gpt-3.5-turbo-0613",
            "deepseek-chat-v3.1 (no thinking)",
            "llama-3.3-70b-instruct",
            "gpt-3.5-turbo",
            "llama-4-maverick",
            "deepseek-r1-distill-qwen-32b",
            "grok-3-mini",
            "glm-4.6 (thinking)",
            "gpt-oss-120b (medium)",
            "gpt-oss-120b (high)",
            "kat-coder-pro",
        ]
        engines = ["random-bot", "maia-1100", "maia-1900", "eubos"]

        players_to_test = {}
        for p in engines + cheap_llms:
            if p in all_players:
                players_to_test[p] = all_players[p]

    print(f"\nTesting {len(players_to_test)} players:")
    for p in players_to_test:
        print(f"  - {p}")

    # Initialize Stockfish for evaluation
    print("\nStarting Stockfish...")
    stockfish = chess.engine.SimpleEngine.popen_uci("stockfish")

    all_results = {}

    try:
        for player_id, config in players_to_test.items():
            print(f"\n{'='*60}")
            print(f"Testing: {player_id}")
            print(f"{'='*60}")

            start = time.time()
            result = await run_benchmark(player_id, positions, stockfish, config, args.depth)
            elapsed = time.time() - start

            summary = result["summary"]
            print(f"\nSummary for {player_id}:")
            print(f"  Legal moves: {summary['legal_moves']}/{summary['total_positions']} ({summary['legal_pct']:.1f}%)")
            print(f"  Best moves: {summary['best_moves']} ({summary['best_pct']:.1f}%)")
            print(f"  Avoided blunders: {summary['avoided_blunders']} ({summary['avoided_pct']:.1f}%)")
            print(f"  Avg CPL: {summary['avg_cpl']:.1f}")
            print(f"  Time: {elapsed:.1f}s")

            all_results[player_id] = result

            # Save intermediate results
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)

    finally:
        stockfish.quit()

    print(f"\n{'='*60}")
    print("Final Rankings (by Avg CPL, lower is better):")
    print(f"{'='*60}")

    ranked = sorted(all_results.items(), key=lambda x: x[1]["summary"]["avg_cpl"])
    for i, (player_id, result) in enumerate(ranked, 1):
        s = result["summary"]
        print(f"{i:2}. {player_id:40} CPL: {s['avg_cpl']:7.1f}  Legal: {s['legal_pct']:5.1f}%  Best: {s['best_pct']:5.1f}%")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
