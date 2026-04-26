"""
Run position benchmark on models.

Tests models on blunder and equal positions, measures CPL (centipawn loss).
Uses unified positions.json containing both position types.
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

from llm.openrouter_client import OpenRouterPlayer, TransientAPIError
from llm.openrouter_completion_client import OpenRouterCompletionPlayer
from llm.gemini_client import GeminiPlayer
from llm.codex_subagent_client import CodexSubagentPlayer
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
    eval_before: float  # Eval before the position (for illegal CPL calculation)
    move_rank: Optional[int] = None
    top3: bool = False
    top5: bool = False
    reciprocal_rank: float = 0.0
    prompt_tokens: int = 0  # Prompt tokens used for this position
    completion_tokens: int = 0  # Completion tokens used for this position


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


def annotate_result_with_multipv(position: dict, result: PositionResult) -> PositionResult:
    """Attach rank-style metadata when the position has MultiPV annotations."""
    multipv = position.get("multipv") or []
    if not multipv:
        return result

    move_rank = None
    for rank, candidate in enumerate(multipv, start=1):
        if candidate.get("move") == result.model_move:
            move_rank = rank
            break

    if move_rank is None:
        move_rank = len(multipv) + 1

    result.move_rank = move_rank
    result.top3 = move_rank <= 3
    result.top5 = move_rank <= 5
    result.reciprocal_rank = 1.0 / move_rank
    return result


async def test_llm_on_position(
    player: "BaseLLMPlayer",
    position: dict,
    engine: chess.engine.SimpleEngine,
    depth: int = 30,
) -> PositionResult:
    """Test an LLM on a single position."""
    fen = position["fen"]
    move_history = position.get("move_history") or []
    move_history_san = position.get("move_history_san") or move_history
    side = position["side_to_move"].capitalize()
    best_move_uci = position["best_move"]
    blunder_move_uci = position.get("blunder_move", "")  # Optional for equal positions

    board = chess.Board(fen)
    # For completion-style players, replay move history so board.move_stack is populated
    # (these players prompt with PGN and need the game's move sequence).
    if isinstance(player, OpenRouterCompletionPlayer) and move_history:
        replay = chess.Board()
        try:
            for uci_move in move_history:
                replay.push(chess.Move.from_uci(uci_move))
            if replay.fen() == fen:
                board = replay
        except (ValueError, chess.InvalidMoveError):
            pass
    perspective = board.turn

    # Build prompt and get model's move
    prompt = build_position_prompt(fen, move_history_san, side)

    # Calculate illegal move CPL: half the swing to losing (eval_before + 5000)
    # This represents "half a game loss" since 2 illegals = forfeit
    eval_before = position["eval_before"]
    illegal_cpl = eval_before + 5000

    # Let TransientAPIError propagate — API failures are not illegal moves
    model_move_uci = await player.select_move(board, is_retry=False)

    # Empty response means the API returned no content (truncation, etc.)
    if not model_move_uci or not model_move_uci.strip():
        raise TransientAPIError("Model returned empty response")

    # Check if move is legal
    try:
        move = chess.Move.from_uci(model_move_uci)
        if move not in board.legal_moves:
            raise ValueError("Illegal move")
        model_move_san = board.san(move)
        is_legal = True
    except (ValueError, chess.InvalidMoveError):
        return annotate_result_with_multipv(position, PositionResult(
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
            eval_before=eval_before,
        ))

    # Evaluate model's move
    board.push(move)
    info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
    eval_model = -eval_to_cp(info_after, not perspective)  # Negate for side that moved
    board.pop()

    # Get best move eval
    eval_best = position["eval_before"]

    # Calculate CPL
    cpl = max(0, eval_best - eval_model)

    return annotate_result_with_multipv(position, PositionResult(
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
        eval_before=eval_before,
    ))


def test_engine_on_position(
    engine_player,
    position: dict,
    stockfish: chess.engine.SimpleEngine,
    depth: int = 30,
) -> PositionResult:
    """Test an engine on a single position."""
    fen = position["fen"]
    best_move_uci = position["best_move"]
    blunder_move_uci = position.get("blunder_move", "")  # Optional for equal positions

    board = chess.Board(fen)
    perspective = board.turn

    eval_before = position["eval_before"]

    # Get engine's move
    try:
        move = engine_player.select_move(board)
        model_move_uci = move.uci()
        model_move_san = board.san(move)
        is_legal = True
    except Exception as e:
        return annotate_result_with_multipv(position, PositionResult(
            position_idx=0,
            fen=fen,
            model_move="",
            model_move_san="",
            best_move=best_move_uci,
            best_move_san=position["best_move_san"],
            blunder_move=blunder_move_uci,
            cpl=eval_before + 5000,  # Same formula as LLM illegal moves
            is_legal=False,
            is_best=False,
            avoided_blunder=True,
            eval_model=-10000,
            eval_best=eval_before,
            eval_before=eval_before,
        ))

    # Evaluate engine's move
    board.push(move)
    info_after = stockfish.analyse(board, chess.engine.Limit(depth=depth))
    eval_model = -eval_to_cp(info_after, not perspective)
    board.pop()

    cpl = max(0, eval_before - eval_model)

    return annotate_result_with_multipv(position, PositionResult(
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
        eval_best=eval_before,
        eval_before=eval_before,
    ))


async def run_benchmark(
    player_id: str,
    positions: list[dict],
    stockfish: chess.engine.SimpleEngine,
    player_config: dict,
    depth: int = 30,
    api_backend: str = "openrouter",
) -> dict:
    """Run benchmark for a single player."""
    results = []
    token_acc = {"prompt": 0, "completion": 0}

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
        def create_llm_player():
            """Create a fresh LLM player for this benchmark run."""
            common_kwargs = {
                "player_id": player_id,
                "temperature": player_config.get("temperature", 0.0),
                "reasoning": player_config.get("reasoning"),
                "reasoning_effort": player_config.get("reasoning_effort"),
                "timeout": player_config.get("timeout", 600),
            }

            if player_config.get("api") == "codex" or api_backend == "codex":
                return CodexSubagentPlayer(
                    player_id=common_kwargs["player_id"],
                    model_name=player_config.get("codex_model_name") or player_config.get("model_name"),
                    reasoning_effort=common_kwargs["reasoning_effort"] or player_config.get("codex_reasoning_effort", "medium"),
                    timeout=common_kwargs["timeout"],
                    max_retries=player_config.get("codex_max_retries", 2),
                    max_concurrent=player_config.get("codex_max_concurrent"),
                    sandbox=player_config.get("codex_sandbox", "read-only"),
                    ignore_rules=player_config.get("codex_ignore_rules", True),
                    ephemeral=player_config.get("codex_ephemeral", True),
                    include_legal_moves=player_config.get("codex_include_legal_moves", False),
                    extra_args=player_config.get("codex_extra_args"),
                    working_dir=player_config.get("codex_working_dir"),
                )

            if api_backend == "gemini":
                # Strip provider prefix for direct Gemini API
                model_name = player_config.get("model_name", "")
                if model_name.startswith("google/"):
                    model_name = model_name[len("google/"):]
                return GeminiPlayer(
                    model_name=model_name,
                    **common_kwargs,
                )

            if player_config.get("api") == "completion":
                # OpenRouterCompletionPlayer doesn't accept reasoning_* kwargs
                return OpenRouterCompletionPlayer(
                    player_id=common_kwargs["player_id"],
                    model_name=player_config.get("model_name"),
                    temperature=common_kwargs["temperature"],
                    provider_order=player_config.get("provider_order"),
                    provider_ignore=player_config.get("provider_ignore"),
                    timeout=common_kwargs["timeout"],
                )

            return OpenRouterPlayer(
                model_name=player_config.get("model_name"),
                reasoning_max_tokens=player_config.get("reasoning_max_tokens"),
                max_tokens=player_config.get("max_tokens", 0),
                provider_order=player_config.get("provider_order"),
                provider_ignore=player_config.get("provider_ignore"),
                **common_kwargs,
            )

        # Gemini preview/reasoning models have been more reliable in this
        # benchmark when requests are serialized and use a fresh client per
        # position, instead of sharing one mutable player across tasks.
        if player_config.get("api") == "codex" or api_backend == "codex":
            max_concurrent = player_config.get("codex_position_max_concurrent", 1)
        else:
            max_concurrent = 1 if api_backend == "gemini" else 10
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = [0]  # Use list to allow modification in nested function
        results = [None] * len(positions)  # Pre-allocate to maintain order

        fresh_player_per_position = api_backend == "gemini" or player_config.get("api") == "codex" or api_backend == "codex"
        shared_player = None if fresh_player_per_position else create_llm_player()

        async def test_with_semaphore(idx: int, pos: dict):
            max_api_retries = 3
            for attempt in range(max_api_retries):
                player = shared_player if shared_player is not None else create_llm_player()
                try:
                    async with semaphore:
                        result = await test_llm_on_position(player, pos, stockfish, depth)
                        # Read per-request token counts inside semaphore to avoid race with concurrent tasks
                        result.prompt_tokens = getattr(player, "_last_prompt_tokens", 0)
                        result.completion_tokens = getattr(player, "_last_completion_tokens", 0)
                    result.position_idx = idx
                    results[idx] = result
                    completed[0] += 1
                    status = "illegal" if not result.is_legal else f"CPL: {result.cpl:.0f}"
                    if result.is_best:
                        status += " (best!)"
                    print(f"  [{completed[0]}/{len(positions)}] {status}")
                    return result
                except TransientAPIError as e:
                    if attempt < max_api_retries - 1:
                        print(f"  [API error pos {idx}, retry {attempt+1}/{max_api_retries}]: {e}")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        completed[0] += 1
                        print(f"  [{completed[0]}/{len(positions)}] ERROR (API failure, skipping pos {idx})")
                        results[idx] = None  # Mark as skipped
                        return None
                finally:
                    if shared_player is None:
                        # Accumulate tokens before closing (fresh player per position)
                        token_acc["prompt"] += getattr(player, "prompt_tokens", 0)
                        token_acc["completion"] += getattr(player, "completion_tokens", 0)
                        await player.close()

        try:
            # Launch all tasks
            tasks = [test_with_semaphore(i, pos) for i, pos in enumerate(positions)]
            await asyncio.gather(*tasks)
        finally:
            if shared_player is not None:
                # Read token counts before closing
                token_acc["prompt"] = getattr(shared_player, "prompt_tokens", 0)
                token_acc["completion"] = getattr(shared_player, "completion_tokens", 0)
                await shared_player.close()

        # Filter out skipped positions (API failures)
        skipped = sum(1 for r in results if r is None)
        if skipped:
            print(f"\n  Skipped {skipped} positions due to API errors")
        results = [r for r in results if r is not None]

    # Calculate summary stats
    result_dicts = [asdict(r) for r in results]

    # Build index -> position type mapping for per-type breakdowns
    pos_type_by_idx = {i: p.get("type") for i, p in enumerate(positions)}

    def _calc_type_summary(subset_results):
        """Calculate summary stats for a subset of results."""
        legal = [r for r in subset_results if r.is_legal]
        all_cpls = [r.cpl for r in subset_results]
        legal_cpls = [r.cpl for r in legal]
        n = len(subset_results)
        summary = {
            "total_positions": n,
            "legal_moves": len(legal),
            "legal_pct": len(legal) / n * 100 if n else 0,
            "best_moves": sum(1 for r in subset_results if r.is_best),
            "best_pct": sum(1 for r in subset_results if r.is_best) / n * 100 if n else 0,
            "avoided_blunders": sum(1 for r in subset_results if r.avoided_blunder),
            "avoided_pct": sum(1 for r in subset_results if r.avoided_blunder) / n * 100 if n else 0,
            "avg_cpl": sum(all_cpls) / len(all_cpls) if all_cpls else 10000,
            "avg_cpl_legal": sum(legal_cpls) / len(legal_cpls) if legal_cpls else 10000,
            "median_cpl": _calculate_median(all_cpls),
        }
        rankable = [r for r in subset_results if r.move_rank is not None]
        if rankable:
            summary["top3_pct"] = sum(1 for r in rankable if r.top3) / len(rankable) * 100
            summary["top5_pct"] = sum(1 for r in rankable if r.top5) / len(rankable) * 100
            summary["avg_reciprocal_rank"] = sum(r.reciprocal_rank for r in rankable) / len(rankable)
        return summary

    # Overall summary
    summary = _calc_type_summary(results)
    summary["player_id"] = player_id
    summary["positions_attempted"] = len(positions)
    summary["positions_skipped"] = len(positions) - len(results)

    # Per-type breakdowns using position_idx to look up type
    blunder_results = [r for r in results if pos_type_by_idx.get(r.position_idx) == "blunder"]
    equal_results = [r for r in results if pos_type_by_idx.get(r.position_idx) == "equal"]

    if blunder_results:
        summary["blunder"] = _calc_type_summary(blunder_results)
    if equal_results:
        summary["equal"] = _calc_type_summary(equal_results)

    # Add per-position token stats to summary
    pos_completion_tokens = [ct for r in result_dicts if (ct := r.get("completion_tokens", 0)) > 0]
    if pos_completion_tokens:
        summary["avg_completion_tokens"] = sum(pos_completion_tokens) / len(pos_completion_tokens)
        summary["median_completion_tokens"] = _calculate_median(pos_completion_tokens)
        summary["total_completion_tokens"] = sum(pos_completion_tokens)

    return {"summary": summary, "results": result_dicts, "token_usage": token_acc}


async def run_benchmark_for_scheduler(
    player_id: str,
    player_config: dict,
    stockfish: chess.engine.SimpleEngine,
    positions: list[dict],
    depth: int = 30,
    api_backend: str = "openrouter",
    original_indices: Optional[list[int]] = None,
) -> dict:
    """
    Run position benchmark for a single model, called from the match scheduler.

    Uses a shared Stockfish instance and pre-loaded positions (caller manages lifecycle).
    Merges results into position_benchmark/results.json.

    Args:
        player_id: The model's player ID
        player_config: Config dict for the model (from benchmark.yaml)
        stockfish: Shared Stockfish engine (caller opens/closes)
        positions: Pre-loaded positions list
        depth: Stockfish analysis depth (default 30)
        api_backend: API backend to use ("openrouter" or "gemini")
        original_indices: If positions is a subset, maps subset index -> original index
                         in the full positions.json (needed for correct position_idx in results)

    Returns:
        {"success": True, "summary": dict, "token_usage": dict} on success
        {"success": False, "error": str} on failure
    """
    try:
        result = await run_benchmark(
            player_id=player_id,
            positions=positions,
            stockfish=stockfish,
            player_config=player_config,
            depth=depth,
            api_backend=api_backend,
        )

        # Remap position_idx to original indices if running a subset
        if original_indices is not None:
            for r in result["results"]:
                subset_idx = r["position_idx"]
                if 0 <= subset_idx < len(original_indices):
                    r["position_idx"] = original_indices[subset_idx]

        # Merge into results.json
        results_path = Path(__file__).parent / "results.json"
        all_results = {}
        if results_path.exists():
            with open(results_path) as f:
                all_results = json.load(f)

        all_results[player_id] = {
            "summary": result["summary"],
            "results": result["results"],
        }

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # Sync to Firestore (per-model document to avoid 1 MiB limit)
        try:
            from firebase_client import get_firestore_client, BENCHMARK_RESULTS_COLLECTION
            db = get_firestore_client()
            db.collection(BENCHMARK_RESULTS_COLLECTION).document(player_id).set(
                all_results[player_id]
            )
        except Exception as e:
            print(f"  Warning: Failed to sync benchmark results to Firestore: {e}")

        return {
            "success": True,
            "summary": result["summary"],
            "token_usage": result.get("token_usage", {"prompt": 0, "completion": 0}),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


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
        default=Path("position_benchmark/positions.json"),
        help="Path to positions JSON (unified format)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("position_benchmark/results.json"),
        help="Output results file",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=30,
        help="Stockfish analysis depth",
    )
    parser.add_argument(
        "--players",
        nargs="+",
        help="Specific players to test (default: all cheap + engines)",
    )
    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help="Only run positions missing from existing results (skipped or errored)",
    )
    parser.add_argument(
        "--api",
        choices=["openrouter", "gemini", "codex"],
        default="openrouter",
        help="API backend to use (default: openrouter)",
    )
    parser.add_argument(
        "--type",
        choices=["blunder", "equal", "all"],
        default="equal",
        help="Only test positions of this type (default: equal)",
    )
    parser.add_argument(
        "--sync-firestore",
        action="store_true",
        help="Sync results to Firestore even when using a custom output file.",
    )

    args = parser.parse_args()
    default_output = Path("position_benchmark/results.json").resolve()
    should_sync_firestore = args.sync_firestore or args.output.resolve() == default_output

    # Load positions
    print(f"Loading positions from {args.positions}...")
    with open(args.positions) as f:
        data = json.load(f)

    all_positions = data["positions"]
    # Map from filtered index -> original index (identity when no filter)
    type_filter_idx_map = None
    if args.type and args.type != "all":
        type_filter_idx_map = {new_idx: orig_idx for new_idx, (orig_idx, p)
                               in enumerate((i, p) for i, p in enumerate(all_positions) if p.get("type") == args.type)}
        positions = [all_positions[orig] for orig in type_filter_idx_map.values()]
        print(f"Testing on {len(positions)} {args.type} positions")
    else:
        positions = all_positions
        blunder_count = sum(1 for p in positions if p.get("type") == "blunder")
        equal_count = sum(1 for p in positions if p.get("type") == "equal")
        print(f"Testing on {len(positions)} positions ({blunder_count} blunder, {equal_count} equal)")

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

    # Load existing results to merge with (don't overwrite)
    all_results = {}
    if Path(args.output).exists():
        with open(args.output) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results from {args.output}")

    try:
        for player_id, config in players_to_test.items():
            print(f"\n{'='*60}")
            print(f"Testing: {player_id}")
            print(f"{'='*60}")

            # Determine which positions to test
            positions_to_test = positions
            existing_by_idx = {}

            if args.retry_missing and player_id in all_results:
                existing = all_results[player_id].get("results", [])
                # Stored position_idx is in all_positions coordinates (after any
                # type_filter remap). Key existing_by_idx by that so we can merge
                # new subset results back in.
                existing_by_idx = {r["position_idx"]: r for r in existing}

                def _filtered_to_all(fi: int) -> int:
                    return type_filter_idx_map[fi] if type_filter_idx_map else fi

                missing_filtered = [fi for fi in range(len(positions))
                                    if _filtered_to_all(fi) not in existing_by_idx]

                if not missing_filtered:
                    print(f"  All {len(positions)} positions already have results, skipping")
                    continue

                # Map subset index (the index within positions_to_test, 0..N-1)
                # directly to all_positions index. Used below to rewrite result
                # position_idx in one step, bypassing the type_filter_idx_map remap.
                retry_idx_map = {si: _filtered_to_all(fi) for si, fi in enumerate(missing_filtered)}
                positions_to_test = [positions[fi] for fi in missing_filtered]
                print(f"  Retrying {len(positions_to_test)} missing positions (have {len(existing_by_idx)} existing)")

            start = time.time()
            result = await run_benchmark(player_id, positions_to_test, stockfish, config, args.depth, args.api)
            elapsed = time.time() - start

            # Remap filtered indices back to original position indices.
            # With --retry-missing, retry_idx_map maps subset index directly to
            # all_positions index, so skip the type_filter step to avoid a double
            # remap.
            if args.retry_missing and player_id in all_results:
                for r in result["results"]:
                    r["position_idx"] = retry_idx_map[r["position_idx"]]
            elif type_filter_idx_map is not None:
                for r in result["results"]:
                    r["position_idx"] = type_filter_idx_map[r["position_idx"]]

            # Merge with existing results if retrying missing
            if args.retry_missing and existing_by_idx:
                for r in result["results"]:
                    existing_by_idx[r["position_idx"]] = r
                merged_results = [existing_by_idx[i] for i in sorted(existing_by_idx.keys())]

                # Recalculate summary on merged results
                pos_type_by_idx = {i: p.get("type") for i, p in enumerate(positions)}

                @dataclass
                class _R:
                    """Thin wrapper for summary calculation."""
                    is_legal: bool
                    is_best: bool
                    avoided_blunder: bool
                    cpl: float
                    position_idx: int

                wrapped = [_R(**{k: r[k] for k in ["is_legal", "is_best", "avoided_blunder", "cpl", "position_idx"]}) for r in merged_results]

                def _calc(subset):
                    legal = [r for r in subset if r.is_legal]
                    all_cpls = [r.cpl for r in subset]
                    legal_cpls = [r.cpl for r in legal]
                    n = len(subset)
                    return {
                        "total_positions": n,
                        "legal_moves": len(legal),
                        "legal_pct": len(legal) / n * 100 if n else 0,
                        "best_moves": sum(1 for r in subset if r.is_best),
                        "best_pct": sum(1 for r in subset if r.is_best) / n * 100 if n else 0,
                        "avoided_blunders": sum(1 for r in subset if r.avoided_blunder),
                        "avoided_pct": sum(1 for r in subset if r.avoided_blunder) / n * 100 if n else 0,
                        "avg_cpl": sum(all_cpls) / len(all_cpls) if all_cpls else 10000,
                        "avg_cpl_legal": sum(legal_cpls) / len(legal_cpls) if legal_cpls else 10000,
                        "median_cpl": _calculate_median(all_cpls),
                    }

                summary = _calc(wrapped)
                summary["player_id"] = player_id
                summary["positions_attempted"] = len(positions)
                summary["positions_skipped"] = len(positions) - len(merged_results)

                blunder_r = [r for r in wrapped if pos_type_by_idx.get(r.position_idx) == "blunder"]
                equal_r = [r for r in wrapped if pos_type_by_idx.get(r.position_idx) == "equal"]
                if blunder_r:
                    summary["blunder"] = _calc(blunder_r)
                if equal_r:
                    summary["equal"] = _calc(equal_r)

                result = {"summary": summary, "results": merged_results}

            summary = result["summary"]
            print(f"\nSummary for {player_id}:")
            print(f"  Legal moves: {summary['legal_moves']}/{summary['total_positions']} ({summary['legal_pct']:.1f}%)")
            print(f"  Best moves: {summary['best_moves']} ({summary['best_pct']:.1f}%)")
            print(f"  Avoided blunders: {summary['avoided_blunders']} ({summary['avoided_pct']:.1f}%)")
            print(f"  Avg CPL: {summary['avg_cpl']:.1f}")
            if summary.get("positions_skipped"):
                print(f"  Skipped: {summary['positions_skipped']}")
            for ptype in ["blunder", "equal"]:
                if ptype in summary:
                    ts = summary[ptype]
                    print(f"  [{ptype}] Legal: {ts['legal_pct']:.1f}%  Best: {ts['best_pct']:.1f}%  CPL: {ts['avg_cpl']:.1f}")
            print(f"  Time: {elapsed:.1f}s")

            all_results[player_id] = result

            # Save intermediate results
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)

            if should_sync_firestore:
                # Sync to Firestore (per-model document to avoid 1 MiB limit)
                try:
                    from firebase_client import get_firestore_client, BENCHMARK_RESULTS_COLLECTION
                    db = get_firestore_client()
                    db.collection(BENCHMARK_RESULTS_COLLECTION).document(player_id).set(
                        all_results[player_id]
                    )
                except Exception as e:
                    print(f"  Warning: Failed to sync benchmark results to Firestore: {e}")

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
