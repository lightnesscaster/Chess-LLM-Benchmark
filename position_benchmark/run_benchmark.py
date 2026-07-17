"""Run the history-replayed position benchmark on model configurations.

The production core is the 50 equal positions at Stockfish depth 30. Blunder
positions are an optional historical downside panel. See
``position_benchmark/README.md`` for the canonical methodology and call counts.
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
from llm.base_llm import request_llm_move
from llm.prompts import board_to_ascii
from engines.stockfish_engine import StockfishEngine
from engines.maia_engine import MaiaEngine
from engines.random_engine import RandomEngine
from position_benchmark.predictions import CURRENT_BENCHMARK_VERSION, result_row_is_current
from position_benchmark.layout import CORE_POSITIONS_PATH, CORE_RESULTS_PATH, RESULT_SCHEMA_VERSION
from position_benchmark.token_accounting import sum_result_row_tokens
from position_benchmark.retry_protocol import (
    CONDITIONAL_RETRY_PROTOCOL_VERSION,
    attach_conditional_retry_summary,
)
from position_benchmark.scoring import ILLEGAL_MOVE_EVAL, illegal_move_cpl
from rating.cost_calculator import CostCalculator


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
    conditional_retry_protocol_version: Optional[str] = None
    retry_attempted: bool = False
    retry_move: Optional[str] = None
    retry_move_san: Optional[str] = None
    retry_is_legal: Optional[bool] = None
    initial_prompt_tokens: int = 0
    initial_completion_tokens: int = 0
    retry_prompt_tokens: int = 0
    retry_completion_tokens: int = 0


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


def stamp_current_benchmark_rows(rows: list[dict], depth: int) -> None:
    """Mark result rows produced by the current history-replay benchmark path."""
    for row in rows:
        row["position_benchmark_version"] = CURRENT_BENCHMARK_VERSION
        row["prompt_history_replay"] = True
        row["stockfish_depth"] = depth


def stamp_benchmark_summary(summary: dict, rows: list[dict], depth: int) -> None:
    """Attach summary-level freshness metadata without hiding mixed legacy rows."""
    current_rows = sum(1 for row in rows if result_row_is_current(row, min_stockfish_depth=depth))
    summary["stockfish_depth"] = depth
    summary["current_result_rows"] = current_rows
    summary["legacy_or_mixed_result_rows"] = len(rows) - current_rows
    if rows and current_rows == len(rows):
        summary["position_benchmark_version"] = CURRENT_BENCHMARK_VERSION
        summary["prompt_history_replay"] = True
    else:
        summary["position_benchmark_version"] = "mixed-or-legacy"
        summary["prompt_history_replay"] = False


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


def validate_position_input(data: dict, *, allow_legacy_input: bool = False) -> None:
    """Reject inactive legacy registries and malformed production core panels."""
    metadata = data.get("metadata", {})
    if metadata.get("active_production_input") is False and not allow_legacy_input:
        replacement = metadata.get("replacement", "the canonical panel in benchmark_manifest.json")
        raise ValueError(
            "Refusing inactive legacy position input. "
            f"Use {replacement}, or pass --allow-legacy-input for deliberate historical work."
        )

    positions = data.get("positions", [])
    if metadata.get("panel") == "core-equal":
        if len(positions) != 50 or any(
            position.get("type") != "equal" for position in positions
        ):
            raise ValueError("Production core panel must contain exactly 50 equal positions")


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


def replay_position_board(position: dict) -> chess.Board:
    """
    Build a board for the position, preserving move history when available.

    The normal game runner passes a board with move_stack populated, so chat
    prompts include move history and the opponent's last move. Position
    benchmark positions are stored as FEN plus history; replay that history when
    it reaches the same piece/turn/castling/en-passant state as the target FEN.
    Histories in older artifacts may be SAN while newer ones are UCI.
    """
    target = chess.Board(position["fen"])

    histories = []
    raw_history = position.get("move_history") or []
    san_history = position.get("move_history_san") or []
    if raw_history:
        histories.append(raw_history)
    if san_history and san_history != raw_history:
        histories.append(san_history)

    target_key = " ".join(target.fen().split()[:4])
    for history in histories:
        replay = chess.Board()
        try:
            for move_text in history:
                text = str(move_text).strip()
                try:
                    move = chess.Move.from_uci(text)
                    if move not in replay.legal_moves:
                        raise ValueError("UCI move is not legal in replay position")
                except (ValueError, chess.InvalidMoveError):
                    move = replay.parse_san(text)
                replay.push(move)
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            continue

        replay_key = " ".join(replay.fen().split()[:4])
        if replay_key == target_key:
            replay.halfmove_clock = target.halfmove_clock
            replay.fullmove_number = target.fullmove_number
            return replay

    return target


def _total_token_usage(player: "BaseLLMPlayer") -> tuple[int, int]:
    """Read cumulative usage from one position-isolated player."""
    return (
        int(getattr(player, "prompt_tokens", 0) or 0),
        int(getattr(player, "completion_tokens", 0) or 0),
    )


def _call_token_delta(
    player: "BaseLLMPlayer",
    before: tuple[int, int],
) -> tuple[int, int]:
    """Return all provider tokens consumed by one select-move invocation."""
    after = _total_token_usage(player)
    prompt_delta = after[0] - before[0]
    completion_delta = after[1] - before[1]
    if prompt_delta >= 0 and completion_delta >= 0:
        return prompt_delta, completion_delta
    return (
        int(getattr(player, "_last_prompt_tokens", 0) or 0),
        int(getattr(player, "_last_completion_tokens", 0) or 0),
    )


def _validate_llm_move(
    board: chess.Board,
    response: Optional[str],
) -> tuple[str, Optional[chess.Move], str]:
    """Apply the game runner's stripping, UCI normalization, and legality rules."""
    move_text = response.strip() if response else ""
    if not move_text:
        return "", None, ""

    try:
        move = chess.Move.from_uci(move_text.lower())
        if move not in board.legal_moves:
            return move_text, None, ""
        return move.uci(), move, board.san(move)
    except (ValueError, chess.InvalidMoveError):
        return move_text, None, ""


async def test_llm_on_position(
    player: "BaseLLMPlayer",
    position: dict,
    engine: chess.engine.SimpleEngine,
    depth: int = 30,
) -> PositionResult:
    """Test an LLM on a single position."""
    fen = position["fen"]
    best_move_uci = position["best_move"]
    blunder_move_uci = position.get("blunder_move", "")  # Optional for equal positions

    board = replay_position_board(position)
    perspective = board.turn

    # One illegal is half a forfeit. Floor at zero so a forced-mate position
    # already below the penalty evaluation cannot receive negative CPL.
    eval_before = position["eval_before"]
    illegal_cpl = illegal_move_cpl(eval_before)

    # Let TransientAPIError propagate: provider failures are not illegal moves.
    # Empty/invalid model responses, however, follow the production game retry
    # path and count as first-attempt illegals.
    initial_usage_before = _total_token_usage(player)
    initial_response = await request_llm_move(
        player,
        board,
        is_retry=False,
        last_move_illegal=None,
    )
    initial_prompt_tokens, initial_completion_tokens = _call_token_delta(
        player,
        initial_usage_before,
    )
    model_move_uci, move, model_move_san = _validate_llm_move(board, initial_response)

    if move is None:
        retry_usage_before = _total_token_usage(player)
        retry_response = await request_llm_move(
            player,
            board,
            is_retry=True,
            last_move_illegal=model_move_uci or "invalid",
        )
        retry_prompt_tokens, retry_completion_tokens = _call_token_delta(
            player,
            retry_usage_before,
        )
        retry_move_uci, retry_move, retry_move_san = _validate_llm_move(
            board,
            retry_response,
        )
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
            eval_model=ILLEGAL_MOVE_EVAL,
            eval_best=eval_before,
            eval_before=eval_before,
            prompt_tokens=initial_prompt_tokens + retry_prompt_tokens,
            completion_tokens=initial_completion_tokens + retry_completion_tokens,
            conditional_retry_protocol_version=CONDITIONAL_RETRY_PROTOCOL_VERSION,
            retry_attempted=True,
            retry_move=retry_move_uci,
            retry_move_san=retry_move_san,
            retry_is_legal=retry_move is not None,
            initial_prompt_tokens=initial_prompt_tokens,
            initial_completion_tokens=initial_completion_tokens,
            retry_prompt_tokens=retry_prompt_tokens,
            retry_completion_tokens=retry_completion_tokens,
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
        prompt_tokens=initial_prompt_tokens,
        completion_tokens=initial_completion_tokens,
        conditional_retry_protocol_version=CONDITIONAL_RETRY_PROTOCOL_VERSION,
        initial_prompt_tokens=initial_prompt_tokens,
        initial_completion_tokens=initial_completion_tokens,
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
            cpl=illegal_move_cpl(eval_before),
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
    model_call_acc = [0]

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

        # Each position owns its player so initial/retry token deltas and debug
        # state cannot race across concurrent tasks. The semaphore still bounds
        # provider concurrency.
        if player_config.get("api") == "codex" or api_backend == "codex":
            max_concurrent = player_config.get("codex_position_max_concurrent", 1)
        else:
            max_concurrent = 1 if api_backend == "gemini" else 10
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = [0]  # Use list to allow modification in nested function
        results = [None] * len(positions)  # Pre-allocate to maintain order

        shared_player = None

        async def test_with_semaphore(idx: int, pos: dict):
            max_api_retries = 3
            for attempt in range(max_api_retries):
                player = shared_player if shared_player is not None else create_llm_player()
                try:
                    async with semaphore:
                        result = await test_llm_on_position(player, pos, stockfish, depth)
                    result.position_idx = idx
                    results[idx] = result
                    completed[0] += 1
                    if result.is_legal:
                        status = f"CPL: {result.cpl:.0f}"
                    elif result.retry_is_legal:
                        status = "illegal (retry recovered)"
                    else:
                        status = "illegal (retry failed)"
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
                        model_call_acc[0] += len(getattr(player, "move_times", []))
                        await player.close()

        try:
            # Launch all tasks
            tasks = [asyncio.create_task(test_with_semaphore(i, pos)) for i, pos in enumerate(positions)]
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                completed_count = sum(1 for r in results if r is not None)
                print(f"\n  Cancelled; saving {completed_count}/{len(positions)} completed positions")
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
    for row in result_dicts:
        position = positions[row["position_idx"]]
        if position.get("position_id"):
            row["position_id"] = position["position_id"]
            row["result_schema_version"] = RESULT_SCHEMA_VERSION
        if position.get("panel"):
            row["panel"] = position["panel"]
        if position.get("legacy_position_idx") is not None:
            row["legacy_position_idx"] = position["legacy_position_idx"]
    stamp_current_benchmark_rows(result_dicts, depth)

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
    summary["result_schema_version"] = RESULT_SCHEMA_VERSION
    panels = {position.get("panel") for position in positions if position.get("panel")}
    if len(panels) == 1:
        summary["panel"] = panels.pop()
    stamp_benchmark_summary(summary, result_dicts, depth)
    attach_conditional_retry_summary(summary, result_dicts)
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

    return {
        "summary": summary,
        "results": result_dicts,
        # Canonical totals describe the rows retained in this artifact.
        "token_usage": sum_result_row_tokens(result_dicts),
        # Run totals may be higher because they include failed/retried calls.
        "run_token_usage": token_acc,
        "run_model_calls": model_call_acc[0],
    }


async def run_benchmark_for_scheduler(
    player_id: str,
    player_config: dict,
    stockfish: chess.engine.SimpleEngine,
    positions: list[dict],
    depth: int = 30,
    api_backend: str = "openrouter",
    original_indices: Optional[list[int]] = None,
    results_path: Path = CORE_RESULTS_PATH,
    sync_firestore: bool | None = None,
) -> dict:
    """
    Run position benchmark for a single model, called from the match scheduler.

    Uses a shared Stockfish instance and pre-loaded positions (caller manages lifecycle).
    Merges results into the requested canonical panel results file.

    Args:
        player_id: The model's player ID
        player_config: Config dict for the model (from benchmark.yaml)
        stockfish: Shared Stockfish engine (caller opens/closes)
        positions: Pre-loaded positions list
        depth: Stockfish analysis depth (default 30)
        api_backend: API backend to use ("openrouter" or "gemini")
        original_indices: Legacy compatibility mapping for callers using the old
                         combined position registry.
        results_path: Canonical output for the selected panel.
        sync_firestore: Whether to sync the result to the production core
                        collection. Defaults to true only for ``CORE_RESULTS_PATH``.

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

        # Merge into the canonical panel results file.
        all_results = {}
        if results_path.exists():
            with open(results_path) as f:
                all_results = json.load(f)

        all_results[player_id] = {
            "summary": result["summary"],
            "results": result["results"],
            "token_usage": result.get("token_usage", {"prompt": 0, "completion": 0}),
            "run_token_usage": result.get("run_token_usage", {"prompt": 0, "completion": 0}),
            "run_model_calls": result.get("run_model_calls", 0),
        }

        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        should_sync_firestore = (
            results_path.resolve() == CORE_RESULTS_PATH.resolve()
            if sync_firestore is None
            else sync_firestore
        )
        if should_sync_firestore:
            # Core rows are synced per model to avoid Firestore's 1 MiB document limit.
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
            # The scheduler uses this to charge the run budget, including retries.
            "token_usage": result.get(
                "run_token_usage",
                result.get("token_usage", {"prompt": 0, "completion": 0}),
            ),
            "canonical_token_usage": result.get(
                "token_usage", {"prompt": 0, "completion": 0}
            ),
            "model_calls": result.get("run_model_calls", 0),
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


def build_ad_hoc_player_config(args: argparse.Namespace) -> dict | None:
    """Build a one-off LLM config from CLI args, without editing benchmark.yaml."""
    if not args.model_name:
        return None
    if not args.player_id:
        raise ValueError("--player-id is required when using --model-name")

    config: dict = {
        "player_id": args.player_id,
        "model_name": args.model_name,
        "temperature": args.temperature,
    }
    if args.reasoning_effort:
        config["reasoning_effort"] = args.reasoning_effort
    if args.no_reasoning:
        config["reasoning"] = False
    if args.timeout is not None:
        config["timeout"] = args.timeout
    return config


def config_is_engine(player_config: dict) -> bool:
    """Return whether a player config is an engine rather than an API-backed LLM."""
    return player_config.get("type") in ["stockfish", "maia", "random", "uci", "survival"]


def config_uses_reasoning(player_config: dict) -> bool:
    """Infer whether a position benchmark call should use reasoning cost estimates."""
    if player_config.get("reasoning") is False:
        return False
    return bool(
        player_config.get("reasoning") is True
        or player_config.get("reasoning_effort")
        or player_config.get("reasoning_max_tokens")
    )


def planned_position_count(
    player_id: str,
    *,
    selected_count: int,
    type_filter_idx_map: dict[int, int],
    all_results: dict,
    retry_missing: bool,
    refresh_retry_evidence: bool = False,
) -> int:
    """Return how many selected positions would be run for one player."""
    if (not retry_missing and not refresh_retry_evidence) or player_id not in all_results:
        return selected_count

    existing = all_results[player_id].get("results", [])
    existing_by_idx = {
        row.get("position_idx")
        for row in existing
        if isinstance(row.get("position_idx"), int)
        and (
            not refresh_retry_evidence
            or row.get("conditional_retry_protocol_version")
            == CONDITIONAL_RETRY_PROTOCOL_VERSION
        )
    }
    return sum(
        1
        for filtered_idx in range(selected_count)
        if type_filter_idx_map[filtered_idx] not in existing_by_idx
    )


def estimate_selected_benchmark_cost(
    players_to_test: dict[str, dict],
    *,
    selected_count: int,
    type_filter_idx_map: dict[int, int],
    all_results: dict,
    retry_missing: bool,
    cost_calculator: CostCalculator,
    refresh_retry_evidence: bool = False,
) -> tuple[float, list[str]]:
    """Estimate planned API cost for selected non-engine position benchmarks."""
    known_cost = 0.0
    unknown_players: list[str] = []

    for player_id, config in players_to_test.items():
        if config_is_engine(config):
            continue

        num_positions = planned_position_count(
            player_id,
            selected_count=selected_count,
            type_filter_idx_map=type_filter_idx_map,
            all_results=all_results,
            retry_missing=retry_missing,
            refresh_retry_evidence=refresh_retry_evidence,
        )
        if num_positions == 0:
            continue

        cost = cost_calculator.estimate_position_benchmark_cost(
            player_id,
            model_name=config.get("model_name"),
            num_positions=num_positions,
            reasoning=config_uses_reasoning(config),
            use_budget_overrides=True,
        )
        if cost is None:
            unknown_players.append(player_id)
        else:
            known_cost += cost

    return known_cost, unknown_players


def estimate_player_benchmark_cost(
    player_id: str,
    config: dict,
    *,
    num_positions: int,
    cost_calculator: CostCalculator,
) -> float | None:
    """Estimate cost for one selected player and position count."""
    if config_is_engine(config):
        return 0.0
    return cost_calculator.estimate_position_benchmark_cost(
        player_id,
        model_name=config.get("model_name"),
        num_positions=num_positions,
        reasoning=config_uses_reasoning(config),
        use_budget_overrides=True,
    )


def calculate_actual_benchmark_cost(
    player_id: str,
    config: dict,
    token_usage: dict,
    *,
    num_positions: int,
    cost_calculator: CostCalculator,
    num_model_calls: int | None = None,
) -> float | None:
    """Calculate token-priced cost for one completed position benchmark run."""
    if config_is_engine(config):
        return 0.0

    override_cost = cost_calculator.estimate_position_benchmark_cost(
        player_id,
        model_name=config.get("model_name"),
        num_positions=num_model_calls if num_model_calls is not None else num_positions,
        reasoning=config_uses_reasoning(config),
        use_budget_overrides=True,
    )
    if cost_calculator.get_budget_cost_override(player_id) is not None:
        return override_cost

    model_name = config.get("model_name") or cost_calculator.get_model_for_player(player_id)
    if not model_name:
        return None

    return cost_calculator.calculate_game_cost(
        {
            "prompt_tokens": int(token_usage.get("prompt", 0) or 0),
            "completion_tokens": int(token_usage.get("completion", 0) or 0),
        },
        model_name,
    )


async def main():
    parser = argparse.ArgumentParser(description="Run position benchmark")
    parser.add_argument(
        "--positions",
        type=Path,
        default=CORE_POSITIONS_PATH,
        help="Panel positions JSON (default: required 50-position production core)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CORE_RESULTS_PATH,
        help="Panel results JSON (default: canonical core results)",
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
        "--player-id",
        help="Ad-hoc player id to use with --model-name, without editing config/benchmark.yaml",
    )
    parser.add_argument(
        "--model-name",
        help="Ad-hoc model name to use with --player-id, without editing config/benchmark.yaml",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for an ad-hoc LLM player",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable reasoning for an ad-hoc LLM player",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for an ad-hoc LLM player",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds for an ad-hoc LLM player",
    )
    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help="Only run positions missing from existing results (skipped or errored)",
    )
    parser.add_argument(
        "--refresh-retry-evidence",
        action="store_true",
        help=(
            "Only rerun selected positions whose existing row lacks current "
            "conditional-retry evidence"
        ),
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
        default="all",
        help="Legacy mixed-file filter; canonical panel files are already single-purpose",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only test the first N positions after filtering (useful for cheap smoke tests)",
    )
    parser.add_argument(
        "--position-indices",
        nargs="+",
        type=int,
        help="Specific panel-local position indices to test",
    )
    parser.add_argument(
        "--sync-firestore",
        action="store_true",
        help="Sync results to Firestore even when using a custom output file.",
    )
    parser.add_argument(
        "--max-estimated-cost",
        type=float,
        help=(
            "Abort before API calls if the worst-case estimate, including one "
            "conditional retry per selected position, exceeds this dollar amount"
        ),
    )
    parser.add_argument(
        "--allow-unknown-cost",
        action="store_true",
        help="Allow API calls with unknown cost when --max-estimated-cost is set",
    )
    parser.add_argument(
        "--allow-legacy-input",
        action="store_true",
        help="Allow an explicitly inactive legacy positions file for historical analysis",
    )

    args = parser.parse_args()
    if args.retry_missing and args.refresh_retry_evidence:
        raise ValueError("Use only one of --retry-missing and --refresh-retry-evidence")
    default_output = CORE_RESULTS_PATH.resolve()
    should_sync_firestore = args.sync_firestore or args.output.resolve() == default_output

    # Load positions
    print(f"Loading positions from {args.positions}...")
    with open(args.positions) as f:
        data = json.load(f)

    try:
        validate_position_input(data, allow_legacy_input=args.allow_legacy_input)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    all_positions = data["positions"]
    panel_metadata = data.get("metadata", {})
    selected_orig_indices = [
        idx
        for idx, position in enumerate(all_positions)
        if not args.type or args.type == "all" or position.get("type") == args.type
    ]
    if args.position_indices:
        requested = set(args.position_indices)
        selected_orig_indices = [idx for idx in selected_orig_indices if idx in requested]
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        selected_orig_indices = selected_orig_indices[: args.limit]

    type_filter_idx_map = {new_idx: orig_idx for new_idx, orig_idx in enumerate(selected_orig_indices)}
    positions = [all_positions[orig_idx] for orig_idx in selected_orig_indices]
    if not positions:
        raise ValueError("No positions selected; check the panel file and filters")
    blunder_count = sum(1 for p in positions if p.get("type") == "blunder")
    equal_count = sum(1 for p in positions if p.get("type") == "equal")
    print(
        f"Testing on {len(positions)} selected positions "
        f"({blunder_count} blunder, {equal_count} equal)"
    )
    if panel_metadata.get("panel") == "core-equal":
        print("Production core: exactly 50 equal positions; optional panels are excluded.")

    # Load player configs
    all_players = load_player_configs()
    ad_hoc_config = build_ad_hoc_player_config(args)
    if ad_hoc_config:
        all_players[ad_hoc_config["player_id"]] = ad_hoc_config
        if not args.players:
            args.players = [ad_hoc_config["player_id"]]

    # Determine which players to test
    if args.players:
        missing_players = [p for p in args.players if p not in all_players]
        if missing_players:
            raise ValueError(
                f"Unknown player(s): {', '.join(missing_players)}. "
                "Use --player-id and --model-name for an ad-hoc model."
            )
        players_to_test = {p: all_players[p] for p in args.players}
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

    # Load existing results to merge with (don't overwrite)
    all_results = {}
    if Path(args.output).exists():
        with open(args.output) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results from {args.output}")

    cost_calculator = CostCalculator()
    estimated_cost, unknown_cost_players = estimate_selected_benchmark_cost(
        players_to_test,
        selected_count=len(positions),
        type_filter_idx_map=type_filter_idx_map,
        all_results=all_results,
        retry_missing=args.retry_missing,
        cost_calculator=cost_calculator,
        refresh_retry_evidence=args.refresh_retry_evidence,
    )
    retry_upper_bound_cost = estimated_cost * 2
    if estimated_cost or unknown_cost_players:
        unknown_suffix = (
            f"; unknown for {', '.join(unknown_cost_players)}"
            if unknown_cost_players
            else ""
        )
        print(
            "Estimated API cost for selected positions: "
            f"${estimated_cost:.4f} base; "
            f"${retry_upper_bound_cost:.4f} conditional-retry upper bound"
            f"{unknown_suffix}"
        )
    if args.max_estimated_cost is not None:
        if retry_upper_bound_cost > args.max_estimated_cost:
            raise SystemExit(
                f"Worst-case estimated API cost ${retry_upper_bound_cost:.4f} exceeds "
                f"--max-estimated-cost ${args.max_estimated_cost:.4f}"
            )
        if unknown_cost_players and not args.allow_unknown_cost:
            raise SystemExit(
                "Unknown API cost for "
                f"{', '.join(unknown_cost_players)}; pass --allow-unknown-cost to continue"
            )

    # Initialize Stockfish for evaluation after budget preflight passes.
    print("\nStarting Stockfish...")
    stockfish = chess.engine.SimpleEngine.popen_uci("stockfish")

    try:
        for player_id, config in players_to_test.items():
            print(f"\n{'='*60}")
            print(f"Testing: {player_id}")
            print(f"{'='*60}")

            # Determine which positions to test
            positions_to_test = positions
            existing_by_idx = {}

            incremental_mode = args.retry_missing or args.refresh_retry_evidence
            if incremental_mode and player_id in all_results:
                existing = all_results[player_id].get("results", [])
                # Stored position_idx is in all_positions coordinates (after any
                # type_filter remap). Key existing_by_idx by that so we can merge
                # new subset results back in.
                existing_by_idx = {r["position_idx"]: r for r in existing}
                satisfied_indices = {
                    idx
                    for idx, row in existing_by_idx.items()
                    if not args.refresh_retry_evidence
                    or row.get("conditional_retry_protocol_version")
                    == CONDITIONAL_RETRY_PROTOCOL_VERSION
                }

                def _filtered_to_all(fi: int) -> int:
                    return type_filter_idx_map[fi]

                missing_filtered = [fi for fi in range(len(positions))
                                    if _filtered_to_all(fi) not in satisfied_indices]

                if not missing_filtered:
                    completed_label = (
                        "current retry evidence"
                        if args.refresh_retry_evidence
                        else "results"
                    )
                    print(
                        f"  All {len(positions)} positions already have "
                        f"{completed_label}, skipping"
                    )
                    continue

                # Map subset index (the index within positions_to_test, 0..N-1)
                # directly to all_positions index. Used below to rewrite result
                # position_idx in one step, bypassing the type_filter_idx_map remap.
                retry_idx_map = {si: _filtered_to_all(fi) for si, fi in enumerate(missing_filtered)}
                positions_to_test = [positions[fi] for fi in missing_filtered]
                reason = "missing retry measurements" if args.refresh_retry_evidence else "missing positions"
                print(
                    f"  Retrying {len(positions_to_test)} {reason} "
                    f"(have {len(existing_by_idx)} existing rows)"
                )

            start = time.time()
            result = await run_benchmark(player_id, positions_to_test, stockfish, config, args.depth, args.api)
            elapsed = time.time() - start
            run_estimated_cost = estimate_player_benchmark_cost(
                player_id,
                config,
                num_positions=len(positions_to_test),
                cost_calculator=cost_calculator,
            )
            run_token_usage = result.get("run_token_usage", result.get("token_usage", {}))
            run_model_calls = int(result.get("run_model_calls", 0) or 0)
            run_actual_cost = calculate_actual_benchmark_cost(
                player_id,
                config,
                run_token_usage,
                num_positions=len(result.get("results", [])),
                cost_calculator=cost_calculator,
                num_model_calls=run_model_calls or None,
            )

            # Remap filtered indices back to original position indices.
            # In either incremental mode, retry_idx_map maps subset index directly
            # to all_positions index, so skip the type-filter remap.
            if incremental_mode and player_id in all_results:
                for r in result["results"]:
                    r["position_idx"] = retry_idx_map[r["position_idx"]]
            else:
                for r in result["results"]:
                    r["position_idx"] = type_filter_idx_map[r["position_idx"]]

            # Merge refreshed/missing rows with existing results.
            if incremental_mode and existing_by_idx:
                for r in result["results"]:
                    existing_by_idx[r["position_idx"]] = r
                merged_results = [existing_by_idx[i] for i in sorted(existing_by_idx.keys())]

                # Recalculate summary on merged results
                pos_type_by_idx = {i: p.get("type") for i, p in enumerate(all_positions)}

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
                summary["result_schema_version"] = RESULT_SCHEMA_VERSION
                stamp_benchmark_summary(summary, merged_results, args.depth)
                attach_conditional_retry_summary(summary, merged_results)
                summary["positions_attempted"] = len(all_positions)
                summary["positions_skipped"] = max(
                    0,
                    len(all_positions) - len(merged_results),
                )

                blunder_r = [r for r in wrapped if pos_type_by_idx.get(r.position_idx) == "blunder"]
                equal_r = [r for r in wrapped if pos_type_by_idx.get(r.position_idx) == "equal"]
                if blunder_r:
                    summary["blunder"] = _calc(blunder_r)
                if equal_r:
                    summary["equal"] = _calc(equal_r)

                result = {
                    "summary": summary,
                    "results": merged_results,
                    "token_usage": sum_result_row_tokens(merged_results),
                    "run_token_usage": run_token_usage,
                    "run_model_calls": run_model_calls,
                }

            summary = result["summary"]
            if run_estimated_cost is not None:
                summary["run_estimated_api_cost"] = run_estimated_cost
                summary["run_estimated_base_api_cost"] = run_estimated_cost
                summary["run_estimated_retry_upper_bound_api_cost"] = run_estimated_cost * 2
            if run_actual_cost is not None:
                summary["run_actual_api_cost"] = run_actual_cost
            summary["run_prompt_tokens"] = run_token_usage.get("prompt", 0)
            summary["run_completion_tokens"] = run_token_usage.get("completion", 0)
            summary["run_model_calls"] = run_model_calls
            print(f"\nSummary for {player_id}:")
            print(f"  Legal moves: {summary['legal_moves']}/{summary['total_positions']} ({summary['legal_pct']:.1f}%)")
            print(f"  Best moves: {summary['best_moves']} ({summary['best_pct']:.1f}%)")
            print(f"  Avoided blunders: {summary['avoided_blunders']} ({summary['avoided_pct']:.1f}%)")
            print(f"  Avg CPL: {summary['avg_cpl']:.1f}")
            retry_summary = summary.get("conditional_retry")
            if retry_summary:
                recovery = retry_summary.get("recovery_pct")
                recovery_text = "n/a" if recovery is None else f"{recovery:.1f}%"
                print(
                    "  Conditional retries: "
                    f"{retry_summary['retry_recoveries']}/"
                    f"{retry_summary['retry_attempts']} recovered ({recovery_text})"
                )
            if run_estimated_cost is not None or run_actual_cost is not None:
                estimated_text = "unknown" if run_estimated_cost is None else f"${run_estimated_cost:.4f} base"
                actual_text = "unknown" if run_actual_cost is None else f"${run_actual_cost:.4f}"
                print(f"  API cost: estimated {estimated_text}, actual {actual_text}")
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
