"""
Extract blunder positions from benchmark games.

A blunder is defined as a move that loses 500+ centipawns (5 pawns).
We store the position BEFORE the blunder was made, so models can be tested
on whether they avoid making the same mistake.
"""

import chess
import chess.pgn
import chess.engine
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import argparse


@dataclass
class BlunderPosition:
    """A position where a blunder was made."""
    fen: str
    move_history: list[str]  # SAN moves leading to this position
    blunder_move: str  # The bad move that was played (UCI)
    blunder_move_san: str  # SAN notation
    best_move: str  # Best move according to Stockfish (UCI)
    best_move_san: str  # SAN notation
    eval_before: float  # Eval before blunder (from side to move's perspective)
    eval_after: float  # Eval after blunder (from side to move's perspective)
    cpl_loss: float  # Centipawn loss
    game_id: str
    move_number: int
    side_to_move: str  # "white" or "black"
    player_id: str  # Who made the blunder
    opponent_id: str


def eval_to_cp(info: chess.engine.InfoDict, perspective: chess.Color) -> float:
    """Convert engine info to centipawns from perspective's view."""
    score = info.get("score")
    if score is None:
        return 0.0

    pov_score = score.pov(perspective)

    if pov_score.is_mate():
        mate_in = pov_score.mate()
        # Convert mate to large centipawn value
        if mate_in > 0:
            return 10000 - mate_in * 10  # Winning mate
        else:
            return -10000 - mate_in * 10  # Losing mate

    cp = pov_score.score()
    return cp if cp is not None else 0.0


def analyze_game(
    pgn_path: Path,
    engine: chess.engine.SimpleEngine,
    depth: int = 18,
    blunder_threshold: float = 500.0,  # 5 pawns
) -> list[BlunderPosition]:
    """
    Analyze a game and extract blunder positions.

    Returns positions where a move lost >= blunder_threshold centipawns.
    """
    blunders = []

    with open(pgn_path) as f:
        game = chess.pgn.read_game(f)

    if game is None:
        return blunders

    # Get player IDs from headers
    white_id = game.headers.get("White", "unknown")
    black_id = game.headers.get("Black", "unknown")
    game_id = pgn_path.stem

    board = game.board()
    move_history = []
    move_number = 0

    for node in game.mainline():
        move = node.move
        side_to_move = board.turn  # Before the move
        perspective = side_to_move

        # Evaluate position before the move
        info_before = engine.analyse(board, chess.engine.Limit(depth=depth))
        eval_before = eval_to_cp(info_before, perspective)
        best_move_uci = info_before.get("pv", [None])[0]

        # Make the move
        move_san = board.san(move)
        board.push(move)
        move_number += 1

        # Evaluate position after the move (negate because perspective switched)
        info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
        # After the move, it's opponent's turn, so we negate their eval
        eval_after = -eval_to_cp(info_after, not perspective)

        # Calculate centipawn loss
        cpl_loss = eval_before - eval_after

        # Check if this is a blunder
        if cpl_loss >= blunder_threshold:
            # Skip random-bot blunders - they're just noise
            blunderer = white_id if side_to_move == chess.WHITE else black_id
            if blunderer == "random-bot":
                move_history.append(move_san)
                continue

            # Get best move SAN (need to undo move first)
            board.pop()
            best_move_san = board.san(best_move_uci) if best_move_uci else "?"
            board.push(move)  # Re-apply the move

            blunder = BlunderPosition(
                fen=board.fen(),  # Position AFTER the blunder (for reference)
                move_history=move_history.copy(),  # Moves BEFORE this position
                blunder_move=move.uci(),
                blunder_move_san=move_san,
                best_move=best_move_uci.uci() if best_move_uci else "",
                best_move_san=best_move_san,
                eval_before=eval_before,
                eval_after=eval_after,
                cpl_loss=cpl_loss,
                game_id=game_id,
                move_number=move_number,
                side_to_move="white" if side_to_move == chess.WHITE else "black",
                player_id=white_id if side_to_move == chess.WHITE else black_id,
                opponent_id=black_id if side_to_move == chess.WHITE else white_id,
            )

            # Actually store the FEN BEFORE the blunder (undo move)
            board.pop()
            blunder.fen = board.fen()
            board.push(move)  # Re-apply

            # Filter: skip if still winning by +5, unless they missed mate in 4 or less
            # Mate in N is encoded as 10000 - N*10, so mate in 4 = 9960
            still_winning_big = eval_after >= 500
            missed_quick_mate = eval_before >= 9960
            if still_winning_big and not missed_quick_mate:
                move_history.append(move_san)
                continue

            blunders.append(blunder)

        move_history.append(move_san)

    return blunders


def main():
    parser = argparse.ArgumentParser(description="Extract blunder positions from games")
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=Path("position_benchmark/games"),
        help="Directory containing PGN files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("position_benchmark/blunders.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default="stockfish",
        help="Path to Stockfish binary",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=16,
        help="Stockfish analysis depth",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=500.0,
        help="Centipawn loss threshold for blunder (default: 500 = 5 pawns)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=1000,
        help="Maximum number of positions to collect",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a partial results file",
    )

    args = parser.parse_args()

    # Find all PGN files
    pgn_files = sorted(args.games_dir.glob("*.pgn"))
    print(f"Found {len(pgn_files)} PGN files")

    # Load existing results if resuming
    all_blunders = []
    processed_games = set()

    if args.resume and args.resume.exists():
        print(f"Resuming from {args.resume}...")
        with open(args.resume) as f:
            existing = json.load(f)
        all_blunders = [BlunderPosition(**b) for b in existing]
        processed_games = {b.game_id for b in all_blunders}
        print(f"  Loaded {len(all_blunders)} existing blunders from {len(processed_games)} games")

    # Filter out already processed games
    pgn_files = [p for p in pgn_files if p.stem not in processed_games]
    print(f"Will process {len(pgn_files)} remaining games")

    if not pgn_files:
        print("No new games to process!")
        return

    # Initialize Stockfish
    print(f"Starting Stockfish at depth {args.depth}...")
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)

    start_time = time.time()
    games_processed = 0

    try:
        for i, pgn_path in enumerate(pgn_files):
            elapsed = time.time() - start_time
            if games_processed > 0:
                rate = games_processed / elapsed
                remaining = len(pgn_files) - i
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_min = eta_seconds / 60
                print(f"[{i+1}/{len(pgn_files)}] {pgn_path.stem[:8]}... "
                      f"({len(all_blunders)} blunders, {rate:.1f} games/s, ETA: {eta_min:.0f}m)")
            else:
                print(f"[{i+1}/{len(pgn_files)}] {pgn_path.stem[:8]}...")

            blunders = analyze_game(
                pgn_path,
                engine,
                depth=args.depth,
                blunder_threshold=args.threshold,
            )

            for b in blunders:
                print(f"    Blunder: {b.player_id} played {b.blunder_move_san} "
                      f"(CPL: {b.cpl_loss:.0f}, best: {b.best_move_san})")

            all_blunders.extend(blunders)
            games_processed += 1

            # Save progress every 100 games
            if games_processed % 100 == 0:
                _save_progress(args.output, all_blunders)

            if len(all_blunders) >= args.max_positions:
                print(f"\nReached {args.max_positions} positions, stopping early.")
                break

    finally:
        engine.quit()

    # Sort by CPL loss (worst blunders first)
    all_blunders.sort(key=lambda b: b.cpl_loss, reverse=True)

    # Trim to max positions
    all_blunders = all_blunders[:args.max_positions]

    # Save final results
    _save_progress(args.output, all_blunders)

    print(f"\nSaved {len(all_blunders)} blunder positions to {args.output}")

    # Print summary stats
    _print_summary(all_blunders)


def _save_progress(output_path: Path, blunders: list):
    """Save current progress to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(b) for b in blunders], f, indent=2)


def _print_summary(all_blunders: list):
    """Print summary statistics."""
    if not all_blunders:
        print("No blunders found!")
        return

    player_counts = {}
    for b in all_blunders:
        player_counts[b.player_id] = player_counts.get(b.player_id, 0) + 1

    print("\nBlunders by player:")
    for player, count in sorted(player_counts.items(), key=lambda x: -x[1]):
        print(f"  {player}: {count}")

    avg_cpl = sum(b.cpl_loss for b in all_blunders) / len(all_blunders)
    print(f"\nAverage CPL loss: {avg_cpl:.0f}")
    print(f"Max CPL loss: {all_blunders[0].cpl_loss:.0f}")
    print(f"Min CPL loss: {all_blunders[-1].cpl_loss:.0f}")


if __name__ == "__main__":
    main()
