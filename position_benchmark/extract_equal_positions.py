"""
Extract positions with roughly equal evaluation (between +3 and -3) from games.
"""

import sys
import json
import random
import chess
import chess.pgn
import chess.engine
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def eval_to_cp(info: dict, perspective_white: bool = True) -> int:
    """Convert engine info to centipawns from white's perspective."""
    score = info.get("score")
    if score is None:
        return 0

    pov_score = score.white()
    if pov_score.is_mate():
        mate_in = pov_score.mate()
        # Encode mate as large value
        if mate_in > 0:
            return 10000 - mate_in * 10
        else:
            return -10000 - mate_in * 10
    else:
        return pov_score.score()


def extract_equal_positions(
    games_dir: Path,
    existing_positions_file: Path,
    output_file: Path,
    num_positions: int = 100,
    min_eval: int = -300,
    max_eval: int = 300,
    min_ply: int = 10,  # Skip very early positions
    stockfish_path: str = "stockfish",
    depth: int = 16,
):
    """Extract positions with equal evaluation."""

    # Load existing positions to exclude
    existing_fens = set()
    if existing_positions_file.exists():
        with open(existing_positions_file) as f:
            data = json.load(f)
            positions = data.get("positions", data.get("blunders", []))
            for pos in positions:
                existing_fens.add(pos["fen"])
        print(f"Loaded {len(existing_fens)} existing positions to exclude")

    # Find all PGN files
    pgn_files = list(games_dir.glob("*.pgn"))
    print(f"Found {len(pgn_files)} PGN files")

    # Start Stockfish
    print("Starting Stockfish...")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Collect candidate positions
    candidates = []
    games_processed = 0

    random.shuffle(pgn_files)  # Randomize order

    for pgn_path in pgn_files:
        if len(candidates) >= num_positions * 5:  # Collect 5x what we need
            break

        try:
            with open(pgn_path) as f:
                game = chess.pgn.read_game(f)

            if game is None:
                continue

            board = game.board()
            move_history = []

            for move in game.mainline_moves():
                board.push(move)
                move_history.append(board.uci(move) if len(move_history) == 0 else move.uci())

                # Skip early positions
                if board.ply() < min_ply:
                    continue

                fen = board.fen()

                # Skip if already in existing positions
                if fen in existing_fens:
                    continue

                # Skip if in check (might be tactical)
                if board.is_check():
                    continue

                # Analyze position
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
                eval_cp = eval_to_cp(info, board.turn == chess.WHITE)

                # Check if eval is in range (from side to move's perspective)
                side_eval = eval_cp if board.turn == chess.WHITE else -eval_cp

                if min_eval <= side_eval <= max_eval:
                    # Get best move
                    best_move = info.get("pv", [None])[0]
                    if best_move:
                        candidates.append({
                            "fen": fen,
                            "eval_before": side_eval,
                            "best_move": best_move.uci(),
                            "best_move_san": board.san(best_move),
                            "move_history": move_history.copy(),
                            "side_to_move": "white" if board.turn == chess.WHITE else "black",
                            "game_file": pgn_path.name,
                            "ply": board.ply(),
                        })
                        existing_fens.add(fen)  # Don't pick same position twice

            games_processed += 1
            if games_processed % 50 == 0:
                print(f"Processed {games_processed} games, found {len(candidates)} candidates...")

        except Exception as e:
            print(f"Error processing {pgn_path.name}: {e}")
            continue

    engine.quit()

    print(f"\nFound {len(candidates)} candidate positions")

    # Randomly select the requested number
    if len(candidates) < num_positions:
        print(f"Warning: Only found {len(candidates)} positions, less than requested {num_positions}")
        selected = candidates
    else:
        selected = random.sample(candidates, num_positions)

    # Save to output file
    output_data = {
        "metadata": {
            "description": "Equal positions (eval between +3 and -3)",
            "num_positions": len(selected),
            "eval_range": f"{min_eval} to {max_eval} centipawns",
            "analysis_depth": depth,
        },
        "positions": selected,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(selected)} positions to {output_file}")

    # Print some stats
    evals = [p["eval_before"] for p in selected]
    print(f"Eval range: {min(evals)} to {max(evals)}")
    print(f"Mean eval: {sum(evals)/len(evals):.1f}")


def main():
    games_dir = Path("position_benchmark/games")
    existing_file = Path("position_benchmark/blunders.json")
    output_file = Path("position_benchmark/equal_positions.json")

    extract_equal_positions(
        games_dir=games_dir,
        existing_positions_file=existing_file,
        output_file=output_file,
        num_positions=100,
        min_eval=-300,
        max_eval=300,
    )


if __name__ == "__main__":
    main()
