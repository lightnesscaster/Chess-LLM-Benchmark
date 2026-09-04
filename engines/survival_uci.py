#!/usr/bin/env python3
"""
UCI protocol wrapper for SurvivalEngine.

This allows survival-bot to be used with any UCI-compatible interface,
including lichess-bot, chess GUIs (Arena, Cute Chess, etc.), or other tools.

Usage:
    python -m engines.survival_uci

Or make executable:
    chmod +x engines/survival_uci.py
    ./engines/survival_uci.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
from engines.survival_engine import SurvivalEngine


class UCIWrapper:
    """UCI protocol wrapper for SurvivalEngine."""

    def __init__(self):
        self.engine: SurvivalEngine = None
        self.board = chess.Board()
        self.debug = False

    def log(self, msg: str) -> None:
        """Log debug messages to stderr."""
        if self.debug:
            print(f"info string {msg}", flush=True)

    def send(self, msg: str) -> None:
        """Send a message to the GUI."""
        print(msg, flush=True)

    def handle_uci(self) -> None:
        """Handle 'uci' command."""
        self.send("id name Survival-Bot")
        self.send("id author Chess LLM Benchmark")
        # UCI options
        self.send("option name Debug type check default false")
        self.send("option name BookPath type string default data/openings/gm2001.bin")
        self.send("option name BaseDepth type spin default 12 min 1 max 20")
        self.send("option name BlunderThreshold type spin default 300 min 100 max 1000")
        self.send("uciok")

    def handle_setoption(self, args: list[str]) -> None:
        """Handle 'setoption' command."""
        # Parse: setoption name <name> value <value>
        try:
            name_idx = args.index("name") + 1
            value_idx = args.index("value") + 1 if "value" in args else -1

            name = args[name_idx]
            value = args[value_idx] if value_idx > 0 else None

            if name.lower() == "debug":
                self.debug = value.lower() == "true" if value else False
                self.log(f"Debug set to {self.debug}")
        except (ValueError, IndexError):
            pass

    def handle_isready(self) -> None:
        """Handle 'isready' command."""
        # Initialize engine if not already done
        if self.engine is None:
            book_path = Path(__file__).parent.parent / "data" / "openings" / "gm2001.bin"
            self.engine = SurvivalEngine(
                player_id="survival-bot",
                rating=1200,
                opening_book_path=str(book_path) if book_path.exists() else None,
                base_depth=12,
                blunder_threshold=3.0,
            )
            self.log("Engine initialized")
        self.send("readyok")

    def handle_ucinewgame(self) -> None:
        """Handle 'ucinewgame' command."""
        self.board = chess.Board()
        # Engine will auto-detect new game via ply tracking
        self.log("New game started")

    def handle_position(self, args: list[str]) -> None:
        """Handle 'position' command."""
        if not args:
            return

        idx = 0
        if args[0] == "startpos":
            self.board = chess.Board()
            idx = 1
        elif args[0] == "fen":
            # Find where moves start (if any)
            fen_parts = []
            idx = 1
            while idx < len(args) and args[idx] != "moves":
                fen_parts.append(args[idx])
                idx += 1
            fen = " ".join(fen_parts)
            self.board = chess.Board(fen)

        # Apply moves if present
        if idx < len(args) and args[idx] == "moves":
            for move_str in args[idx + 1:]:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                    else:
                        self.log(f"Illegal move ignored: {move_str}")
                except ValueError:
                    self.log(f"Invalid move format: {move_str}")

        self.log(f"Position set: {self.board.fen()}")

    def handle_go(self, args: list[str]) -> None:
        """Handle 'go' command."""
        if self.engine is None:
            self.handle_isready()

        # Get best move from survival engine
        try:
            move = self.engine.select_move(self.board)
            self.send(f"bestmove {move.uci()}")
        except Exception as e:
            self.log(f"Error selecting move: {e}")
            # Fallback to first legal move
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                self.send(f"bestmove {legal_moves[0].uci()}")
            else:
                self.send("bestmove 0000")  # No legal moves

    def handle_quit(self) -> None:
        """Handle 'quit' command."""
        if self.engine:
            self.engine.close()
        sys.exit(0)

    def run(self) -> None:
        """Main UCI loop."""
        while True:
            try:
                line = input().strip()
            except EOFError:
                break

            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd == "uci":
                self.handle_uci()
            elif cmd == "setoption":
                self.handle_setoption(args)
            elif cmd == "isready":
                self.handle_isready()
            elif cmd == "ucinewgame":
                self.handle_ucinewgame()
            elif cmd == "position":
                self.handle_position(args)
            elif cmd == "go":
                self.handle_go(args)
            elif cmd == "quit":
                self.handle_quit()
            elif cmd == "stop":
                pass  # We don't support pondering, so nothing to stop
            elif cmd == "debug":
                self.debug = args[0].lower() == "on" if args else False
            else:
                self.log(f"Unknown command: {cmd}")


def main():
    wrapper = UCIWrapper()
    wrapper.run()


if __name__ == "__main__":
    main()
