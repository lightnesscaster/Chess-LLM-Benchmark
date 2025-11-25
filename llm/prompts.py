"""
Prompt templates for LLM chess players.
"""

import chess


def board_to_ascii(board: chess.Board) -> str:
    """
    Convert a chess board to ASCII representation.

    Args:
        board: python-chess Board object

    Returns:
        ASCII string representation of the board
    """
    rows = []
    for rank in range(8, 0, -1):
        row = []
        for file in range(8):
            square = chess.square(file, rank - 1)
            piece = board.piece_at(square)
            row.append(piece.symbol() if piece else ".")
        rows.append(f"{rank} | " + " ".join(row))
    rows.append("  +-----------------")
    rows.append("    a b c d e f g h")
    return "\n".join(rows)


CHESS_PROMPT_TEMPLATE = """You are playing chess as {side_to_move}.

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
e2e4"""


RETRY_PROMPT_TEMPLATE = """You are playing chess as {side_to_move}.

Current position (FEN):
{fen}

Board:
{ascii_board}

Your previous move "{illegal_move}" was ILLEGAL. Please try again.

Your task:
- Play exactly ONE LEGAL move for {side_to_move}.
- Use UCI notation only (examples: e2e4, g1f3, e7e8q for promotion).
- Do NOT include any commentary, explanations, or additional text.

Output format:
- Only the move in UCI, e.g.:
e2e4"""


def build_chess_prompt(board: chess.Board, is_retry: bool = False,
                       illegal_move: str = None) -> str:
    """
    Build the prompt to send to the LLM.

    Args:
        board: python-chess Board object
        is_retry: Whether this is a retry after an illegal move
        illegal_move: The illegal move that was attempted (if retry)

    Returns:
        The formatted prompt string
    """
    fen = board.fen()
    ascii_board = board_to_ascii(board)
    side = "White" if board.turn == chess.WHITE else "Black"

    if is_retry and illegal_move:
        return RETRY_PROMPT_TEMPLATE.format(
            fen=fen,
            ascii_board=ascii_board,
            side_to_move=side,
            illegal_move=illegal_move,
        )
    else:
        return CHESS_PROMPT_TEMPLATE.format(
            fen=fen,
            ascii_board=ascii_board,
            side_to_move=side,
        )
