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


def format_move_history(board: chess.Board) -> str:
    """
    Format the move history in standard chess notation.

    Args:
        board: python-chess Board object

    Returns:
        Move history string (e.g., "1. e4 e5 2. Nf3 Nc6")
    """
    if not board.move_stack:
        return "(Game just started - no moves yet)"

    # Create a temporary board to replay moves and get SAN notation
    temp_board = chess.Board()
    moves = []

    for i, move in enumerate(board.move_stack):
        if i % 2 == 0:
            # White's move - add move number
            move_num = (i // 2) + 1
            san = temp_board.san(move)
            moves.append(f"{move_num}. {san}")
        else:
            # Black's move
            san = temp_board.san(move)
            moves.append(san)
        temp_board.push(move)

    return " ".join(moves)


def get_last_move_info(board: chess.Board) -> str:
    """
    Get information about the opponent's last move.

    Args:
        board: python-chess Board object

    Returns:
        String describing the last move, or empty string if no moves
    """
    if not board.move_stack:
        return ""

    last_move = board.move_stack[-1]

    # Get SAN notation by temporarily going back one move
    temp_board = board.copy()
    temp_board.pop()
    san = temp_board.san(last_move)

    # Determine who made the last move (opposite of current turn)
    last_player = "Black" if board.turn == chess.WHITE else "White"

    return f"{last_player}'s last move: {san} ({last_move.uci()})"


CHESS_PROMPT_TEMPLATE = """You are playing chess as {side_to_move}.

{move_history_section}
{last_move_section}
{previous_thinking_section}
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


RETRY_PROMPT_TEMPLATE = """You are playing chess as {side_to_move}.

{move_history_section}
{last_move_section}
{previous_thinking_section}
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
b1c3"""


def build_chess_prompt(
    board: chess.Board,
    is_retry: bool = False,
    illegal_move: str = None,
    previous_response: str = None,
) -> str:
    """
    Build the prompt to send to the LLM.

    Args:
        board: python-chess Board object
        is_retry: Whether this is a retry after an illegal move
        illegal_move: The illegal move that was attempted (if retry)
        previous_response: The LLM's previous response/chain of thought

    Returns:
        The formatted prompt string
    """
    fen = board.fen()
    ascii_board = board_to_ascii(board)
    side = "White" if board.turn == chess.WHITE else "Black"

    # Build move history section
    move_history = format_move_history(board)
    move_history_section = f"Move history:\n{move_history}"

    # Build last move section
    last_move_info = get_last_move_info(board)
    last_move_section = f"\n>>> {last_move_info} <<<\n" if last_move_info else ""

    # Build previous thinking section (only include if we have a previous response)
    if previous_response and previous_response.strip():
        previous_thinking_section = f"\nYour previous thinking:\n{previous_response}\n"
    else:
        previous_thinking_section = ""

    if is_retry and illegal_move:
        return RETRY_PROMPT_TEMPLATE.format(
            fen=fen,
            ascii_board=ascii_board,
            side_to_move=side,
            illegal_move=illegal_move,
            move_history_section=move_history_section,
            last_move_section=last_move_section,
            previous_thinking_section=previous_thinking_section,
        )
    else:
        return CHESS_PROMPT_TEMPLATE.format(
            fen=fen,
            ascii_board=ascii_board,
            side_to_move=side,
            move_history_section=move_history_section,
            last_move_section=last_move_section,
            previous_thinking_section=previous_thinking_section,
        )
