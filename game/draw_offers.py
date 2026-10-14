"""Validate and annotate explicit, two-player draw negotiations in PGN."""

from __future__ import annotations

import chess.pgn


def annotate_draw_events(game: chess.pgn.Game, events: list[dict] | None) -> None:
    """Add draw comments after validating their players, move and offer sequence.

    Offers follow a legal move. Acceptance or refusal annotates that same move;
    a subsequent move must have consumed the previous offer. No board move is
    fabricated for an acceptance. Legacy games simply have no events.
    """
    if events is None:
        return
    nodes = [game, *game.mainline()]
    if not isinstance(events, list) or len(events) > 3 * len(nodes):
        raise ValueError("Invalid draw event history")
    pending = None
    last_ply = 0
    last_offer_ply = 0
    accepted = False
    annotations = []
    for event in events:
        if not isinstance(event, dict) or set(event) != {"action", "by", "ply"}:
            raise ValueError("Invalid draw event")
        action, side, ply = event["action"], event["by"], event["ply"]
        if (not isinstance(action, str) or not isinstance(side, str)
                or action not in {"offer", "accept", "decline"} or side not in {"white", "black"}
                or type(ply) is not int or not 0 < ply < len(nodes)
                or ply < last_ply or accepted):
            raise ValueError("Invalid draw event")
        if action == "offer":
            board = nodes[ply].board()
            mover = "black" if board.turn == chess.WHITE else "white"
            if pending is not None or mover != side or board.is_game_over() or ply <= last_offer_ply:
                raise ValueError("Invalid draw offer")
            pending = event
            last_offer_ply = ply
        else:
            if pending is None or side == pending["by"] or ply != pending["ply"]:
                raise ValueError("Draw response requires the opponent's pending offer")
            pending = None
            accepted = action == "accept"
            if accepted and ply != len(nodes) - 1:
                raise ValueError("Moves cannot follow an accepted draw")
        annotations.append((nodes[ply], f"[%draw_{action} {side}]"))
        last_ply = ply
    if pending is not None and pending["ply"] != len(nodes) - 1:
        raise ValueError("A later move must decline the previous draw offer")
    for node, annotation in annotations:
        node.comment = " ".join(part for part in (node.comment, annotation) if part)
