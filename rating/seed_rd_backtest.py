"""Read-only, prequential evaluation of benchmark-seeded rating uncertainty.

This intentionally does not use the production multi-pass recalculation: each
forecast sees only earlier completed outcomes. Results have completion times,
not start times: this cannot reconstruct actual forecasts at overlapping games'
starts. Fixed benchmark means and their availability are supplied by the caller.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence

from rating.anchor_history import anchor_rating_at
from rating.glicko2 import Glicko2System, PlayerRating
from rating.rating_store import _extract_effort


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def replay(
    results: Sequence[Mapping[str, Any]],
    seeds: Mapping[str, Mapping[str, Any]],
    engines: Sequence[Mapping[str, Any]],
    model_ids: Mapping[str, str],
    seed_rd: float,
) -> dict[str, Any]:
    """Score before updating; return forecasts, final state and explicit rejects.

    No filesystem, network or persistence calls are made. Human identities are
    not copied into forecasts. A two-model game may yield two player forecasts;
    these are correlated observations, not independent games.
    """
    if not 0 < seed_rd <= 350:
        raise ValueError("seed_rd must be in (0, 350]")
    ratings: dict[str, PlayerRating] = {}
    anchors = {e["player_id"]: e for e in engines if e.get("anchor", True)}
    ghosts = {e["player_id"] for e in engines if e.get("ghost", False)}
    for engine in engines:
        player_id = engine["player_id"]
        ratings[player_id] = PlayerRating(
            player_id=player_id, rating=float(engine["rating"]),
            rating_deviation=30 if player_id in anchors else 350,
            last_updated="1970-01-01T00:00:00+00:00",
        )
    glicko = Glicko2System()
    observations: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()

    def initial(player_id: str, date: str) -> PlayerRating:
        if player_id in ratings:
            return ratings[player_id]
        prediction = seeds.get(player_id)
        if prediction and _time(prediction["available_at"]) >= _time(date):
            raise ValueError(f"First game of {player_id} predates its seed lock")
        starting = [] if prediction is None else [float(prediction["rating"])]
        model_id = model_ids.get(player_id)
        effort = _extract_effort(player_id)
        if model_id and effort is not None:
            starting.extend(
                sibling.rating for sibling_id, sibling in ratings.items()
                if model_ids.get(sibling_id) == model_id
                and _extract_effort(sibling_id) is not None
                and _extract_effort(sibling_id) < effort
            )
        return PlayerRating(
            player_id=player_id, rating=max(starting) if starting else 1500,
            rating_deviation=seed_rd if starting else 350,
            last_updated=date,
        )

    for game in sorted(results, key=lambda r: (_time(r["created_at"]), r["game_id"])):
        game_id = game["game_id"]
        white_id, black_id = game.get("white_id"), game.get("black_id")
        reason = None
        if game_id in seen:
            reason = "duplicate game_id"
        elif not white_id or not black_id or white_id == black_id:
            reason = "invalid players"
        elif game.get("winner") not in ("white", "black", "draw"):
            reason = "invalid winner"
        seen.add(game_id)
        fixed: dict[str, PlayerRating] = {}
        if not reason and game.get("game_type") == "human_challenge":
            human_id = f"lichess:{game.get('human_lichess_username') or ''}"
            human_rating, human_rd = game.get("human_rating"), game.get("human_rating_deviation")
            if (human_id not in (white_id, black_id) or human_rating is None
                    or human_rd is None or not 0 < float(human_rd) <= 500
                    or not math.isfinite(float(human_rating))):
                reason = "invalid human snapshot"
            else:
                fixed[human_id] = PlayerRating(
                    player_id=human_id, rating=float(human_rating),
                    rating_deviation=float(human_rd), last_updated=game["created_at"],
                )
        if reason:
            rejected.append({"game_id": game_id, "reason": reason})
            continue
        # Construct both sides against the same pre-result state, then insert them.
        before = {pid: fixed[pid] if pid in fixed else initial(pid, game["created_at"])
                  for pid in (white_id, black_id)}
        for pid, player in before.items():
            if pid not in fixed:
                ratings[pid] = player
            if pid in anchors:
                before[pid] = PlayerRating.from_dict({
                    **player.to_dict(), "rating": anchor_rating_at(anchors[pid], game["created_at"]),
                })
        white_score = {"white": 1.0, "black": 0.0, "draw": 0.5}[game["winner"]]
        updates = {}
        for pid, opponent_id, score in ((white_id, black_id, white_score),
                                        (black_id, white_id, 1 - white_score)):
            player, opponent = before[pid], before[opponent_id]
            if pid in anchors or pid in fixed or opponent_id in ghosts:
                continue
            if pid in seeds:
                mu, _ = glicko.glicko_to_glicko2(player.rating, player.rating_deviation)
                opponent_mu, phi = glicko.glicko_to_glicko2(opponent.rating, opponent.rating_deviation)
                observations.append({
                    "game_id": game_id, "created_at": game["created_at"],
                    "player_id": pid, "family": seeds[pid]["family"], "lab": seeds[pid]["lab"],
                    "prior_games": player.games_played, "rating": player.rating,
                    "rating_deviation": player.rating_deviation,
                    "opponent_rating": opponent.rating, "opponent_rd": opponent.rating_deviation,
                    "score": score, "expected_score": glicko.e_function(mu, opponent_mu, phi),
                })
            updates[pid] = glicko.update_rating(player, [opponent], [score])
            updates[pid].last_updated = game["created_at"]
        ratings.update(updates)
    return {"observations": observations,
            "ratings": {pid: player.to_dict() for pid, player in ratings.items()},
            "rejected": rejected}


def summarize(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate squared expected-score error and fractional-score cross entropy."""
    families: dict[str, list[float]] = defaultdict(list)
    errors, entropies = [], []
    for row in observations:
        expected, score = row["expected_score"], row["score"]
        error = (expected - score) ** 2
        p = min(1 - 1e-12, max(1e-12, expected))
        errors.append(error)
        entropies.append(-score * math.log(p) - (1 - score) * math.log(1 - p))
        families[row["family"]].append(error)
    return {
        "observations": len(observations),
        "games": len({r["game_id"] for r in observations}),
        "players": len({r["player_id"] for r in observations}),
        "families": len(families),
        "brier": sum(errors) / len(errors) if errors else None,
        "cross_entropy": sum(entropies) / len(entropies) if entropies else None,
        "family_macro_brier": (sum(sum(v) / len(v) for v in families.values()) / len(families)
                               if families else None),
        "family_brier": {f: sum(v) / len(v) for f, v in sorted(families.items())},
    }
