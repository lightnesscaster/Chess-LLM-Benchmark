"""Estimate leaderboard Elo and RD jointly from completed rated games."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit

from game.models import GameResult
from game.stats_collector import StatsCollector
from rating.anchor_history import anchor_rating_at
from rating.glicko2 import Glicko2System, PlayerRating
from rating.rating_store import BENCHMARK_SEED_RD, RatingStore
from utils import is_reasoning_model

REASONING_START_RATING = 1200
NON_REASONING_START_RATING = 400
LOW_LEGAL_MOVE_RATING = 0
LEGAL_MOVE_THRESHOLD_LOW = 0.90
LEGAL_MOVE_THRESHOLD_MED = 0.95
WHITE_PRIOR_SD = 100.0
SCALE = Glicko2System().SCALE_FACTOR
WHITE_SCORE = {"white": 1., "draw": .5, "black": 0.}


def _eligible(game: GameResult) -> bool:
    if not game.white_id or not game.black_id or game.white_id == game.black_id or game.winner not in WHITE_SCORE:
        return False
    if game.game_type != "human_challenge":
        return True
    return (f"lichess:{game.human_lichess_username}" in (game.white_id, game.black_id)
            and game.human_rating is not None and np.isfinite(game.human_rating)
            and 0 < (game.human_rating_deviation or 0) <= 500)


def _initial_elo(name: str, legal_rate: float) -> float:
    if legal_rate < LEGAL_MOVE_THRESHOLD_LOW:
        return LOW_LEGAL_MOVE_RATING
    if legal_rate < LEGAL_MOVE_THRESHOLD_MED or not is_reasoning_model(name):
        return NON_REASONING_START_RATING
    return REASONING_START_RATING


def fit_leaderboard(store: RatingStore, results: Sequence[GameResult], engines: list[dict]) -> None:
    """Fit the public table in memory. The caller owns this store and never saves it."""
    if not results:
        return
    games = [g for g in results if _eligible(g)]
    anchors = {e["player_id"]: e for e in engines if e.get("anchor", True)}
    ghosts = {e["player_id"] for e in engines if e.get("ghost")}
    priors = store.benchmark_predictions()
    stats = StatsCollector()
    stats.add_results([g for g in games if g.game_type != "human_challenge"])
    legality = stats.get_player_stats()
    records: dict[str, Counter] = {}
    fitted: dict[str, tuple[float, float]] = {}
    batches = ([], [])

    for game in games:
        players = (game.white_id, game.black_id)
        human = f"lichess:{game.human_lichess_username}" if game.game_type == "human_challenge" else None
        fixed = {}
        for name, side in zip(players, ("white", "black")):
            if name == human:
                fixed[name] = game.human_rating
                continue
            record = records.setdefault(name, Counter())
            outcome = "draws" if game.winner == "draw" else "wins" if game.winner == side else "losses"
            record[outcome] += 1
            if name in anchors:
                fixed[name] = anchor_rating_at(anchors[name], game.created_at)
            elif name not in fitted:
                mean = priors.get(name, _initial_elo(name, legality.get(name, {}).get("legal_move_rate", 1.)))
                fitted[name] = (mean, BENCHMARK_SEED_RD if name in priors else 350.)
        batches[any(name in ghosts for name in players)].append((players, fixed, WHITE_SCORE[game.winner]))

    white_bonus = 0.
    for is_ghost, batch in enumerate(batches):
        names = sorted(n for n in fitted if (n in ghosts) == is_ghost)
        if not names or not batch:
            continue
        learn_white = not is_ghost
        indices = {n: i for i, n in enumerate(names)}
        row_ids, cols, values, offsets, scores = [], [], [], [], []
        for players, fixed, score in batch:
            if not any(n in indices for n in players):
                continue
            row = len(scores)
            offset = white_bonus
            for name, sign in zip(players, (1, -1)):
                if name in indices:
                    row_ids.append(row)
                    cols.append(indices[name])
                    values.append(sign)
                else:
                    rating = fixed[name] if name in fixed else fitted[name][0]
                    offset += sign * rating / SCALE
            if learn_white:
                row_ids.append(row)
                cols.append(len(names))
                values.append(1.)
            offsets.append(offset)
            scores.append(score)
        if not scores:
            continue
        prior = [fitted[n] for n in names]
        if learn_white:
            prior.append((0., WHITE_PRIOR_SD))
        mean, sd = np.array(prior).T / SCALE
        precision = 1 / sd**2
        design = csr_matrix((values, (row_ids, cols)), shape=(len(scores), len(mean)))
        y = np.asarray(scores)
        shift = np.asarray(offsets)

        def objective(v):
            z = design @ v + shift
            loss = np.logaddexp(0., z).sum() - y @ z + .5 * precision @ ((v - mean)**2)
            return loss, design.T @ (expit(z) - y) + precision * (v - mean)

        opt = minimize(objective, mean, jac=True, method="L-BFGS-B",
                       options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 1000})
        if not opt.success:
            raise RuntimeError(f"Leaderboard Elo fit failed: {opt.message}")
        p = expit(design @ opt.x + shift)
        information = (design.T @ design.multiply((p*(1-p))[:, None])).toarray() + np.diag(precision)
        covariance = cho_solve(cho_factor(information), np.eye(len(mean)))
        rd = SCALE * np.sqrt(covariance.diagonal())
        for i, name in enumerate(names):
            fitted[name] = (float(opt.x[i]*SCALE), float(rd[i]))
        if learn_white:
            white_bonus = float(opt.x[-1])

    current = store.get_all()
    for name, counts in records.items():
        base = current.get(name) or PlayerRating(name)
        elo, rd = (anchors[name]["rating"], base.rating_deviation) if name in anchors else fitted[name]
        store.set(replace(base, rating=elo, rating_deviation=rd, unclamped_rating=elo,
                          games_played=sum(counts.values()), wins=counts["wins"], draws=counts["draws"],
                          losses=counts["losses"]), auto_save=False)
