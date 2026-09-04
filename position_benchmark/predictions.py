"""Production position-benchmark readiness and rating prediction helpers.

Keep these rules synchronized with ``position_benchmark/README.md``, the canonical
methodology and validation record.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


CURRENT_BENCHMARK_VERSION = "history-replay-v2"
DEFAULT_MIN_EQUAL_POSITIONS = 50
DEFAULT_MIN_BLUNDER_POSITIONS = 25
DEFAULT_MIN_GAME_LIKE_POSITIONS = 48
DEFAULT_MIN_STABILITY_POSITIONS = 8
DEFAULT_MIN_STABILITY_SCORED_MOVES = 24
DEFAULT_MIN_STABILITY_SCORE_DEPTH = 30
DEFAULT_MIN_STOCKFISH_DEPTH = 30
CURRENT_STABILITY_PROBE_VERSION = "stratified-depth30-v3"
CURRENT_STABILITY_SELECTION_POLICY = "category-round-robin-v1"
CURRENT_STABILITY_POSITION_INDICES = (0, 12, 24, 36, 1, 13, 25, 37)
DEFAULT_GAME_LIKE_CPL_CAP = 5000.0
GAME_LIKE_CAP_TRIGGER = 150.0
STABILITY_RISK_TRIGGER = 5.0
STABILITY_CAP_BASE = 650.0
STABILITY_FORFEIT_PENALTY = 6.0
STABILITY_CATASTROPHE_PENALTY = 8.0
STABILITY_CAP_FLOOR = -500.0


@dataclass(frozen=True)
class EqualPositionMetrics:
    """Aggregated metrics used by the benchmark rating predictor."""

    total: int
    equal_cpl: float
    best_pct: float
    legal_pct: float
    skipped_mismatched_fen: int = 0


@dataclass(frozen=True)
class PredictionReadiness:
    """Whether a model's benchmark result is safe to use as a rating seed."""

    is_ready: bool
    reason: str


@dataclass(frozen=True)
class EqualPositionCoverage:
    """Coverage diagnostics for equal-position result rows."""

    expected: int
    unique_valid: int
    duplicate_rows: int
    missing: int
    stale_rows: int
    mismatched_fen: int


@dataclass(frozen=True)
class StabilityProbeMetrics:
    """Aggregated continuation-probe metrics used as a downside cap."""

    attempted_positions: int
    scored_moves: int
    legal_moves: int
    attempts: int
    legal_pct: float
    forfeit_pct: float
    avg_cpl: float | None
    p90_cpl: float | None
    catastrophe_pct: float
    deduplicated_catastrophe_pct: float


@dataclass(frozen=True)
class CombinedLegalityMetrics:
    """Cross-panel legality evidence used by the game-like downside check."""

    panel_legal_pcts: tuple[float, ...]
    pooled_legal_pct: float
    conservative_legal_pct: float
    legal_moves: int
    attempts: int


def resolve_result_position_index(
    row: dict[str, Any],
    positions: list[dict[str, Any]],
) -> int | None:
    """Resolve a result row to a panel-local index, including legacy rows.

    New rows use a stable ``position_id`` and a panel-local ``position_idx``.
    Historical rows may only contain an index into the old combined 75-position
    registry. FEN validation disambiguates overlapping local and legacy indices.
    """
    position_id = row.get("position_id")
    if position_id:
        for index, position in enumerate(positions):
            if position.get("position_id") == position_id:
                return index

    row_index = row.get("position_idx")
    if not isinstance(row_index, int):
        return None

    row_fen = row.get("fen")
    if 0 <= row_index < len(positions):
        local_position = positions[row_index]
        if not row_fen or row_fen == local_position.get("fen"):
            return row_index

    for index, position in enumerate(positions):
        if position.get("legacy_position_idx") == row_index:
            return index
    return None


def result_row_is_current(
    row: dict[str, Any],
    *,
    min_stockfish_depth: int = DEFAULT_MIN_STOCKFISH_DEPTH,
) -> bool:
    """Return whether one result row was produced by the current benchmark path."""
    if row.get("position_benchmark_version") != CURRENT_BENCHMARK_VERSION:
        return False
    if row.get("prompt_history_replay") is not True:
        return False

    depth = row.get("stockfish_depth", row.get("reevaluated_depth"))
    try:
        return depth is not None and int(depth) >= min_stockfish_depth
    except (TypeError, ValueError):
        return False


def survival_probability(legal_pct: float, game_length: int = 40) -> float:
    """Current production survival proxy, which implicitly assumes retry q=0."""
    illegal_rate = max(0.0, min(1.0, 1.0 - legal_pct / 100.0))
    if illegal_rate <= 0:
        return 100.0
    if illegal_rate >= 1:
        return 0.0

    return 100.0 * (
        (1.0 - illegal_rate) ** game_length
        + game_length * illegal_rate * (1.0 - illegal_rate) ** (game_length - 1)
    )


def two_strike_survival_probability(
    legal_pct: float,
    retry_failure_pct: float,
    game_length: int = 40,
) -> float:
    """Survival probability under the runner's two-strikes-per-game policy.

    This is a research metric until its prediction coefficients are validated.
    A game survives with either zero first-attempt illegals, or exactly one
    first-attempt illegal followed by a legal retry.
    """
    p = max(0.0, min(1.0, 1.0 - legal_pct / 100.0))
    q = max(0.0, min(1.0, retry_failure_pct / 100.0))
    if game_length <= 0:
        return 100.0
    return 100.0 * (
        (1.0 - p) ** game_length
        + game_length * p * (1.0 - p) ** (game_length - 1) * (1.0 - q)
    )


def predict_rating(equal_cpl: float, best_pct: float, legal_pct: float) -> float:
    """Predict game rating from equal-position CPL, exact-best rate, and legal rate."""
    return (
        1298.57
        - 200.43 * math.log(max(0.0, equal_cpl) + 1.0)
        + 15.39 * best_pct
        + 5.85 * survival_probability(legal_pct)
    )


def benchmark_result_readiness(
    model_data: dict[str, Any],
    positions: list[dict[str, Any]],
    *,
    min_equal_positions: int = DEFAULT_MIN_EQUAL_POSITIONS,
    min_stockfish_depth: int = DEFAULT_MIN_STOCKFISH_DEPTH,
    position_type: str = "equal",
) -> PredictionReadiness:
    """
    Check whether one model's benchmark result is fresh enough for rating seeding.

    Legacy rows are still useful for diagnostics, but rows generated before
    history replay should not seed ratings or be treated as authoritative
    prediction evidence.
    """
    coverage = collect_equal_position_coverage(
        model_data.get("results", []),
        positions,
        min_stockfish_depth=min_stockfish_depth,
        position_type=position_type,
    )
    position_label = position_type or "selected"
    if coverage.stale_rows:
        return PredictionReadiness(False, f"stale {position_label}-position rows")
    if coverage.mismatched_fen:
        return PredictionReadiness(False, "mismatched FEN rows")
    if coverage.duplicate_rows:
        return PredictionReadiness(False, f"duplicate {position_label}-position rows")
    if coverage.unique_valid < min_equal_positions:
        return PredictionReadiness(False, f"{position_label} positions < {min_equal_positions}")
    if coverage.missing:
        return PredictionReadiness(False, f"missing {position_label}-position rows")

    metrics = collect_equal_position_metrics(
        model_data.get("results", []),
        positions,
        position_type=position_type,
    )
    if metrics is None:
        return PredictionReadiness(False, "no equal-position rows")

    return PredictionReadiness(True, "ready")


def collect_equal_position_coverage(
    model_results: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    *,
    min_stockfish_depth: int = DEFAULT_MIN_STOCKFISH_DEPTH,
    position_type: str = "equal",
) -> EqualPositionCoverage:
    """Inspect valid, missing, duplicate, stale, and mismatched equal-position rows."""
    expected_indices = {
        idx for idx, position in enumerate(positions) if position.get("type") == position_type
    }
    seen_indices: set[int] = set()
    duplicate_rows = 0
    stale_rows = 0
    mismatched_fen = 0

    for row in model_results:
        idx = resolve_result_position_index(row, positions)
        if idx is None or idx not in expected_indices:
            continue

        result_fen = row.get("fen")
        if result_fen and result_fen != positions[idx].get("fen"):
            mismatched_fen += 1
            continue

        if idx in seen_indices:
            duplicate_rows += 1
        else:
            seen_indices.add(idx)

        if not result_row_is_current(row, min_stockfish_depth=min_stockfish_depth):
            stale_rows += 1

    return EqualPositionCoverage(
        expected=len(expected_indices),
        unique_valid=len(seen_indices),
        duplicate_rows=duplicate_rows,
        missing=len(expected_indices - seen_indices),
        stale_rows=stale_rows,
        mismatched_fen=mismatched_fen,
    )


def collect_equal_position_metrics(
    model_results: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    *,
    cpl_cap: float | None = None,
    position_type: str = "equal",
) -> EqualPositionMetrics | None:
    """
    Collect equal-position metrics from one model's result rows.

    Historical benchmark rows can outlive position re-evaluations. Trust the
    current positions file for position type and best move, and skip rows whose
    stored FEN points at a different position.
    """
    eq_cpls: list[float] = []
    best_count = 0
    legal_count = 0
    skipped_mismatched_fen = 0
    seen_indices: set[int] = set()

    for result in model_results:
        idx = resolve_result_position_index(result, positions)
        if idx is None or idx < 0 or idx >= len(positions):
            continue

        position = positions[idx]
        if position.get("type") != position_type:
            continue

        result_fen = result.get("fen")
        if result_fen and result_fen != position.get("fen"):
            skipped_mismatched_fen += 1
            continue
        if idx in seen_indices:
            continue
        seen_indices.add(idx)

        cpl = float(result.get("cpl", 0.0))
        if cpl_cap is not None:
            cpl = min(max(0.0, cpl), cpl_cap)
        eq_cpls.append(cpl)
        if result.get("model_move") == position.get("best_move"):
            best_count += 1
        if result.get("is_legal", True):
            legal_count += 1

    total = len(eq_cpls)
    if total == 0:
        return None

    return EqualPositionMetrics(
        total=total,
        equal_cpl=sum(eq_cpls) / total,
        best_pct=100.0 * best_count / total,
        legal_pct=100.0 * legal_count / total,
        skipped_mismatched_fen=skipped_mismatched_fen,
    )


def predict_rating_from_results(
    model_results: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    *,
    cpl_cap: float | None = None,
    position_type: str = "equal",
) -> float | None:
    """Compute the benchmark rating prediction for one model's result rows."""
    metrics = collect_equal_position_metrics(
        model_results,
        positions,
        cpl_cap=cpl_cap,
        position_type=position_type,
    )
    if metrics is None:
        return None
    return predict_rating(metrics.equal_cpl, metrics.best_pct, metrics.legal_pct)


def predict_rating_from_model_data(
    model_data: dict[str, Any],
    positions: list[dict[str, Any]],
    *,
    require_ready: bool = True,
    min_equal_positions: int = DEFAULT_MIN_EQUAL_POSITIONS,
    min_stockfish_depth: int = DEFAULT_MIN_STOCKFISH_DEPTH,
    cpl_cap: float | None = None,
    position_type: str = "equal",
) -> float | None:
    """Compute a model prediction, optionally requiring production readiness."""
    if require_ready:
        readiness = benchmark_result_readiness(
            model_data,
            positions,
            min_equal_positions=min_equal_positions,
            min_stockfish_depth=min_stockfish_depth,
            position_type=position_type,
        )
        if not readiness.is_ready:
            return None
    return predict_rating_from_results(
        model_data.get("results", []),
        positions,
        cpl_cap=cpl_cap,
        position_type=position_type,
    )


def combine_prediction_with_downside_cap(
    prediction: float,
    downside_cap: float | None,
    *,
    cap_trigger: float = GAME_LIKE_CAP_TRIGGER,
) -> float:
    """Apply a supplemental downside cap only when it is materially lower."""
    if downside_cap is None:
        return prediction
    if downside_cap + cap_trigger < prediction:
        return downside_cap
    return prediction


def combine_equal_and_game_like_predictions(
    equal_prediction: float,
    game_like_prediction: float | None,
    *,
    cap_trigger: float = GAME_LIKE_CAP_TRIGGER,
) -> float:
    """
    Combine the canonical equal-position prediction with a game-like panel.

    The game-like panel is a downside check, not an upside booster. It only caps
    the equal-position seed when it is materially lower, because the observed
    failure mode is optimistic puzzle scores for models that do not convert that
    strength in game-like positions.
    """
    return combine_prediction_with_downside_cap(
        equal_prediction,
        game_like_prediction,
        cap_trigger=cap_trigger,
    )


def stability_probe_readiness(
    model_data: dict[str, Any],
    *,
    min_positions: int = DEFAULT_MIN_STABILITY_POSITIONS,
    min_scored_moves: int = DEFAULT_MIN_STABILITY_SCORED_MOVES,
    min_score_depth: int = DEFAULT_MIN_STABILITY_SCORE_DEPTH,
) -> PredictionReadiness:
    """Check whether a continuation-probe summary is strong enough to cap ratings."""
    summary = model_data.get("summary", model_data)
    if (
        summary.get("stability_probe_version") != CURRENT_STABILITY_PROBE_VERSION
        or summary.get("position_selection_policy") != CURRENT_STABILITY_SELECTION_POLICY
    ):
        return PredictionReadiness(False, "outdated stability position selection")
    if tuple(summary.get("selected_position_indices") or ()) != CURRENT_STABILITY_POSITION_INDICES:
        return PredictionReadiness(False, "unexpected stability position indices")
    try:
        attempted = int(summary.get("attempted_positions") or 0)
        scored = int(summary.get("model_scored_moves") or 0)
        move_attempts = int(summary.get("model_attempts") or 0)
        forfeits = int(summary.get("model_forfeits") or 0)
        score_depth = int(summary.get("score_depth") or 0)
    except (TypeError, ValueError):
        return PredictionReadiness(False, "invalid stability summary")

    if attempted < min_positions:
        return PredictionReadiness(False, f"stability positions < {min_positions}")
    enough_forfeit_evidence = forfeits >= max(2, min_positions // 4)
    if scored < min_scored_moves and move_attempts < min_scored_moves and not enough_forfeit_evidence:
        return PredictionReadiness(False, f"stability scored moves < {min_scored_moves}")
    if score_depth < min_score_depth:
        return PredictionReadiness(
            False,
            f"stability score depth < {min_score_depth}",
        )
    if int(summary.get("api_errors", 0) or 0) != 0:
        return PredictionReadiness(False, "stability probe contains API errors")
    return PredictionReadiness(True, "ready")


def collect_stability_probe_metrics(model_data: dict[str, Any]) -> StabilityProbeMetrics | None:
    """Collect continuation-probe summary metrics, accepting either full or compact data."""
    summary = model_data.get("summary", model_data)
    try:
        scored_moves = int(summary.get("model_scored_moves") or 0)
        catastrophe_pct = float(
            summary.get("model_1000cp_catastrophe_pct") or 0.0
        )
        catastrophe_positions = summary.get(
            "model_1000cp_catastrophe_positions"
        )
        deduplicated_catastrophe_pct = catastrophe_pct
        if catastrophe_positions is not None and scored_moves:
            deduplicated_catastrophe_pct = (
                100.0 * int(catastrophe_positions) / scored_moves
            )
        return StabilityProbeMetrics(
            attempted_positions=int(summary.get("attempted_positions") or 0),
            scored_moves=scored_moves,
            legal_moves=int(summary.get("model_legal_moves") or 0),
            attempts=int(summary.get("model_attempts") or 0),
            legal_pct=float(summary.get("model_legal_pct") or 0.0),
            forfeit_pct=float(summary.get("model_forfeit_pct") or 0.0),
            avg_cpl=(
                float(summary["model_avg_cpl"])
                if summary.get("model_avg_cpl") is not None
                else None
            ),
            p90_cpl=(
                float(summary["model_p90_cpl"])
                if summary.get("model_p90_cpl") is not None
                else None
            ),
            catastrophe_pct=catastrophe_pct,
            deduplicated_catastrophe_pct=deduplicated_catastrophe_pct,
        )
    except (TypeError, ValueError):
        return None


def combine_legality_metrics(
    primary_metrics: EqualPositionMetrics,
    game_like_metrics: EqualPositionMetrics,
    stability_metrics: StabilityProbeMetrics | None = None,
) -> CombinedLegalityMetrics:
    """Combine legality evidence without allowing another panel to hide failures.

    The pooled rate is retained as a diagnostic, but averaging can make a model
    with a weak core legality rate look better merely because another context was
    easier. The game-like predictor therefore uses the lowest eligible panel rate.
    This is a one-way reliability correction: supplemental evidence can reduce,
    but never inflate, the game-like legality term.
    """
    panel_rates = [primary_metrics.legal_pct, game_like_metrics.legal_pct]
    legal_moves = round(primary_metrics.total * primary_metrics.legal_pct / 100.0)
    legal_moves += round(game_like_metrics.total * game_like_metrics.legal_pct / 100.0)
    attempts = primary_metrics.total + game_like_metrics.total

    if stability_metrics is not None:
        panel_rates.append(stability_metrics.legal_pct)
        legal_moves += stability_metrics.legal_moves
        attempts += stability_metrics.attempts

    pooled = 100.0 * legal_moves / attempts if attempts else 0.0
    return CombinedLegalityMetrics(
        panel_legal_pcts=tuple(panel_rates),
        pooled_legal_pct=pooled,
        conservative_legal_pct=min(panel_rates),
        legal_moves=legal_moves,
        attempts=attempts,
    )


def stability_probe_prediction_cap(model_data: dict[str, Any]) -> float | None:
    """
    Estimate a weak-play downside cap from continuation-probe instability.

    Equal-position CPL can overrate models that solve isolated positions but
    later forfeit or make mate-scale live-continuation mistakes. This cap is
    deliberately one-way and only activates when the probe shows material risk.
    """
    metrics = collect_stability_probe_metrics(model_data)
    if metrics is None:
        return None

    # Once a move loses at catastrophe scale, later catastrophe-scale moves in
    # that same continuation are correlated consequences rather than independent
    # risk events. Count at most one event per starting trajectory while retaining
    # the original scored-move exposure and cap coefficients. This correction can
    # only make the downside cap less severe.
    catastrophe_risk = metrics.deduplicated_catastrophe_pct
    material_risk = metrics.forfeit_pct + catastrophe_risk
    if material_risk < STABILITY_RISK_TRIGGER:
        return None

    cap = (
        STABILITY_CAP_BASE
        - STABILITY_FORFEIT_PENALTY * metrics.forfeit_pct
        - STABILITY_CATASTROPHE_PENALTY * catastrophe_risk
    )
    return max(STABILITY_CAP_FLOOR, min(STABILITY_CAP_BASE, cap))


def predict_rating_from_model_data_with_supplement(
    model_data: dict[str, Any],
    positions: list[dict[str, Any]],
    *,
    blunder_model_data: dict[str, Any] | None = None,
    blunder_positions: list[dict[str, Any]] | None = None,
    game_like_model_data: dict[str, Any] | None = None,
    game_like_positions: list[dict[str, Any]] | None = None,
    stability_probe_model_data: dict[str, Any] | None = None,
    require_ready: bool = True,
    min_equal_positions: int = DEFAULT_MIN_EQUAL_POSITIONS,
    min_blunder_positions: int = DEFAULT_MIN_BLUNDER_POSITIONS,
    min_game_like_positions: int = DEFAULT_MIN_GAME_LIKE_POSITIONS,
    min_stability_positions: int = DEFAULT_MIN_STABILITY_POSITIONS,
    min_stability_scored_moves: int = DEFAULT_MIN_STABILITY_SCORED_MOVES,
    min_stockfish_depth: int = DEFAULT_MIN_STOCKFISH_DEPTH,
    game_like_cpl_cap: float | None = DEFAULT_GAME_LIKE_CPL_CAP,
) -> float | None:
    """Compute the production prediction, optionally capped by supplemental panels."""
    equal_prediction = predict_rating_from_model_data(
        model_data,
        positions,
        require_ready=require_ready,
        min_equal_positions=min_equal_positions,
        min_stockfish_depth=min_stockfish_depth,
    )
    if equal_prediction is None:
        return None
    primary_metrics = collect_equal_position_metrics(model_data.get("results", []), positions)
    if primary_metrics is None:
        return None

    stability_readiness = None
    eligible_stability_metrics = None
    if stability_probe_model_data is not None:
        stability_readiness = stability_probe_readiness(
            stability_probe_model_data,
            min_positions=min_stability_positions,
            min_scored_moves=min_stability_scored_moves,
        )
        if not require_ready or stability_readiness.is_ready:
            eligible_stability_metrics = collect_stability_probe_metrics(
                stability_probe_model_data
            )
    # New layouts store the optional blunder panel separately. Fall back to
    # embedded blunder rows only for the legacy combined 75-position registry.
    selected_blunder_data = blunder_model_data
    selected_blunder_positions = blunder_positions
    if selected_blunder_data is None and any(
        position.get("type") == "blunder" for position in positions
    ):
        selected_blunder_data = model_data
        selected_blunder_positions = positions

    if selected_blunder_data is not None and selected_blunder_positions is not None:
        blunder_readiness = benchmark_result_readiness(
            selected_blunder_data,
            selected_blunder_positions,
            min_equal_positions=min_blunder_positions,
            min_stockfish_depth=min_stockfish_depth,
            position_type="blunder",
        )
        if blunder_readiness.is_ready:
            blunder_prediction = predict_rating_from_model_data(
                selected_blunder_data,
                selected_blunder_positions,
                require_ready=False,
                min_equal_positions=min_blunder_positions,
                min_stockfish_depth=min_stockfish_depth,
                cpl_cap=game_like_cpl_cap,
                position_type="blunder",
            )
            equal_prediction = combine_equal_and_game_like_predictions(
                equal_prediction,
                blunder_prediction,
            )

    if game_like_model_data is not None and game_like_positions is not None:
        game_like_readiness = benchmark_result_readiness(
            game_like_model_data,
            game_like_positions,
            min_equal_positions=min_game_like_positions,
            min_stockfish_depth=min_stockfish_depth,
        )
        if not require_ready or game_like_readiness.is_ready:
            game_like_metrics = collect_equal_position_metrics(
                game_like_model_data.get("results", []),
                game_like_positions,
                cpl_cap=game_like_cpl_cap,
            )
            if game_like_metrics is not None:
                legality = combine_legality_metrics(
                    primary_metrics,
                    game_like_metrics,
                    eligible_stability_metrics,
                )
                game_like_prediction = predict_rating(
                    game_like_metrics.equal_cpl,
                    game_like_metrics.best_pct,
                    legality.conservative_legal_pct,
                )
                equal_prediction = combine_equal_and_game_like_predictions(
                    equal_prediction,
                    game_like_prediction,
                )

    if stability_probe_model_data is None:
        return equal_prediction

    if require_ready:
        if stability_readiness is None or not stability_readiness.is_ready:
            return equal_prediction

    stability_cap = stability_probe_prediction_cap(stability_probe_model_data)
    return combine_prediction_with_downside_cap(equal_prediction, stability_cap)
