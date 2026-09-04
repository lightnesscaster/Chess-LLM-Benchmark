"""Research helpers for extrapolating final ratings across reasoning efforts.

The observations consumed here are final game-benchmark ratings. Their normal
position-benchmark initialization has already happened upstream; position
features are deliberately not part of this model.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EFFORTS = ("no thinking", "minimal", "low", "medium", "high", "xhigh")
EFFORT_INDEX = {effort: index for index, effort in enumerate(EFFORTS)}
EFFORT_ALIASES = {"max": "xhigh"}
EFFORT_PATTERN = re.compile(r"\(([^()]*)\)$")


@dataclass(frozen=True)
class RatingObservation:
    """One final game-benchmark rating at a named reasoning effort."""

    player_id: str
    effort: str
    rating: float
    rating_deviation: float
    games_played: int


@dataclass(frozen=True)
class ReasoningCurve:
    """All observed final ratings for one underlying model line."""

    model_id: str
    lab: str
    release_cohort: str
    observations: Mapping[str, RatingObservation]


@dataclass(frozen=True)
class CurvePrior:
    """Expected adjacent-effort increments learned without a target release."""

    increments: tuple[float, ...]
    global_increments: tuple[float, ...]
    training_lines: tuple[str, ...]
    lab_training_lines: tuple[str, ...]

    def delta(self, source_effort: str, target_effort: str) -> float:
        """Return the expected rating change between two efforts."""
        source = EFFORT_INDEX[normalize_effort(source_effort)]
        target = EFFORT_INDEX[normalize_effort(target_effort)]
        if source == target:
            return 0.0
        if source < target:
            return float(sum(self.increments[source:target]))
        return -float(sum(self.increments[target:source]))


@dataclass(frozen=True)
class ReasoningRatingPrediction:
    """A final-rating extrapolation and its provenance."""

    rating: float
    anchor_efforts: tuple[str, ...]
    target_effort: str
    combination: str
    prior: CurvePrior


def normalize_effort(effort: str) -> str:
    """Normalize supported effort aliases to the common effort ladder."""
    normalized = effort.strip().lower()
    normalized = EFFORT_ALIASES.get(normalized, normalized)
    if normalized not in EFFORT_INDEX:
        raise ValueError(f"Unsupported reasoning effort: {effort}")
    return normalized


def effort_from_player_id(player_id: str) -> str | None:
    """Extract a supported reasoning effort suffix from a player ID."""
    match = EFFORT_PATTERN.search(player_id)
    if not match:
        return None
    try:
        return normalize_effort(match.group(1))
    except ValueError:
        return None


def lab_for_model_id(model_id: str) -> str:
    """Use the provider namespace as the laboratory identifier."""
    return model_id.split("/", 1)[0]


def release_cohort_for_model_id(model_id: str) -> str:
    """Group sibling models released as one reasoning family."""
    if model_id.startswith("openai/gpt-5.6-"):
        return "openai/gpt-5.6"
    return model_id


def build_reasoning_curves(
    ratings: Mapping[str, Mapping[str, Any]],
    model_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, ReasoningCurve]:
    """Join final ratings to underlying model lines and effort suffixes."""
    grouped: dict[str, dict[str, RatingObservation]] = {}
    for player_id, rating_data in ratings.items():
        metadata = model_metadata.get(player_id)
        effort = effort_from_player_id(player_id)
        if metadata is None or effort is None:
            continue
        model_id = str(metadata["model_id"])
        observation = RatingObservation(
            player_id=player_id,
            effort=effort,
            rating=float(rating_data["rating"]),
            rating_deviation=float(rating_data["rating_deviation"]),
            games_played=int(rating_data["games_played"]),
        )
        existing = grouped.setdefault(model_id, {}).get(effort)
        if existing is None or observation.games_played > existing.games_played:
            grouped[model_id][effort] = observation

    return {
        model_id: ReasoningCurve(
            model_id=model_id,
            lab=lab_for_model_id(model_id),
            release_cohort=release_cohort_for_model_id(model_id),
            observations=dict(observations),
        )
        for model_id, observations in grouped.items()
        if len(observations) >= 2
    }


def _difference_rows(
    curves: Iterable[ReasoningCurve],
) -> list[tuple[np.ndarray, float, float, str]]:
    """Create independent-ish consecutive observed differences per line."""
    rows: list[tuple[np.ndarray, float, float, str]] = []
    for curve in curves:
        ordered = sorted(
            (EFFORT_INDEX[effort], observation)
            for effort, observation in curve.observations.items()
        )
        for (left_index, left), (right_index, right) in zip(ordered, ordered[1:]):
            design = np.zeros(len(EFFORTS) - 1, dtype=float)
            design[left_index:right_index] = 1.0
            delta = right.rating - left.rating
            left_rd = max(60.0, left.rating_deviation)
            right_rd = max(60.0, right.rating_deviation)
            weight = 1.0 / (left_rd * left_rd + right_rd * right_rd)
            rows.append((design, delta, weight, curve.model_id))
    return rows


def _fit_increments(
    rows: Sequence[tuple[np.ndarray, float, float, str]],
    *,
    prior: np.ndarray | None,
    shrinkage: float,
) -> np.ndarray:
    """Fit adjacent increments with a ridge prior and RD-based weights."""
    width = len(EFFORTS) - 1
    if not rows:
        return np.zeros(width, dtype=float) if prior is None else prior.copy()
    design = np.asarray([row[0] for row in rows], dtype=float)
    target = np.asarray([row[1] for row in rows], dtype=float)
    weights = np.asarray([row[2] for row in rows], dtype=float)
    weights /= float(np.mean(weights))
    center = np.zeros(width, dtype=float) if prior is None else prior
    penalty = 0.5 if prior is None else float(shrinkage)
    system = design.T @ (weights[:, None] * design) + penalty * np.eye(width)
    rhs = design.T @ (weights * target) + penalty * center
    return np.linalg.solve(system, rhs)


def fit_curve_prior(
    curves: Mapping[str, ReasoningCurve],
    *,
    target_model_id: str,
    exclude_release_cohort: bool = True,
    lab_shrinkage: float = 64.0,
) -> CurvePrior:
    """Fit a global-to-lab effort curve without target-line leakage.

    By default, sibling models from the same release cohort are excluded too.
    This simulates predicting a newly released family before any of its target
    effort ratings have been observed.
    """
    target_curve = curves[target_model_id]
    excluded = {target_model_id}
    if exclude_release_cohort:
        excluded.update(
            model_id
            for model_id, curve in curves.items()
            if curve.release_cohort == target_curve.release_cohort
        )
    training = [curve for key, curve in curves.items() if key not in excluded]
    global_rows = _difference_rows(training)
    global_increments = _fit_increments(
        global_rows,
        prior=None,
        shrinkage=0.5,
    )
    lab_curves = [curve for curve in training if curve.lab == target_curve.lab]
    lab_rows = _difference_rows(lab_curves)
    increments = _fit_increments(
        lab_rows,
        prior=global_increments,
        shrinkage=lab_shrinkage,
    )
    return CurvePrior(
        increments=tuple(float(value) for value in increments),
        global_increments=tuple(float(value) for value in global_increments),
        training_lines=tuple(sorted({row[3] for row in global_rows})),
        lab_training_lines=tuple(sorted({row[3] for row in lab_rows})),
    )


def predict_final_rating(
    curve: ReasoningCurve,
    *,
    anchor_efforts: Sequence[str],
    target_effort: str,
    prior: CurvePrior,
    combination: str = "latest",
) -> ReasoningRatingPrediction:
    """Predict an untested final rating from one or more final-rating anchors.

    ``latest`` uses the highest-effort anchor. ``mean`` averages the target
    rating implied independently by every supplied anchor.
    """
    target = normalize_effort(target_effort)
    anchors = tuple(normalize_effort(effort) for effort in anchor_efforts)
    if not anchors:
        raise ValueError("At least one anchor effort is required")
    missing = [effort for effort in anchors if effort not in curve.observations]
    if missing:
        raise ValueError(f"Missing anchor ratings: {', '.join(missing)}")
    ordered = tuple(sorted(set(anchors), key=EFFORT_INDEX.__getitem__))
    implied = [
        curve.observations[effort].rating + prior.delta(effort, target)
        for effort in ordered
    ]
    if combination == "latest":
        rating = implied[-1]
    elif combination == "mean":
        rating = float(np.mean(implied))
    else:
        raise ValueError(f"Unsupported anchor combination: {combination}")
    return ReasoningRatingPrediction(
        rating=float(rating),
        anchor_efforts=ordered,
        target_effort=target,
        combination=combination,
        prior=prior,
    )
