"""Manifest-backed automatic acquisition policy for production benchmark panels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from position_benchmark.layout import MANIFEST_PATH
from position_benchmark.predictions import (
    PredictionReadiness,
    benchmark_result_readiness,
    stability_probe_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_AUTOMATIC_PANELS = (
    "core",
    "game_like",
    "continuation_stability",
)


@dataclass(frozen=True)
class AcquisitionPanel:
    """One automatically acquired production or downside-only panel."""

    name: str
    runner: str
    positions_path: Path
    results_path: Path
    position_count: int
    score_depth: int
    planned_first_attempt_calls: int
    retry_protocol_version: str
    probe_plies: int = 0
    random_seed: int = 0


@dataclass(frozen=True)
class AcquisitionPolicy:
    """Ordered automatic panel-acquisition contract."""

    version: str
    panels: tuple[AcquisitionPanel, ...]
    defer_games_after_acquisition: bool


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _repo_path(value: str) -> Path:
    return ROOT / value


def load_acquisition_policy(manifest_path: Path = MANIFEST_PATH) -> AcquisitionPolicy:
    """Load and validate the automatic acquisition policy from the manifest."""
    manifest = _load_json(manifest_path)
    config = manifest.get("automatic_acquisition", {})
    if config.get("enabled") is not True:
        raise ValueError("automatic benchmark acquisition must be enabled")

    panel_order = tuple(config.get("panel_order", ()))
    if panel_order != SUPPORTED_AUTOMATIC_PANELS:
        raise ValueError(
            "automatic panel order must be core, game_like, continuation_stability"
        )

    panels: list[AcquisitionPanel] = []
    retry_protocol_version = str(manifest["conditional_retry_protocol_version"])
    for name in panel_order:
        panel = manifest["panels"][name]
        if panel.get("acquisition") != "automatic":
            raise ValueError(f"{name} must declare acquisition=automatic")
        if name == "continuation_stability":
            panels.append(
                AcquisitionPanel(
                    name=name,
                    runner="continuation",
                    positions_path=_repo_path(panel["starting_positions"]),
                    results_path=_repo_path(panel["results"]),
                    position_count=int(panel["default_starting_positions"]),
                    score_depth=int(panel["stockfish_score_depth"]),
                    planned_first_attempt_calls=int(
                        panel["planned_base_first_attempt_calls"]
                    ),
                    retry_protocol_version=retry_protocol_version,
                    probe_plies=int(panel["probe_plies"]),
                    random_seed=int(panel["random_seed"]),
                )
            )
        else:
            panels.append(
                AcquisitionPanel(
                    name=name,
                    runner="positions",
                    positions_path=_repo_path(panel["positions"]),
                    results_path=_repo_path(panel["results"]),
                    position_count=int(panel["position_count"]),
                    score_depth=int(panel["stockfish_depth"]),
                    planned_first_attempt_calls=int(
                        panel["model_call_count"]["base_first_attempt_calls"]
                    ),
                    retry_protocol_version=retry_protocol_version,
                )
            )

    return AcquisitionPolicy(
        version=str(config["policy_version"]),
        panels=tuple(panels),
        defer_games_after_acquisition=bool(
            config.get("defer_games_after_acquisition", True)
        ),
    )


def panel_readiness(
    panel: AcquisitionPanel,
    player_id: str,
) -> PredictionReadiness:
    """Return whether a player's saved panel result satisfies the current contract."""
    if not panel.positions_path.exists():
        return PredictionReadiness(False, "missing positions file")
    if not panel.results_path.exists():
        return PredictionReadiness(False, "missing results file")

    try:
        positions_data = _load_json(panel.positions_path)
        results = _load_json(panel.results_path)
    except (json.JSONDecodeError, OSError, TypeError):
        return PredictionReadiness(False, "unreadable panel artifacts")

    record = results.get(player_id) if isinstance(results, dict) else None
    if not isinstance(record, dict):
        return PredictionReadiness(False, "missing player result")

    return _record_readiness(panel, record, positions_data)


def _record_readiness(
    panel: AcquisitionPanel,
    record: dict[str, Any],
    positions_data: Any,
) -> PredictionReadiness:
    """Check one already-loaded record against one already-loaded panel."""

    if panel.runner == "continuation":
        readiness = stability_probe_readiness(
            record,
            min_positions=panel.position_count,
            min_score_depth=panel.score_depth,
        )
    else:
        positions = (
            positions_data.get("positions", [])
            if isinstance(positions_data, dict)
            else positions_data
        )
        readiness = benchmark_result_readiness(
            record,
            positions,
            min_equal_positions=panel.position_count,
            min_stockfish_depth=panel.score_depth,
        )

    if not readiness.is_ready:
        return readiness
    # Retry evidence is additive for static rows and does not change their current
    # rating formula. Do not force expensive core/game-like reruns solely because
    # an otherwise-current historical row predates conditional retry measurement.
    # Every newly acquired static row still records the retry protocol.
    if panel.runner == "positions":
        return readiness
    rows = record.get("results", [])
    if any(
        row.get("conditional_retry_protocol_version")
        != panel.retry_protocol_version
        for row in rows
    ):
        return PredictionReadiness(False, "missing current retry evidence")
    summary = record.get("summary", {})
    summary_retry_version = (
        summary.get("conditional_retry_protocol_version")
        or summary.get("conditional_retry", {}).get("protocol_version")
    )
    if summary_retry_version != panel.retry_protocol_version:
        return PredictionReadiness(False, "missing current retry summary")
    return readiness


def load_acquisition_state(
    policy: AcquisitionPolicy,
) -> dict[str, set[str]]:
    """Load all ready player/panel pairs without repeatedly parsing result files."""
    state: dict[str, set[str]] = {}
    for panel in policy.panels:
        if not panel.positions_path.exists() or not panel.results_path.exists():
            continue
        try:
            positions_data = _load_json(panel.positions_path)
            results = _load_json(panel.results_path)
        except (json.JSONDecodeError, OSError, TypeError):
            continue
        if not isinstance(results, dict):
            continue
        for player_id, record in results.items():
            if not isinstance(record, dict):
                continue
            if _record_readiness(panel, record, positions_data).is_ready:
                state.setdefault(player_id, set()).add(panel.name)
    return state


def missing_panels(
    player_id: str,
    policy: AcquisitionPolicy,
    state: dict[str, set[str]] | None = None,
) -> list[AcquisitionPanel]:
    """Return automatic panels that are absent, stale, or incomplete for a player."""
    if state is not None:
        completed = state.get(player_id, set())
        return [panel for panel in policy.panels if panel.name not in completed]
    return [
        panel for panel in policy.panels if not panel_readiness(panel, player_id).is_ready
    ]
