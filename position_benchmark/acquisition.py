"""Manifest-backed automatic acquisition policy for production benchmark panels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from position_benchmark.evaluation import SCORING_VERSION, scoring_profile_identity
from position_benchmark.layout import BLUNDER_POSITIONS_PATH, BLUNDER_RESULTS_PATH, MANIFEST_PATH
from rating.rating_store import PositionBenchmarkInputs, load_position_benchmark_inputs
from position_benchmark.panel_records import validate_probe_record
from position_benchmark.predictions import (
    PredictionReadiness,
    benchmark_result_readiness,
    stability_probe_readiness,
    predict_rating_from_model_data_with_supplement,
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
    screen_efforts: tuple[str, ...]


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
        screen_efforts=tuple(config.get("supplement_deferral", {}).get("effort_order", ())),
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
    try:
        validate_probe_record(record)
    except (TypeError, ValueError, AttributeError) as error:
        return PredictionReadiness(False, str(error))
    rows = record.get("results", [])
    if record["summary"].get("probe_plies") != panel.probe_plies or any(
        row.get("probe_plies_requested") != panel.probe_plies for row in rows
    ):
        return PredictionReadiness(False, "unexpected continuation length")
    starts = positions_data.get("positions", []) if isinstance(positions_data, dict) else positions_data
    for row in rows:
        idx = row["position_idx"]
        if not isinstance(idx, int) or not 0 <= idx < len(starts):
            return PredictionReadiness(False, "unknown continuation starting position")
        if any(row.get(key) != starts[idx].get(key) for key in ("position_id", "fen")):
            return PredictionReadiness(False, "mismatched continuation starting position")
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


class AcquisitionState(dict[str, set[str]]):
    """Measured panels and, separately, the evidence supporting any deferral."""

    def __init__(self) -> None:
        super().__init__()
        self.deferred_sources: dict[str, str] = {}
        self.local_ready_panels: dict[str, set[str]] = {}


def _supplement(
    inputs: PositionBenchmarkInputs, player_id: str, name: str, fallback: dict | None,
) -> dict | None:
    core = inputs.core_results.get(player_id)
    remote = core.get("supplements", {}) if isinstance(core, dict) else {}
    return remote.get(name, (fallback or {}).get(player_id))


def player_panel_records(
    inputs: PositionBenchmarkInputs, player_id: str,
) -> tuple[dict | None, dict | None, dict | None]:
    """Return actual core, game-like and continuation records in acquisition order."""
    return (
        inputs.core_results.get(player_id),
        _supplement(inputs, player_id, "game_like", inputs.game_like_results),
        _supplement(inputs, player_id, "stability", inputs.stability_results),
    )


def deferred_supplement_source(
    player_id: str, inputs: PositionBenchmarkInputs, *, policy: AcquisitionPolicy | None = None,
) -> str | None:
    """Name the measured lower effort that supports deferring both new panels."""
    family, _, suffix = player_id.rpartition(" (")
    effort = suffix.removesuffix(")")
    if not family or not suffix.endswith(")"):
        return None
    core, game_like, stability = player_panel_records(inputs, player_id)
    if not core or game_like is not None or stability is not None:
        return None
    positions = (inputs.core_positions, inputs.game_like_positions, inputs.game_like_positions)
    if not all(positions):
        return None
    policy = policy or load_acquisition_policy()
    if effort not in policy.screen_efforts or not _record_readiness(policy.panels[0], core, positions[0]).is_ready:
        return None

    def seed(pid: str, game_panel: dict | None = None, probes: dict | None = None) -> float | None:
        return predict_rating_from_model_data_with_supplement(
            inputs.core_results[pid], inputs.core_positions,
            blunder_model_data=_supplement(inputs, pid, "blunder", inputs.blunder_results),
            blunder_positions=inputs.blunder_positions,
            game_like_model_data=game_panel, game_like_positions=inputs.game_like_positions,
            stability_probe_model_data=probes,
        )

    for lower in reversed(policy.screen_efforts[:policy.screen_efforts.index(effort)]):
        source = f"{family} ({lower})"
        records = player_panel_records(inputs, source)
        if not all(
            record and _record_readiness(panel, record, pos).is_ready
            for panel, record, pos in zip(policy.panels, records, positions)
        ):
            continue
        # The nearest fully measured sibling decides. Never shop for a more
        # favorable sibling if its panels bind or its scoring is incompatible.
        summaries = [record.get("summary", {}) for record in (core, *records)]
        if any(s.get("player_id") != pid for s, pid in zip(summaries, (player_id, source, source, source))):
            return None
        if any(s.get("scoring_version") != SCORING_VERSION or not s.get("scoring_profile")
               for s in summaries):
            return None
        try:
            profiles = [scoring_profile_identity(s["scoring_profile"]) for s in summaries]
        except (KeyError, TypeError, ValueError):
            return None
        if any(profile != profiles[0] for profile in profiles[1:]):
            return None
        for pid in (source, player_id):
            baseline = seed(pid)
            screened = seed(pid, records[1], records[2])
            if baseline is None or screened != baseline:
                return None
        return source
    return None


def _local_acquisition_inputs(policy: AcquisitionPolicy) -> PositionBenchmarkInputs:
    def read(path: Path) -> dict:
        try:
            value = _load_json(path)
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    core, game_like, stability = policy.panels
    return PositionBenchmarkInputs(
        core_results=read(core.results_path),
        core_positions=read(core.positions_path).get("positions", []),
        game_like_results=read(game_like.results_path),
        game_like_positions=read(game_like.positions_path).get("positions", []),
        stability_results=read(stability.results_path),
        blunder_results=read(BLUNDER_RESULTS_PATH),
        blunder_positions=read(BLUNDER_POSITIONS_PATH).get("positions", []),
    )


def load_acquisition_state(
    policy: AcquisitionPolicy, *, inputs: PositionBenchmarkInputs | None = None,
    use_firestore: bool = False,
) -> AcquisitionState:
    """Load measured coverage and screen eligible core-only targets."""
    if inputs is None and use_firestore:
        published = load_position_benchmark_inputs(True, published_only=True)
        if published is None:
            raise RuntimeError("Production acquisition inputs are unavailable")
        state = load_acquisition_state(policy, inputs=published)
        local = _local_acquisition_inputs(policy)
        state.local_ready_panels = dict(load_acquisition_state(policy, inputs=local))
        for target in list(state.deferred_sources):
            if any(record is not None for record in player_panel_records(local, target)[1:]):
                del state.deferred_sources[target]
        return state
    if inputs is None:
        inputs = _local_acquisition_inputs(policy)
    state = AcquisitionState()
    positions = (inputs.core_positions, inputs.game_like_positions, inputs.game_like_positions)
    players = set(inputs.core_results).union(inputs.game_like_results or {}, inputs.stability_results or {})
    for player_id in players:
        measured = {
            panel.name for panel, record, pos in zip(policy.panels, player_panel_records(inputs, player_id), positions)
            if isinstance(record, dict) and pos and _record_readiness(panel, record, pos).is_ready
        }
        if measured:
            state[player_id] = measured
        if measured == {"core"}:
            source = deferred_supplement_source(player_id, inputs, policy=policy)
            if source:
                state.deferred_sources[player_id] = source
    return state


def missing_panels(
    player_id: str, policy: AcquisitionPolicy, state: dict[str, set[str]] | None = None,
    *, full_audit: bool = False,
) -> list[AcquisitionPanel]:
    """Panels that need acquisition. An explicit audit requires real measurements."""
    if state is None:
        state = load_acquisition_state(policy)
    if not full_audit and player_id in getattr(state, "deferred_sources", {}):
        return []
    measured = state.get(player_id, set())
    return [panel for panel in policy.panels if panel.name not in measured]
