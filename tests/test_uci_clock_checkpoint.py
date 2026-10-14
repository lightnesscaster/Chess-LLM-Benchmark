"""Clock checkpoints retain real search limits across process replacement."""

import copy
import json
import math
import threading
from types import SimpleNamespace
from unittest import mock

import chess
import pytest

from engines.uci_engine import UCIEngine


def test_clock_checkpoint_round_trip_retains_remaining_times_and_increment():
    original = UCIEngine("anchor", 2000, "engine", initial_time=1800,
                         increment=2, time_scale=0.25)
    original._white_clock = 112.125
    original._black_clock = 298.375
    state = json.loads(json.dumps(original.snapshot_clock()))
    restored = UCIEngine("anchor", 2000, "engine", initial_time=1800,
                         increment=2, time_scale=0.25)
    restored.restore_clock(state)

    limits = []
    restored._engine = SimpleNamespace(play=lambda board, limit: (
        limits.append(limit) or SimpleNamespace(move=chess.Move.from_uci("e2e4"))))
    with mock.patch("engines.uci_engine.time.perf_counter", side_effect=[10, 13]):
        restored.select_move(chess.Board())
    assert (limits[0].white_clock, limits[0].black_clock,
            limits[0].white_inc, limits[0].black_inc) == (112.125, 298.375, 0.5, 0.5)
    assert restored._white_clock == 109.625
    assert restored._black_clock == 298.375
    assert restored.game_time_control == (1800, 2)


def test_snapshot_initializes_deferred_clock_once_and_restores_without_probe_or_draw():
    options = dict(player_id="anchor", rating=2000, engine_path="engine",
                   time_controls=[[1200, 10, 1], [1800, 2, 1]], reference_nps=500000)
    original = UCIEngine(**options)
    def probe():
        original.measured_nps = 2000000
        original.time_scale = 0.25
    with mock.patch.object(original, "_measure_nps", side_effect=probe), \
         mock.patch("engines.uci_engine.random.choices", return_value=[[1800, 2, 1]]):
        state = original.snapshot_clock()
    # A repeated snapshot must not re-probe or choose a new control.
    with mock.patch.object(original, "_measure_nps", side_effect=AssertionError("probe")), \
         mock.patch("engines.uci_engine.random.choices", side_effect=AssertionError("draw")):
        assert original.snapshot_clock() == state
        restored = UCIEngine(**options)
        restored.restore_clock(state)
        assert restored.snapshot_clock() == state
    assert state["white_clock"] == 450
    assert state["increment"] == 0.5
    assert state["measured_nps"] == 2000000
    assert state["reference_nps"] == 500000


def test_clock_checkpoint_rejects_changed_engine_configuration_without_mutation():
    original = UCIEngine("anchor", 2000, "engine", initial_time=1800, increment=2)
    state = original.snapshot_clock()
    for config in ({"engine_path": "other"}, {"increment": 3}, {"options": {"Hash": 64}}):
        args = dict(player_id="anchor", rating=2000, engine_path="engine",
                    initial_time=1800, increment=2)
        args.update(config)
        restored = UCIEngine(**args)
        before = restored.snapshot_clock()
        with pytest.raises(ValueError):
            restored.restore_clock(state)
        assert restored.snapshot_clock() == before


@pytest.mark.parametrize("key,value", [
    ("version", 2), ("version", True), ("player_id", "other"),
    ("white_clock", -1), ("white_clock", math.inf), ("black_clock", math.nan),
    ("white_clock", 10 ** 1000),
    ("white_clock", "10"), ("black_clock", True), ("increment", -1),
    ("increment", 7), ("time_scale", 0), ("time_scale", math.inf),
    ("game_time_control", [1800]), ("game_time_control", [900, 2]),
    ("game_time_control", [1800, math.nan]), ("measured_nps", 0),
    ("measured_nps", math.inf), ("reference_nps", 500000),
])
def test_clock_checkpoint_rejects_malformed_values_without_mutation(key, value):
    engine = UCIEngine("anchor", 2000, "engine", initial_time=1800, increment=2)
    before = engine.snapshot_clock()
    malformed = copy.deepcopy(before)
    malformed[key] = value
    with pytest.raises(ValueError):
        engine.restore_clock(malformed)
    assert engine.snapshot_clock() == before


def test_clock_checkpoint_requires_all_fields_and_an_object():
    engine = UCIEngine("anchor", 2000, "engine", initial_time=1800)
    state = engine.snapshot_clock()
    for malformed in (None, [], {}, {k: v for k, v in state.items() if k != "white_clock"}):
        with pytest.raises(ValueError):
            engine.restore_clock(malformed)


def test_fixed_limit_engine_has_no_clock_checkpoint():
    engine = UCIEngine("anchor", 2000, "engine", depth=5)
    assert engine.snapshot_clock() is None
    with pytest.raises(ValueError):
        engine.restore_clock({})


@pytest.mark.parametrize("key,value", [
    ("measured_nps", None), ("measured_nps", 0), ("measured_nps", 1000000),
    ("reference_nps", None), ("reference_nps", True), ("time_scale", 1),
])
def test_reference_clock_rejects_inconsistent_speed_without_starting_engine(key, value):
    engine = UCIEngine("anchor", 2000, "engine", initial_time=1800, reference_nps=500000)
    engine.measured_nps = 2000000
    engine.time_scale = 0.25
    state = engine.snapshot_clock()
    state[key] = value
    restored = UCIEngine("anchor", 2000, "engine", initial_time=1800, reference_nps=500000)
    with pytest.raises(ValueError):
        restored.restore_clock(state)
    assert restored._white_clock is None
    assert restored.measured_nps is None
    assert restored._engine is None


@pytest.mark.parametrize("quit_error", [None, TimeoutError("engine did not quit")])
def test_close_releases_engine_transport_even_when_quit_fails(quit_error):
    released = threading.Event()

    class Transport:
        def quit(self):
            if quit_error is not None:
                raise quit_error

        def close(self):
            released.set()

    engine = UCIEngine("anchor", 2000, "engine", depth=5)
    engine._engine = Transport()
    if quit_error is None:
        engine.close()
    else:
        with pytest.raises(TimeoutError):
            engine.close()
    assert released.is_set()
    assert engine._engine is None
    engine.close()  # Closing a released transport is harmless.
