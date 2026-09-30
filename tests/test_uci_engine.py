from unittest import mock

import chess

from engines.uci_engine import UCIEngine


def test_time_scale_shrinks_clock_the_engine_sees():
    engine = UCIEngine("slow", 2000, "engine", initial_time=1800, increment=20, time_scale=0.25)

    assert engine._white_clock == 450
    assert engine.increment == 5


def test_options_are_applied_and_scaled_clock_is_sent():
    fake = mock.MagicMock()
    fake.play.return_value = mock.MagicMock(move=chess.Move.from_uci("e2e4"))
    engine = UCIEngine("slow", 2000, "engine", initial_time=100, increment=2,
                       options={"Hash": 64, "Threads": 1}, time_scale=0.5)
    with mock.patch("chess.engine.SimpleEngine.popen_uci", return_value=fake):
        engine.select_move(chess.Board())

    fake.configure.assert_called_once_with({"Hash": 64, "Threads": 1})
    limit = fake.play.call_args.args[1]
    assert (limit.white_clock, limit.white_inc) == (50, 1)


def _fake_engine(nps=2_000_000):
    fake = mock.MagicMock()
    fake.analyse.return_value = {"nps": nps}
    fake.play.return_value = mock.MagicMock(move=chess.Move.from_uci("e2e4"))
    return fake


def test_time_control_is_drawn_per_game_and_scaled():
    engine = UCIEngine("mix", 2000, "engine", time_controls=[[1200, 10, 1], [1800, 0, 1]],
                       time_scale=0.5)

    assert engine.game_time_control in {(1200, 10), (1800, 0)}
    initial, increment = engine.game_time_control
    assert (engine._white_clock, engine.increment) == (initial * 0.5, increment * 0.5)


def test_reference_nps_sets_time_scale_before_first_move():
    fake = _fake_engine(nps=2_000_000)
    engine = UCIEngine("ref", 2356, "engine", initial_time=1800, increment=0, reference_nps=500_000)
    assert engine._white_clock is None  # probe deferred until the engine runs

    with mock.patch("chess.engine.SimpleEngine.popen_uci", return_value=fake):
        engine.select_move(chess.Board())

    assert engine.time_scale == 0.25
    assert fake.play.call_args.args[1].white_clock == 450


def test_clone_gets_fresh_clock_and_reuses_measured_speed():
    template_process, clone_process = _fake_engine(), _fake_engine()
    template = UCIEngine("ref", 2356, "engine", initial_time=1800, increment=0, reference_nps=500_000)
    with mock.patch("chess.engine.SimpleEngine.popen_uci", side_effect=[template_process, clone_process]):
        first = template.clone_for_game()
        first._white_clock = 3.0  # a finished game's low clock
        second = template.clone_for_game()
        second.select_move(chess.Board())

    assert template_process.analyse.call_count == 5  # warm-up + 4 probes, measured once
    clone_process.analyse.assert_not_called()
    assert second.time_scale == 0.25
    assert clone_process.play.call_args.args[1].white_clock == 450
