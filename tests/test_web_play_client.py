import subprocess


def test_human_drag_move_is_rendered_before_model_response():
    result = subprocess.run(
        ["node", "tests/js/play_immediate_move_test.js"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_model_effort_choices_are_synced_and_submitted_separately():
    result = subprocess.run(
        ["node", "tests/js/play_effort_selector_test.js"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_current_fen_and_pgn_can_be_copied_or_downloaded():
    result = subprocess.run(
        ["node", "tests/js/play_export_test.js"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_rated_game_displays_lichess_snapshot_and_model_rating():
    result = subprocess.run(
        ["node", "tests/js/play_rating_snapshot_test.js"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_live_game_move_history_can_be_navigated_without_mutating_live_state():
    result = subprocess.run(
        ["node", "tests/js/play_move_navigation_test.js"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
