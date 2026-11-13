import subprocess


def test_stockfish_searches_are_serialized_and_stale_results_are_ignored():
    result = subprocess.run(["node", "tests/js/analysis_engine_test.js"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_variations_follow_inline_notation_order_and_export_nested_pgn():
    result = subprocess.run(["node", "tests/js/analysis_notation_test.js"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_board_selection_clears_after_moves_escape_outside_clicks_and_invalid_drags():
    result = subprocess.run(["node", "tests/js/analysis_board_selection_test.js"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
