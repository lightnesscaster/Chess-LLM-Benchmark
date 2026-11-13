import subprocess


def test_stockfish_searches_are_serialized_and_stale_results_are_ignored():
    result = subprocess.run(["node", "tests/js/analysis_engine_test.js"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
