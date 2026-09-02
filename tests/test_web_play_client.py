import subprocess


def test_human_drag_move_is_rendered_before_model_response():
    result = subprocess.run(
        ["node", "tests/js/play_immediate_move_test.js"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
