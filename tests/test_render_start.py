import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("overrides,workers,threads", [({}, "1", "4"), ({"WEB_CONCURRENCY": "2", "WEB_THREADS": "2"}, "2", "2")])
def test_render_start_preserves_thread_capacity_and_allows_rollback(tmp_path, overrides, workers, threads):
    bin_dir = tmp_path / ".render/bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    gunicorn = bin_dir / "gunicorn"
    gunicorn.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    gunicorn.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in {"WEB_CONCURRENCY", "WEB_THREADS"}}
    env.update(PORT="8765", **overrides)
    script = Path(__file__).resolve().parents[1] / "scripts/render_start.sh"
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    args = result.stdout.splitlines()
    assert args[args.index("--workers") + 1] == workers
    assert args[args.index("--threads") + 1] == threads
    assert args[args.index("--timeout") + 1] == "620"
