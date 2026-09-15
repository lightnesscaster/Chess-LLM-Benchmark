"""Best-effort Linux memory telemetry; never read process arguments or secrets."""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, g, request

logger = logging.getLogger(__name__)
PLAY_PATHS = {"/api/play/move", "/api/admin/play/move", "/api/play/start", "/api/admin/play/start"}


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except (OSError, UnicodeError):
        return ""


def _number(path: Path) -> int | None:
    try:
        return int(_read(path).strip())
    except ValueError:
        return None


def memory_snapshot(proc: Path = Path("/proc"), cgroup: Path = Path("/sys/fs/cgroup")) -> dict:
    """Return container counters and namespace-visible process RSS in bytes.

    Process RSS includes shared pages, so its sum is NOT container memory usage.
    Missing counters are null/empty, never interpreted as zero.
    """
    processes = []
    for path in proc.glob("[0-9]*/status"):
        fields = dict(line.split(":", 1) for line in _read(path).splitlines() if ":" in line)
        try:
            processes.append({
                "pid": int(fields["Pid"]),
                "ppid": int(fields["PPid"]),
                "name": fields.get("Name", "").strip()[:64],
                "rss_bytes": int(fields["VmRSS"].split()[0]) * 1024,
            })
        except (KeyError, ValueError, IndexError):
            continue  # Process exited while sampling, or has no resident memory.
    # Render normally exposes the container's cgroup at the mount root. Also
    # handle a unified hierarchy where /proc/self/cgroup names a nested group.
    for line in _read(proc / "self/cgroup").splitlines():
        if line.startswith("0::"):
            relative = line[3:].lstrip("/")
            if ".." not in Path(relative).parts:
                candidate = cgroup / relative
                if (candidate / "memory.current").exists():
                    cgroup = candidate
            break
    events = {}
    for line in _read(cgroup / "memory.events").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in {"low", "high", "max", "oom", "oom_kill", "oom_group_kill"}:
            try:
                events[parts[0]] = int(parts[1])
            except ValueError:
                pass
    container = {
        "current_bytes": _number(cgroup / "memory.current"),
        "peak_bytes": _number(cgroup / "memory.peak"),
        "limit_bytes": _number(cgroup / "memory.max"),
        "events": events,
    }
    return {"container": container, "processes": sorted(processes, key=lambda p: p["rss_bytes"], reverse=True)}


def _log_snapshot(**context) -> None:
    try:
        logger.info("Memory diagnostics: %s", json.dumps({**context, "worker_pid": os.getpid(), **memory_snapshot()}))
    except Exception:
        # Instrumentation must not prevent a move or expose exception payloads.
        logger.warning("Memory diagnostics unavailable")


def install_memory_diagnostics(app: Flask, interval: float = 10.0) -> None:
    """Sample play requests until teardown, including slow subprocess execution."""
    _log_snapshot(phase="worker_start")

    @app.before_request
    def start_memory_monitor():
        if request.method != "POST" or request.path not in PLAY_PATHS:
            return
        request_id = uuid.uuid4().hex[:12]
        context = {"request_id": request_id, "path": request.path}
        started = time.monotonic()
        stop = threading.Event()
        _log_snapshot(**context, phase="start", elapsed_seconds=0)

        def sample():
            while not stop.wait(interval):
                _log_snapshot(**context, phase="pending", elapsed_seconds=round(time.monotonic() - started, 2))

        thread = threading.Thread(target=sample, name="move-memory-" + request_id, daemon=True)
        g.memory_monitor = (stop, thread, context, started)
        try:
            thread.start()
        except RuntimeError:
            g.memory_monitor = None
            logger.warning("Memory monitor could not start")

    @app.after_request
    def remember_status(response):
        g.memory_response_status = response.status_code
        return response

    @app.teardown_request
    def stop_memory_monitor(error):
        monitor = g.pop("memory_monitor", None)
        if monitor is None:
            return
        stop, thread, context, started = monitor
        stop.set()
        thread.join(timeout=1)
        _log_snapshot(**context, phase="end", elapsed_seconds=round(time.monotonic() - started, 2),
                      status=g.get("memory_response_status", 500 if error else None))
