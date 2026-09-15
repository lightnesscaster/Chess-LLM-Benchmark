import json
import logging

from flask import Flask


def test_snapshot_reads_rss_and_cgroup_without_process_arguments(tmp_path):
    from web.memory_diagnostics import memory_snapshot

    proc = tmp_path / "proc"
    process = proc / "42"
    process.mkdir(parents=True)
    (process / "status").write_text("Name:\tpython\nPid:\t42\nPPid:\t1\nVmRSS:\t2048 kB\n")
    (process / "cmdline").write_text("secret-prompt-and-token")
    cg = tmp_path / "cgroup"
    cg.mkdir()
    (cg / "memory.current").write_text("4096")
    (cg / "memory.max").write_text("536870912")
    (cg / "memory.events").write_text("low 0\nhigh 0\nmax 3\noom 2\noom_kill 1\n")
    result = memory_snapshot(proc, cg)
    assert result["processes"] == [{"pid": 42, "ppid": 1, "name": "python", "rss_bytes": 2097152}]
    assert result["container"]["current_bytes"] == 4096
    assert result["container"]["limit_bytes"] == 536870912
    assert result["container"]["events"]["oom_kill"] == 1
    assert "secret" not in json.dumps(result)


def test_snapshot_tolerates_missing_files_and_unlimited_memory(tmp_path):
    from web.memory_diagnostics import memory_snapshot

    (tmp_path / "memory.max").write_text("max")
    result = memory_snapshot(tmp_path / "missing", tmp_path)
    assert result["processes"] == []
    assert result["container"]["current_bytes"] is None
    assert result["container"]["limit_bytes"] is None


def test_snapshot_uses_nested_container_cgroup(tmp_path):
    from web.memory_diagnostics import memory_snapshot

    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/service\n")
    cg = tmp_path / "cgroup"
    (cg / "service").mkdir(parents=True)
    (cg / "memory.current").write_text("9000")
    (cg / "service/memory.current").write_text("1000")
    assert memory_snapshot(proc, cg)["container"]["current_bytes"] == 1000


def test_monitor_logs_during_work_and_stops_on_exception(caplog, monkeypatch):
    import threading
    from web import memory_diagnostics as diagnostics

    sampled = threading.Event()
    calls = []

    def snapshot():
        calls.append(1)
        if len(calls) >= 2:
            sampled.set()
        return {"container": {"events": {"oom_kill": 0}}, "processes": []}

    monkeypatch.setattr(diagnostics, "memory_snapshot", snapshot)
    app = Flask(__name__)
    diagnostics.install_memory_diagnostics(app, interval=0.01)
    calls.clear()

    @app.post("/api/play/move")
    def move():
        assert sampled.wait(2)
        raise RuntimeError("private-provider-message")

    with caplog.at_level(logging.INFO, logger="web.memory_diagnostics"):
        response = app.test_client().post("/api/play/move", json={"token": "private-token"})
    assert response.status_code == 500
    records = [json.loads(r.message.removeprefix("Memory diagnostics: "))
               for r in caplog.records if r.name == "web.memory_diagnostics"]
    phases = [r["phase"] for r in records]
    assert "start" in phases and "pending" in phases and phases[-1] == "end"
    assert len({r["request_id"] for r in records}) == 1
    assert records[-1]["status"] == 500
    assert "private" not in json.dumps(records)
    assert not any(t.name.startswith("move-memory-") for t in threading.enumerate())


def test_diagnostics_failure_does_not_break_move(monkeypatch):
    from web import memory_diagnostics as diagnostics

    def broken_snapshot():
        raise OSError("private")

    monkeypatch.setattr(diagnostics, "memory_snapshot", broken_snapshot)
    app = Flask(__name__)
    diagnostics.install_memory_diagnostics(app)
    app.add_url_rule("/api/play/move", view_func=lambda: {"ok": True}, methods=["POST"])
    assert app.test_client().post("/api/play/move").json == {"ok": True}
