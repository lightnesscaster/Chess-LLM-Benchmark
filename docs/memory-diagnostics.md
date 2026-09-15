# Web-play memory diagnostics

`web.memory_diagnostics` logs one structured `Memory diagnostics:` record at
worker startup, at the start/end of play start/move requests, and every ten
seconds while those requests are pending. Request IDs tie samples together;
worker PID distinguishes processes. No prompts, arguments, environment values,
user identities, or model response text are collected.

On Linux with cgroup v2, records include `memory.current`, `memory.peak`,
`memory.max`, and `memory.events` counters. Missing/unsupported counters are
null or empty (including on macOS), not zero. Per-process RSS comes from
namespace-visible `/proc/*/status` and includes the CLI child while running.
RSS includes shared pages: do not sum process RSS to infer container usage.
Compare container usage to its limit instead.

An increase in `oom_kill` is evidence of an OOM kill; high RSS alone is not.
Counters/peaks may reset if the platform recreates the cgroup. A hard kill can
prevent the final log from being emitted, and a spike between samples can be
missed. These records do not guarantee retrospective proof of every restart.

The Render startup script defaults to one Gunicorn worker and four threads,
reducing duplicated application state while retaining concurrent request
handling. `WEB_CONCURRENCY` and `WEB_THREADS` override these defaults; the prior
configuration was `WEB_CONCURRENCY=2 WEB_THREADS=2`. Threads still allow
overlapping model subprocesses, so this is not a memory/concurrency hard cap.

After deployment, compare warmed-up usage and pending-move peaks with the old
367–393 MiB baseline and 512 MiB container limit. Check that public pages stay
responsive during a model move. No ratings or game records need migration.
