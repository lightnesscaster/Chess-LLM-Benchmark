"""Atomic persistence and structural checks for measured panel records."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def merge_player_record(
    results_path: Path,
    player_id: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Atomically merge one keyed record into a shared JSON file.

    Player benchmarks may finish concurrently. Each save must
    therefore reload the canonical file while holding a stable interprocess lock;
    otherwise the last process to finish can silently discard earlier results.
    """
    results_path = results_path.resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    lock_digest = hashlib.sha256(str(results_path).encode()).hexdigest()[:20]
    lock_path = Path(tempfile.gettempdir()) / f"chess_stability_{lock_digest}.lock"

    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            output: dict[str, Any] = {}
            if results_path.exists():
                with results_path.open() as existing:
                    output = json.load(existing)
            output[player_id] = record

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=results_path.parent,
                prefix=f".{results_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                json.dump(output, temporary, indent=2)
                temporary.write("\n")
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, results_path)
            return output
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def validate_probe_record(record: dict[str, Any]) -> None:
    """Reject incomplete or API-failed records before canonical persistence."""
    summary = record.get("summary") or {}
    rows = record.get("results") or []
    expected_indices = summary.get("selected_position_indices") or []
    actual_indices = [row.get("position_idx") for row in rows]

    if not expected_indices:
        raise ValueError("probe record has no selected position indices")
    if actual_indices != expected_indices:
        raise ValueError(
            f"incomplete probe rows: expected {expected_indices}, got {actual_indices}"
        )
    if int(summary.get("attempted_positions", 0) or 0) != len(expected_indices):
        raise ValueError("probe attempted-position count is incomplete")
    if int(summary.get("api_errors", 0) or 0) != 0 or any(
        row.get("termination") == "api_error" for row in rows
    ):
        raise ValueError("probe contains API errors")

    score_depth = int(summary.get("score_depth", 0) or 0)
    if any(int(row.get("score_depth", -1)) != score_depth for row in rows):
        raise ValueError("probe row score-depth stamps do not match its summary")
