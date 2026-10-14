"""Private, atomic checkpoints for interrupted local match workers."""

import fcntl
import json
import os
import tempfile
from pathlib import Path


class CheckpointError(RuntimeError):
    """A checkpoint could not be read or durably written."""


class CheckpointStore:
    """Hold an exclusive process lock while reading and replacing one checkpoint."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        self._lock = os.fdopen(descriptor, "a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock.close()
            self._lock = None
            raise CheckpointError("This checkpoint is already in use") from exc
        return self

    def __exit__(self, *_args):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def read(self) -> dict:
        with self.path.open() as source:
            state = json.load(source)
        if not isinstance(state, dict):
            raise ValueError("Checkpoint must be a JSON object")
        return state

    def write(self, state: dict) -> None:
        """Replace only after the complete new state has reached durable storage."""
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=self.path.parent)
            with os.fdopen(descriptor, "w") as target:
                json.dump(state, target, allow_nan=False)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except Exception as exc:
            raise CheckpointError("Could not durably save game checkpoint") from exc
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
