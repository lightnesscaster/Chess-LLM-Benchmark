"""Prepare server-side CLI credentials without exposing their contents."""

from __future__ import annotations

import base64
import binascii
import json
import os
from collections.abc import Mapping
from pathlib import Path


def prepare_codex_auth(environ: Mapping[str, str] = os.environ) -> Path | None:
    """Seed Codex's persistent auth cache from a base64 Render secret."""
    encoded_auth = environ.get("CODEX_AUTH_JSON_B64", "").strip()
    if not encoded_auth:
        return None

    codex_home = Path(environ.get("CODEX_HOME", "/var/data/codex"))
    auth_path = codex_home / "auth.json"
    codex_home.mkdir(mode=0o700, parents=True, exist_ok=True)
    codex_home.chmod(0o700)

    if auth_path.is_file() and auth_path.stat().st_size > 0:
        auth_path.chmod(0o600)
        return auth_path

    try:
        decoded = base64.b64decode(encoded_auth, validate=True)
        payload = json.loads(decoded)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("CODEX_AUTH_JSON_B64 is not valid encoded JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError("CODEX_AUTH_JSON_B64 is not valid encoded JSON.")

    serialized = json.dumps(payload, separators=(",", ":")).encode()
    descriptor = os.open(
        auth_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as auth_file:
            auth_file.write(serialized)
    except Exception:
        auth_path.unlink(missing_ok=True)
        raise
    auth_path.chmod(0o600)
    return auth_path


if __name__ == "__main__":
    prepare_codex_auth()
