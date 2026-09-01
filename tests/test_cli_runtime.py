"""Tests for preparing hosted CLI subscription credentials."""

import base64
import json
import stat
from pathlib import Path


def _encoded_auth(account_id: str) -> str:
    payload = json.dumps({"tokens": {"account_id": account_id}}).encode()
    return base64.b64encode(payload).decode()


def test_prepare_codex_auth_decodes_secret_with_owner_only_permissions(tmp_path):
    from web.cli_runtime import prepare_codex_auth

    codex_home = tmp_path / "codex"
    auth_path = prepare_codex_auth({
        "CODEX_HOME": str(codex_home),
        "CODEX_AUTH_JSON_B64": _encoded_auth("first-account"),
    })

    assert auth_path == codex_home / "auth.json"
    assert json.loads(auth_path.read_text())["tokens"]["account_id"] == "first-account"
    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600


def test_prepare_codex_auth_preserves_refreshed_disk_credentials(tmp_path):
    from web.cli_runtime import prepare_codex_auth

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    auth_path = codex_home / "auth.json"
    auth_path.write_text(json.dumps({"tokens": {"account_id": "refreshed"}}))

    result = prepare_codex_auth({
        "CODEX_HOME": str(codex_home),
        "CODEX_AUTH_JSON_B64": _encoded_auth("stale-seed"),
    })

    assert result == auth_path
    assert json.loads(auth_path.read_text())["tokens"]["account_id"] == "refreshed"


def test_prepare_codex_auth_rejects_malformed_secret(tmp_path):
    from web.cli_runtime import prepare_codex_auth

    with __import__("pytest").raises(ValueError, match="CODEX_AUTH_JSON_B64"):
        prepare_codex_auth({
            "CODEX_HOME": str(tmp_path / "codex"),
            "CODEX_AUTH_JSON_B64": "not-base64!",
        })


def test_render_build_installs_codex_non_interactively():
    script = Path("scripts/render_build.sh").read_text()

    assert "CODEX_NON_INTERACTIVE=1" in script
