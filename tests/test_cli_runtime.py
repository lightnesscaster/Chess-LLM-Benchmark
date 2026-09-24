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


def test_explicit_rotation_replaces_old_auth_but_preserves_subsequent_refresh(tmp_path):
    from web.cli_runtime import prepare_codex_auth

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    auth_path = codex_home / "auth.json"
    auth_path.write_text(json.dumps({"tokens": {"account_id": "expired"}}))
    env = {
        "CODEX_HOME": str(codex_home),
        "CODEX_AUTH_JSON_B64": _encoded_auth("new-login"),
        "CODEX_AUTH_REVISION": "login-20260915",
    }
    prepare_codex_auth(env)
    assert json.loads(auth_path.read_text())["tokens"]["account_id"] == "new-login"
    auth_path.write_text(json.dumps({"tokens": {"account_id": "refreshed-new-login"}}))
    prepare_codex_auth(env)
    assert json.loads(auth_path.read_text())["tokens"]["account_id"] == "refreshed-new-login"
    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600


def test_render_build_installs_codex_non_interactively():
    script = Path("scripts/render_build.sh").read_text()

    assert "CODEX_NON_INTERACTIVE=1" in script


def test_build_installer_does_not_use_runtime_credential_disk(tmp_path):
    import os
    import subprocess

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text("#!/bin/sh\nexit 0\n")
    curl = fake_bin / "curl"
    curl.write_text("#!/bin/sh\nprintf '%s\\n' 'test -z \"${CODEX_HOME:-}\" || exit 91' 'exit 92'\n")
    python.chmod(0o755)
    curl.chmod(0o755)
    env = dict(os.environ, PATH=str(fake_bin) + ":" + os.environ["PATH"], CODEX_HOME="/var/data/codex")
    result = subprocess.run(["bash", str(Path("scripts/render_build.sh").resolve())],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    # The fake installer stops the build after verifying its environment.
    assert result.returncode == 92, result.stderr


def test_render_build_tracks_latest_claude_models():
    script = Path("scripts/render_build.sh").read_text()

    assert "https://claude.ai/install.sh | bash -s latest" in script


def test_render_scripts_persist_cli_binaries_in_deploy_artifact():
    build_script = Path("scripts/render_build.sh").read_text()
    start_script = Path("scripts/render_start.sh").read_text()

    assert 'mkdir -p "$PWD/.render/bin"' in build_script
    assert 'python scripts/package_codex_runtime.py' in build_script
    assert 'install -m 0755 "$(readlink -f "$(command -v claude)")" "$PWD/.render/bin/claude"' in build_script
    assert 'export PATH="$PWD/.render/bin:$HOME/.local/bin:$PATH"' in start_script


def test_packaged_codex_runs_without_original_installation(tmp_path):
    import subprocess
    from scripts.package_codex_runtime import package_runtime

    release = tmp_path / "release"
    (release / "bin").mkdir(parents=True)
    (release / "codex-resources").mkdir()
    (release / "codex-resources/resource").write_text("runtime-ready")
    binary = release / "bin/codex"
    binary.write_text('#!/bin/sh\nexec "$(dirname "$0")/codex-code-mode-host"\n')
    host = release / "bin/codex-code-mode-host"
    host.write_text('#!/bin/sh\ncat "$(dirname "$0")/../codex-resources/resource"\n')
    for executable in (binary, host):
        executable.chmod(0o755)
    link = tmp_path / "codex"
    link.symlink_to(binary)
    destination = tmp_path / "artifact"
    packaged = package_runtime(link, destination)
    release.rename(tmp_path / "removed-original")
    result = subprocess.run([str(packaged)], capture_output=True, text=True, check=True)
    assert result.stdout == "runtime-ready"


def test_packaging_replaces_previous_artifact_with_read_only_files(tmp_path):
    from scripts.package_codex_runtime import package_runtime

    release = tmp_path / "release"
    (release / "bin").mkdir(parents=True)
    for name in ("codex", "codex-code-mode-host"):
        (release / "bin" / name).write_text("#!/bin/sh\n")
    library = release / "codex-resources/voice/lib/libz.so.1"
    library.parent.mkdir(parents=True)
    library.write_text("v1")
    library.chmod(0o555)
    destination = tmp_path / "artifact"
    package_runtime(release / "bin/codex", destination)
    (destination / "stale-from-old-release").write_text("stale")

    library.chmod(0o755)
    library.write_text("v2")
    library.chmod(0o555)
    package_runtime(release / "bin/codex", destination)

    assert (destination / "codex-resources/voice/lib/libz.so.1").read_text() == "v2"
    assert not (destination / "stale-from-old-release").exists()


def test_packaging_rejects_incomplete_codex_installation(tmp_path):
    import pytest
    from scripts.package_codex_runtime import package_runtime

    binary = tmp_path / "release/bin/codex"
    binary.parent.mkdir(parents=True)
    binary.write_text("incomplete")
    with pytest.raises(ValueError, match="codex-code-mode-host"):
        package_runtime(binary, tmp_path / "artifact")
