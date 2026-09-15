#!/usr/bin/env bash
set -euo pipefail

export PATH="$PWD/.render/bin:$HOME/.local/bin:$PATH"
export CODEX_HOME="${CODEX_HOME:-/var/data/codex}"

python -m web.cli_runtime
python -m web.claude_catalog

exec gunicorn web.app:app \
    --bind "0.0.0.0:$PORT" \
    --workers "${WEB_CONCURRENCY:-1}" \
    --threads "${WEB_THREADS:-4}" \
    --timeout 620
