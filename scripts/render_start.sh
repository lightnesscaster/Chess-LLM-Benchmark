#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export CODEX_HOME="${CODEX_HOME:-/var/data/codex}"

python -m web.cli_runtime
python -m web.claude_catalog

exec gunicorn web.app:app \
    --bind "0.0.0.0:$PORT" \
    --workers 2 \
    --threads 2 \
    --timeout 620
