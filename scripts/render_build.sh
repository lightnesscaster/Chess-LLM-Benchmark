#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt

curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_NON_INTERACTIVE=1 bash
curl -fsSL https://claude.ai/install.sh | bash -s stable

export PATH="$HOME/.local/bin:$PATH"
codex --version
claude --version
