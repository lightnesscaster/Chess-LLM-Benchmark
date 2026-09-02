#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt

curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_NON_INTERACTIVE=1 bash
curl -fsSL https://claude.ai/install.sh | bash -s latest

export PATH="$HOME/.local/bin:$PATH"
codex --version
claude --version

mkdir -p "$PWD/.render/bin"
install -m 0755 "$(readlink -f "$(command -v codex)")" "$PWD/.render/bin/codex"
install -m 0755 "$(readlink -f "$(command -v claude)")" "$PWD/.render/bin/claude"
