#!/bin/bash
# UCI wrapper script for survival-bot
# Use this with lichess-bot or any UCI-compatible GUI

cd "$(dirname "$0")"
exec python -m engines.survival_uci "$@"
