# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run a test game (LLM vs Stockfish)
python cli.py test --white-model meta-llama/llama-4-maverick --black-engine --stockfish-skill 5

# Run a test game (LLM vs LLM)
python cli.py test --white-model meta-llama/llama-4-maverick --black-model deepseek/deepseek-chat-v3-0324

# Run multiple games with color alternation
python cli.py test --white-model meta-llama/llama-4-maverick --black-engine --games 10

# Run with reasoning model
python cli.py test --white-model deepseek/deepseek-r1 --black-engine --white-reasoning-effort high

# Run full benchmark
python cli.py run -c config/benchmark.yaml -v

# View leaderboard
python cli.py leaderboard --min-games 5

# Recalculate ratings from stored results
python cli.py recalculate -c config/benchmark.yaml

# Run web interface
python web/app.py
```

Requires `OPENROUTER_API_KEY` environment variable. Stockfish/lc0 required for engine anchors.

## Architecture

This is an async Python benchmark that evaluates LLM chess-playing ability using Glicko-2 ratings calibrated against engine anchors.

### Core Flow

1. **CLI (`cli.py`)** - Entry point with four commands: `run`, `test`, `leaderboard`, `recalculate`
2. **MatchScheduler (`game/match_scheduler.py`)** - Orchestrates parallel game execution with semaphore-based concurrency control
3. **GameRunner (`game/game_runner.py`)** - Runs individual games, enforces illegal move policy (2 strikes = forfeit)
4. **OpenRouterPlayer (`llm/openrouter_client.py`)** - Async LLM client that parses UCI/SAN moves from responses
5. **Glicko2System (`rating/glicko2.py`)** - Updates ratings after each game; anchors have fixed ratings
6. **Web App (`web/app.py`)** - Flask app for viewing leaderboard and game library with PGN viewer

### Player Abstraction

Two base classes with different interfaces:
- `BaseEngine` (sync): `select_move(board) -> chess.Move`
- `BaseLLMPlayer` (async): `select_move(board, is_retry, last_move_illegal) -> str` (UCI)

The `GameRunner` handles both types, calling engines synchronously and LLMs with `await`.

### Prompt Strategy

LLMs receive FEN + ASCII board and must return a single UCI move. On illegal move, they get one retry with the retry prompt (`llm/prompts.py`) that tells them which move was illegal but does not provide legal moves list.

### Rating System

- LLM ratings are updated incrementally after each game via `RatingStore`
- Engine anchors (Stockfish, Maia, Random) have fixed ratings and are never updated
- Multi-pass convergence algorithm in `recalculate` command for stable ratings
- Ratings stored in `data/ratings.json`

### Data Output

- PGN files: `data/games/`
- Game results JSON: `data/results/`
- Ratings: `data/ratings.json`

## Configuration

Edit `config/benchmark.yaml` to configure:
- `benchmark.games_vs_anchor_per_color`: Games per LLM vs each anchor
- `benchmark.games_vs_llm_per_color`: Games per LLM pair
- `benchmark.max_concurrent`: Parallel game limit
- `engines`: Anchors with fixed ratings (types: `stockfish`, `maia`, `random`)
- `llms`: Models to benchmark with optional `reasoning` and `reasoning_effort` settings
