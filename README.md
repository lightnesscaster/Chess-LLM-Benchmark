# Chess LLM Benchmark

Benchmark suite for evaluating LLM chess-playing ability using Glicko-2 ratings calibrated against engine anchors.

## How It Works

1. LLMs receive the current position (FEN + ASCII board) and must return a single UCI move
2. Illegal moves get one retry with a warning; second illegal move = forfeit
3. Games are played against Stockfish at various skill levels as rating anchors
4. Glicko-2 ratings are calculated based on game outcomes

## Installation

```bash
pip install -r requirements.txt
```

Requires:
- Stockfish in PATH (or specify path in config) for Stockfish anchors
- lc0 for Maia anchors (optional)

## Usage

### Set API Key

```bash
export OPENROUTER_API_KEY="your-key"
```

### Run a Test Game

```bash
# LLM vs Stockfish (default engine)
python cli.py test --white-model meta-llama/llama-4-maverick --black-engine --stockfish-skill 5

# LLM vs LLM
python cli.py test --white-model meta-llama/llama-4-maverick --black-model deepseek/deepseek-chat-v3-0324

# Multiple games (alternates colors each game)
python cli.py test --white-model meta-llama/llama-4-maverick --black-engine --games 10

# Against different engine types
python cli.py test --white-model meta-llama/llama-4-maverick --black-engine --engine-type maia-1100
python cli.py test --white-model meta-llama/llama-4-maverick --black-engine --engine-type random
python cli.py test --white-model meta-llama/llama-4-maverick --black-engine --engine-type eubos

# With reasoning models (use max-tokens 0 for extended thinking)
python cli.py test --white-model deepseek/deepseek-r1 --black-engine --white-reasoning-effort high --max-tokens 0

# Enable reasoning mode for hybrid models
python cli.py test --white-model deepseek/deepseek-chat --black-engine --reasoning

# Don't save the game
python cli.py test --white-model meta-llama/llama-4-maverick --black-engine --no-save
```

**Test command engine presets:** `stockfish`, `maia-1100`, `maia-1900`, `random`, `eubos`

> Note: `eubos` is a hardcoded preset. For custom UCI engines in benchmarks, use `type: uci` in config.

### Run Full Benchmark

```bash
python cli.py run -c config/benchmark.yaml -v
```

### View Leaderboard

```bash
python cli.py leaderboard --min-games 5
```

### Recalculate Ratings

Recalculate all ratings from stored game results (useful after changing anchor ratings):

```bash
python cli.py recalculate -c config/benchmark.yaml
```

### Web Interface

```bash
python web/app.py
# Open http://localhost:5000
```

Features:
- **Leaderboard** with Glicko-2 ratings, FIDE estimates, confidence intervals, and legal move rates
- **Game library** with filtering by player
- **Interactive game viewer** with move-by-move navigation
- **Client-side Stockfish analysis** (toggle-able eval bar + top engine lines)
- **Methodology page** explaining the rating system
- **JSON API** at `/api/leaderboard`, `/api/games`, `/api/game/<id>`

## Configuration

Edit `config/benchmark.yaml` to configure:

- **LLM models** to benchmark (via OpenRouter)
- **Engine anchors** (Stockfish, Maia, Random, or any UCI engine)
- **Games per matchup** and concurrency settings

Example:
```yaml
benchmark:
  games_vs_anchor_per_color: 10
  games_vs_llm_per_color: 5
  max_concurrent: 4
  max_moves: 200

engines:
  - player_id: "random-bot"
    type: random
    rating: 400

  - player_id: "maia-1100"
    type: maia
    lc0_path: "/opt/homebrew/bin/lc0"
    weights_path: "maia-1100.pb.gz"
    rating: 1628

  - player_id: "sf-skill-5"
    type: stockfish
    rating: 1300
    skill_level: 5

  - player_id: "eubos"
    type: uci                    # Generic UCI engine
    path: "/path/to/engine"
    rating: 2200
    initial_time: 900            # Clock-based time control (seconds)
    increment: 10

llms:
  - player_id: "llama-4-maverick"
    model_name: "meta-llama/llama-4-maverick"
    temperature: 0.0
    max_tokens: 10

  - player_id: "deepseek-r1"
    model_name: "deepseek/deepseek-r1"
    reasoning_effort: "medium"  # low, medium, high, xhigh
```

**Engine types:** `stockfish`, `maia`, `random`, `uci` (generic UCI engine)

## Project Structure

```
├── cli.py                 # Main CLI entrypoint
├── config/
│   └── benchmark.yaml     # Benchmark configuration
├── engines/               # Chess engine wrappers
│   ├── base_engine.py     # Base engine class
│   ├── stockfish_engine.py
│   ├── maia_engine.py
│   ├── random_engine.py
│   └── uci_engine.py      # Generic UCI engine wrapper
├── llm/                   # LLM player clients
│   ├── base_llm.py        # Base LLM player class
│   ├── openrouter_client.py
│   └── prompts.py         # Chess prompt templates
├── game/                  # Game execution
│   ├── game_runner.py     # Core game loop
│   ├── match_scheduler.py # Parallel game execution
│   ├── models.py          # Pydantic data models
│   ├── pgn_logger.py      # PGN/result saving
│   └── stats_collector.py # Win/loss/draw stats
├── rating/                # Rating system
│   ├── glicko2.py         # Glicko-2 implementation
│   ├── rating_store.py    # Local JSON storage
│   ├── leaderboard.py     # Leaderboard formatting
│   └── fide_estimate.py   # FIDE rating estimation
├── web/                   # Web interface
│   ├── app.py             # Flask application
│   ├── templates/         # HTML templates
│   └── static/            # CSS/JS assets
└── data/                  # Output (gitignored)
    ├── games/             # PGN files
    ├── results/           # JSON game results
    └── ratings.json       # Current ratings
```

## Rating System

Uses Glicko-2 with:
- **Rating (μ)**: Estimated skill level (starts at 1500)
- **Rating Deviation (RD)**: Uncertainty (decreases with more games)
- **Volatility (σ)**: Expected rating fluctuation
- **FIDE Estimate**: Approximate FIDE rating based on ChessGoals.com Lichess-to-FIDE conversion (valid for 1715-2500 range)
- **Legal Move Rate**: Percentage of moves that were legal on first attempt

Engine anchors have fixed ratings based on their approximate Elo and are never updated.


## Illegal Move Policy

- First illegal move: Warning sent, LLM gets one retry
- Second illegal move: Immediate forfeit (loss)

The retry prompt tells the LLM which move was illegal but does not provide a list of legal moves.
