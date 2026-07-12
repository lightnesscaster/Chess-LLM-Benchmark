#!/usr/bin/env python3
"""Run an explicit, resumable set of rated benchmark matchups."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli import create_engines, create_llm_players
from game.match_scheduler import GameTask, MatchScheduler
from game.pgn_logger import PGNLogger
from game.stats_collector import StatsCollector
from rating.glicko2 import Glicko2System
from rating.rating_store import RatingStore
from utils import resolve_player_id


@dataclass(frozen=True)
class PlannedGame:
    """One directed game required by a fixed matchup plan."""

    white_id: str
    black_id: str


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping with a clear error for malformed files."""
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def normalize_matchups(raw_matchups: Any) -> list[tuple[str, str]]:
    """Validate and deduplicate unordered two-player matchups."""
    if not isinstance(raw_matchups, list):
        raise ValueError("matchups must be a list of two-player lists")

    matchups: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for index, raw in enumerate(raw_matchups):
        if not isinstance(raw, list) or len(raw) != 2 or not all(isinstance(v, str) for v in raw):
            raise ValueError(f"matchups[{index}] must contain exactly two player IDs")
        white_id, black_id = raw
        if white_id == black_id:
            raise ValueError(f"matchups[{index}] pairs {white_id} with itself")
        key = frozenset((white_id, black_id))
        if key in seen:
            raise ValueError(f"Duplicate matchup: {white_id} vs {black_id}")
        seen.add(key)
        matchups.append((white_id, black_id))
    return matchups


def build_missing_games(
    matchups: Iterable[tuple[str, str]],
    existing_results: Iterable[Any],
    games_per_color: int,
) -> list[PlannedGame]:
    """Return only directed games not already represented by saved results."""
    if games_per_color < 1:
        raise ValueError("games_per_color must be at least 1")

    existing = Counter((result.white_id, result.black_id) for result in existing_results)
    planned: list[PlannedGame] = []
    for player_a, player_b in matchups:
        for white_id, black_id in ((player_a, player_b), (player_b, player_a)):
            missing = max(0, games_per_color - existing[(white_id, black_id)])
            planned.extend(PlannedGame(white_id, black_id) for _ in range(missing))
    return planned


def selected_player_configs(
    source: dict[str, Any],
    player_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select exact LLM/engine configs and require Codex for every LLM."""
    llm_configs: dict[str, dict[str, Any]] = {}
    for config in source.get("llms", []):
        player_id = resolve_player_id(config["player_id"], config.get("reasoning_effort"))
        llm_configs[player_id] = config

    engine_configs = {config["player_id"]: config for config in source.get("engines", [])}
    configured_ids = set(llm_configs) | set(engine_configs)

    missing = sorted(player_ids - configured_ids)
    if missing:
        raise ValueError("Players missing from source config: " + ", ".join(missing))

    selected_llm_ids = player_ids & llm_configs.keys()
    selected_engine_ids = player_ids & engine_configs.keys()
    not_codex = sorted(
        player_id for player_id in selected_llm_ids if llm_configs[player_id].get("api") != "codex"
    )
    if not_codex:
        raise ValueError("Fixed matchup players must use api: codex: " + ", ".join(not_codex))
    return (
        [llm_configs[player_id] for player_id in sorted(selected_llm_ids)],
        [engine_configs[player_id] for player_id in sorted(selected_engine_ids)],
    )


async def run_plan(plan_path: Path, *, dry_run: bool = False) -> int:
    """Run all missing games from one fixed matchup plan."""
    plan = load_yaml(plan_path)
    matchups = normalize_matchups(plan.get("matchups"))
    games_per_color = int(plan.get("games_per_color", 1))

    source_path = Path(plan.get("source_config", "benchmark.yaml"))
    if not source_path.is_absolute():
        source_path = plan_path.parent / source_path
    source = load_yaml(source_path)

    player_ids = {player_id for matchup in matchups for player_id in matchup}
    llm_configs, engine_configs = selected_player_configs(source, player_ids)

    pgn_logger = PGNLogger()
    existing_results = pgn_logger.load_all_results()
    games = build_missing_games(matchups, existing_results, games_per_color)

    print(f"Fixed matchup plan: {len(matchups)} pairings, {games_per_color} game(s) per color")
    print(f"Saved directed games already satisfying this plan: {2 * len(matchups) * games_per_color - len(games)}")
    print(f"Missing games to run: {len(games)}")
    for game in games:
        print(f"  {game.white_id} vs {game.black_id}")
    if dry_run or not games:
        return 0

    llm_players, reasoning_ids = create_llm_players({"llms": llm_configs}, api_backend="codex")
    engine_players, anchor_ids, ghost_ids = create_engines({"engines": engine_configs})
    players = {**engine_players, **llm_players}
    rating_store = RatingStore(
        path="data/ratings.json",
        anchor_ids=anchor_ids,
        ghost_ids=ghost_ids,
    )
    for engine_id in anchor_ids:
        rating_store.set_anchor(engine_id, engine_players[engine_id].rating)
    stats_collector = StatsCollector()
    stats_collector.add_results(existing_results)
    scheduler = MatchScheduler(
        players=players,
        rating_store=rating_store,
        glicko=Glicko2System(),
        pgn_logger=pgn_logger,
        stats_collector=stats_collector,
        max_concurrent=int(plan.get("max_concurrent", 2)),
        max_moves=int(plan.get("max_moves", 200)),
        verbose=True,
        reasoning_ids=reasoning_ids,
        llm_configs={
            resolve_player_id(config["player_id"], config.get("reasoning_effort")): config
            for config in llm_configs
        },
    )

    completed = 0
    failed = 0
    lock = asyncio.Lock()
    pending_games = list(enumerate(games, start=1))
    busy_players: set[str] = set()
    availability = asyncio.Condition()

    async def claim_disjoint_game() -> tuple[int, PlannedGame] | None:
        """Claim the first pending game whose two players are both idle."""
        async with availability:
            while pending_games:
                for index, (game_num, game) in enumerate(pending_games):
                    if game.white_id in busy_players or game.black_id in busy_players:
                        continue
                    pending_games.pop(index)
                    busy_players.update((game.white_id, game.black_id))
                    return game_num, game
                await availability.wait()
            return None

    async def release_players(game: PlannedGame) -> None:
        async with availability:
            busy_players.discard(game.white_id)
            busy_players.discard(game.black_id)
            availability.notify_all()

    async def worker() -> None:
        nonlocal completed, failed
        while True:
            claimed = await claim_disjoint_game()
            if claimed is None:
                return
            game_num, game = claimed
            try:
                result = await scheduler.run_single_game(
                    GameTask(
                        white=players[game.white_id],
                        black=players[game.black_id],
                        game_num=game_num,
                        total_games=len(games),
                    )
                )
                async with lock:
                    if result is None:
                        failed += 1
                    else:
                        completed += 1
            except Exception as exc:
                print(f"Game failed: {game.white_id} vs {game.black_id}: {exc}")
                async with lock:
                    failed += 1
            finally:
                await release_players(game)

    try:
        workers = [asyncio.create_task(worker()) for _ in range(scheduler.max_concurrent)]
        await asyncio.gather(*workers)
        rating_store.save()
    finally:
        for player in llm_players.values():
            await player.close()
        for engine in engine_players.values():
            engine.close()

    print(f"Fixed matchup run complete: {completed} saved, {failed} failed")
    if failed:
        print("Re-run the same command to schedule only the still-missing directed games.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Fixed matchup plan YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print missing games without running them")
    args = parser.parse_args()
    return asyncio.run(run_plan(args.config, dry_run=args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
