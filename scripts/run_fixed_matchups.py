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
from engines.base_engine import BaseEngine
from game.match_scheduler import GameTask, MatchScheduler
from game.pgn_logger import PGNLogger
from game.stats_collector import StatsCollector
from llm.base_llm import BaseLLMPlayer
from rating.glicko2 import Glicko2System
from rating.rating_store import RatingStore
from utils import resolve_player_id


@dataclass(frozen=True)
class PlannedGame:
    """One directed game required by a fixed matchup plan."""

    white_id: str
    black_id: str


@dataclass(frozen=True)
class MatchupPlan:
    """One unordered matchup and its required games in each color direction."""

    player_a: str
    player_b: str
    games_per_color: int


Player = BaseEngine | BaseLLMPlayer


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping with a clear error for malformed files."""
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def normalize_matchups(
    raw_matchups: Any,
    default_games_per_color: int = 1,
) -> list[MatchupPlan]:
    """Validate matchup entries and resolve per-matchup game targets."""
    if not isinstance(raw_matchups, list):
        raise ValueError("matchups must be a list of two-player lists")
    if default_games_per_color < 1:
        raise ValueError("games_per_color must be at least 1")

    matchups: list[MatchupPlan] = []
    seen: set[frozenset[str]] = set()
    for index, raw in enumerate(raw_matchups):
        games_per_color = default_games_per_color
        players = raw
        if isinstance(raw, dict):
            players = raw.get("players")
            games_per_color = int(raw.get("games_per_color", default_games_per_color))
        if (
            not isinstance(players, list)
            or len(players) != 2
            or not all(isinstance(value, str) for value in players)
        ):
            raise ValueError(
                f"matchups[{index}] must be a two-player list or a mapping with players"
            )
        if games_per_color < 1:
            raise ValueError(f"matchups[{index}].games_per_color must be at least 1")
        player_a, player_b = players
        if player_a == player_b:
            raise ValueError(f"matchups[{index}] pairs {player_a} with itself")
        key = frozenset((player_a, player_b))
        if key in seen:
            raise ValueError(f"Duplicate matchup: {player_a} vs {player_b}")
        seen.add(key)
        matchups.append(MatchupPlan(player_a, player_b, games_per_color))
    return matchups


def build_missing_games(
    matchups: Iterable[MatchupPlan | tuple[str, str]],
    existing_results: Iterable[Any],
    games_per_color: int | None = None,
) -> list[PlannedGame]:
    """Return only directed games not already represented by saved results."""
    if games_per_color is not None and games_per_color < 1:
        raise ValueError("games_per_color must be at least 1")

    existing = Counter((result.white_id, result.black_id) for result in existing_results)
    planned: list[PlannedGame] = []
    for matchup in matchups:
        if isinstance(matchup, MatchupPlan):
            player_a = matchup.player_a
            player_b = matchup.player_b
            target = matchup.games_per_color
        else:
            if games_per_color is None:
                raise ValueError("games_per_color is required for tuple matchups")
            player_a, player_b = matchup
            target = games_per_color
        for white_id, black_id in ((player_a, player_b), (player_b, player_a)):
            missing = max(0, target - existing[(white_id, black_id)])
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


def index_player_configs(
    llm_configs: Iterable[dict[str, Any]],
    engine_configs: Iterable[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Index selected configs by their effective benchmark player IDs."""
    llms = {
        resolve_player_id(config["player_id"], config.get("reasoning_effort")): config
        for config in llm_configs
    }
    engines = {config["player_id"]: config for config in engine_configs}
    return llms, engines


def create_game_players(
    game: PlannedGame,
    llm_configs: dict[str, dict[str, Any]],
    engine_configs: dict[str, dict[str, Any]],
) -> tuple[Player, Player]:
    """Create isolated player instances for one game.

    A model may participate in several games concurrently, but mutable player
    state (tokens, timing, retry context, and engine processes) must never be
    shared between those games.
    """
    player_ids = {game.white_id, game.black_id}
    game_llm_configs = [
        llm_configs[player_id] for player_id in player_ids if player_id in llm_configs
    ]
    game_engine_configs = [
        engine_configs[player_id] for player_id in player_ids if player_id in engine_configs
    ]
    llm_players, _ = create_llm_players({"llms": game_llm_configs}, api_backend="codex")
    engine_players, _, _ = create_engines({"engines": game_engine_configs})
    players = {**engine_players, **llm_players}
    return players[game.white_id], players[game.black_id]


async def close_game_players(*players: Player) -> None:
    """Close isolated game players, including both instances on cancellation."""
    for player in players:
        if isinstance(player, BaseLLMPlayer):
            await player.close()
        else:
            player.close()


async def run_plan(plan_path: Path, *, dry_run: bool = False) -> int:
    """Run all missing games from one fixed matchup plan."""
    plan = load_yaml(plan_path)
    default_games_per_color = int(plan.get("games_per_color", 1))
    matchups = normalize_matchups(plan.get("matchups"), default_games_per_color)

    source_path = Path(plan.get("source_config", "benchmark.yaml"))
    if not source_path.is_absolute():
        source_path = plan_path.parent / source_path
    source = load_yaml(source_path)

    player_ids = {
        player_id
        for matchup in matchups
        for player_id in (matchup.player_a, matchup.player_b)
    }
    llm_configs, engine_configs = selected_player_configs(source, player_ids)
    codex_max_concurrent = int(
        plan.get("codex_max_concurrent", plan.get("max_concurrent", 2))
    )
    if codex_max_concurrent < 1:
        raise ValueError("codex_max_concurrent must be at least 1")
    for config in llm_configs:
        config["codex_max_concurrent"] = codex_max_concurrent
    llm_configs_by_id, engine_configs_by_id = index_player_configs(llm_configs, engine_configs)

    pgn_logger = PGNLogger()
    existing_results = pgn_logger.load_all_results()
    games = build_missing_games(matchups, existing_results)

    target_games = 2 * sum(matchup.games_per_color for matchup in matchups)
    targets = sorted({matchup.games_per_color for matchup in matchups})
    target_description = ", ".join(str(target) for target in targets)
    print(
        f"Fixed matchup plan: {len(matchups)} pairings, "
        f"{target_description} game(s) per color"
    )
    print(f"Saved directed games already satisfying this plan: {target_games - len(games)}")
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
    pending_games: asyncio.Queue[tuple[int, PlannedGame]] = asyncio.Queue()
    for scheduled_game in enumerate(games, start=1):
        pending_games.put_nowait(scheduled_game)

    async def worker() -> None:
        nonlocal completed, failed
        while True:
            try:
                game_num, game = pending_games.get_nowait()
            except asyncio.QueueEmpty:
                return
            game_players: tuple[Player, Player] | None = None
            try:
                game_players = create_game_players(game, llm_configs_by_id, engine_configs_by_id)
                result = await scheduler.run_single_game(
                    GameTask(
                        white=game_players[0],
                        black=game_players[1],
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
                if game_players is not None:
                    await close_game_players(*game_players)
                pending_games.task_done()

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
