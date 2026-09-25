import argparse
import asyncio
from unittest import mock

import cli


class FakeLLM:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class FakeEngine:
    rating = 1500

    def close(self):
        pass


def run_with_players(players):
    llms = {"a (high)": FakeLLM(), "b (high)": FakeLLM()}
    scheduler = mock.MagicMock()
    scheduler.return_value.run_benchmark = mock.AsyncMock()
    args = argparse.Namespace(
        config="config/benchmark.yaml", api="codex", api_key=None,
        max_cost=1.0, verbose=False, acquisition_plan=False, players=players,
    )
    with mock.patch.object(cli, "load_config", return_value={}), \
         mock.patch.object(cli, "create_engines", return_value=({"eng": FakeEngine()}, {"eng"}, set())), \
         mock.patch.object(cli, "create_llm_players", return_value=(dict(llms), set())), \
         mock.patch.object(cli, "RatingStore"), \
         mock.patch.object(cli, "PGNLogger"), \
         mock.patch.object(cli, "Leaderboard"), \
         mock.patch.object(cli, "MatchScheduler", scheduler), \
         mock.patch.object(cli, "invalidate_remote_cache"):
        code = asyncio.run(cli.run_benchmark(args))
    return code, llms, scheduler


def test_players_filter_limits_scheduled_llms():
    code, llms, scheduler = run_with_players(["a (high)"])

    assert code == 0
    assert scheduler.return_value.run_benchmark.await_args.kwargs["llm_ids"] == ["a (high)"]
    assert set(scheduler.call_args.kwargs["players"]) == {"eng", "a (high)"}
    assert llms["b (high)"].closed


def test_players_filter_rejects_unknown_player():
    code, _, scheduler = run_with_players(["missing (high)"])

    assert code == 1
    scheduler.assert_not_called()
