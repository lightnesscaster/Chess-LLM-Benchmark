import asyncio
from types import SimpleNamespace

import pytest

import cli
from game import publication
from tests.test_publication import payload


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("flags,published,saved,termination", [([], 1, 1, "resignation"), (["--local-only"], 0, 1, "resignation"), (["--no-save"], 0, 0, "resignation"), ([], 0, 0, "api_error"), ([], 0, 0, "error")])
def test_manual_completion_publishes_by_default_but_respects_opt_outs(monkeypatch, tmp_path, flags, published, saved, termination, mixed):
    captured = []
    original = cli.run_manual_game

    async def capture(args):
        captured.append(args)

    monkeypatch.setattr(cli, "run_manual_game", capture)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    models = (["--api", "openrouter", "--white-model", "deepseek/deepseek-v4.1-flash", "--white-reasoning-effort", "high", "--black-model", "openai/gpt-5.6-sol", "--black-reasoning-effort", "medium"] if mixed else ["--api", "codex", "--white-model", "white", "--black-model", "black"])
    monkeypatch.setattr("sys.argv", ["cli.py", "manual", *models, *flags])
    cli.main()
    calls = []
    writes = []
    constructed = []
    r, pgn = payload()
    r.termination = termination

    class Player:
        def __init__(self, player_id, **kwargs):
            self.player_id = player_id
            constructed.append((type(self).__name__, kwargs))

        async def close(self):
            pass

    class OpenRouterFake(Player):
        pass

    class Runner:
        def __init__(self, **kwargs):
            pass

        async def play_game(self):
            return r, pgn

    class Logger:
        def __init__(self, **kwargs):
            self.results_dir = tmp_path

        def load_all_results(self):
            return []

        def save_game(self, result, text):
            writes.append(result.game_id)
            return result

    monkeypatch.setattr(cli, "CodexSubagentPlayer", Player)
    monkeypatch.setattr(cli, "OpenRouterPlayer", OpenRouterFake)
    monkeypatch.setattr(cli, "GameRunner", Runner)
    monkeypatch.setattr(cli, "PGNLogger", Logger)
    monkeypatch.setattr(cli, "RatingStore", lambda **kwargs: SimpleNamespace(get=lambda pid: SimpleNamespace(rating=1500, rating_deviation=100, games_played=3, games_rd=100)))
    monkeypatch.setattr("rating.prompt_context.rating_snapshot", lambda *args: {})
    monkeypatch.setattr("position_benchmark.stability_cap_shadow.ensure_shadow_lock_before_saved_game", lambda *a, **k: SimpleNamespace(allowed=True, record=None))
    monkeypatch.setattr(cli, "invalidate_remote_cache", lambda: None)
    monkeypatch.setattr(publication, "publish_saved_game", lambda path: calls.append(path) or {"status": "published"})
    monkeypatch.setattr(publication, "load_production_ratings", lambda store: None)
    asyncio.run(original(captured[0]))
    assert len(calls) == published
    assert len(writes) == saved
    if mixed:
        assert [kind for kind, _ in constructed] == ["OpenRouterFake", "Player"]
        assert constructed[0][1]["provider_order"] == ["deepseek", "fireworks", "gmicloud", "siliconflow"]
        assert constructed[0][1]["provider_ignore"] == ["deepinfra"]
        assert constructed[0][1]["timeout"] == 1200
        assert constructed[1][1]["reasoning_effort"] == "medium"
