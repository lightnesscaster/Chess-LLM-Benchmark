import copy
import asyncio
import importlib
import io
import json

import chess.pgn
import pytest
import yaml

from llm import TransientAPIError


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "web_play_models": [
                    {
                        "player_id": "claude-fable-5.1",
                        "model_name": "claude-fable-5-1",
                        "web_api": "claude_code",
                        "web_model_name": "claude-fable-5-1",
                    },
                    {
                        "player_id": "claude-fable-5",
                        "model_name": "claude-fable-5",
                        "web_api": "claude_code",
                        "web_model_name": "claude-fable-5",
                    },
                ],
                "llms": [
                    {
                        "player_id": "chat-model",
                        "model_name": "provider/chat-model",
                        "temperature": 0.0,
                    },
                    {
                        "player_id": "completion-model",
                        "model_name": "provider/completion-model",
                        "api": "completion",
                    },
                    {
                        "player_id": "gemini-direct",
                        "model_name": "google/gemini-direct",
                        "api": "gemini",
                    },
                    {
                        "player_id": "codex-local",
                        "model_name": "openai/codex-local",
                        "api": "codex",
                    },
                    {
                        "player_id": "claude-code-local",
                        "model_name": "anthropic/claude-opus-4.7",
                        "web_api": "claude_code",
                        "web_model_name": "opus",
                        "reasoning_effort": "high",
                    },
                    {
                        "player_id": "unknown-backend",
                        "model_name": "provider/unknown",
                        "api": "unsupported",
                    },
                    {
                        "player_id": "benchmark-only-model",
                        "model_name": "provider/benchmark-only",
                        "web_hidden": True,
                    },
                    {
                        "player_id": "offline-model",
                        "model_name": "provider/offline",
                        "unavailable": True,
                    },
                ]
            }
        )
    )
    return path


def _service():
    return importlib.import_module("web.play_service")


def test_model_list_includes_only_configured_web_backends_with_keys(config_path):
    models = _service().list_playable_models(
        config_path,
        {"OPENROUTER_API_KEY": "openrouter-key"},
    )

    assert models == [
        {
            "id": "chat-model",
            "name": "chat-model",
            "model_name": "provider/chat-model",
            "efforts": [{"id": "default", "name": "Auto"}],
            "default_effort": "default",
        },
        {
            "id": "completion-model",
            "name": "completion-model",
            "model_name": "provider/completion-model",
            "efforts": [{"id": "default", "name": "Auto"}],
            "default_effort": "default",
        },
    ]


def test_model_list_adds_direct_gemini_when_its_key_is_configured(config_path):
    models = _service().list_playable_models(
        config_path,
        {
            "OPENROUTER_API_KEY": "openrouter-key",
            "GEMINI_API_KEY": "gemini-key",
        },
    )

    assert [model["id"] for model in models] == [
        "chat-model",
        "completion-model",
        "gemini-direct",
    ]


def test_model_list_groups_effort_variants_under_one_model(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump({
        "llms": [
            {
                "player_id": "reasoner (low)",
                "model_name": "provider/reasoner",
                "reasoning_effort": "low",
            },
            {
                "player_id": "reasoner (high)",
                "model_name": "provider/reasoner",
                "reasoning_effort": "high",
            },
        ],
    }))

    models = _service().list_playable_models(
        config_path,
        {"OPENROUTER_API_KEY": "openrouter-key"},
    )

    assert models == [{
        "id": "reasoner",
        "name": "reasoner",
        "model_name": "provider/reasoner",
        "efforts": [
            {"id": "low", "name": "Low"},
            {"id": "high", "name": "High"},
        ],
        "default_effort": "low",
    }]


def test_model_list_preserves_named_no_thinking_variant(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump({
        "llms": [
            {
                "player_id": "reasoner (no thinking)",
                "model_name": "provider/reasoner",
            },
            {
                "player_id": "reasoner (thinking)",
                "model_name": "provider/reasoner",
                "reasoning": True,
            },
        ],
    }))

    models = _service().list_playable_models(
        config_path,
        {"OPENROUTER_API_KEY": "openrouter-key"},
    )

    assert models[0]["efforts"] == [
        {"id": "none", "name": "None"},
        {"id": "default", "name": "Auto"},
    ]


def test_explicit_web_efforts_select_requested_claude_effort(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump({
        "web_play_models": [{
            "player_id": "claude-flex",
            "model_name": "anthropic/claude-flex",
            "web_api": "claude_code",
            "web_model_name": "claude-flex",
            "web_reasoning_efforts": ["low", "high", "xhigh"],
            "web_default_reasoning_effort": "high",
        }],
    }))
    catalog_path = tmp_path / "claude-models.json"
    catalog_path.write_text(json.dumps({"models": ["claude-flex"]}))
    received = {}

    state = _service().start_game(
        "claude-flex",
        "black",
        config_path,
        {
            "CLAUDE_CODE_OAUTH_TOKEN": "subscription-token",
            "CLAUDE_MODEL_CATALOG_PATH": str(catalog_path),
        },
        move_provider=lambda model, *_args: (
            received.update(model) or "e2e4"
        ),
        reasoning_effort="xhigh",
    )

    assert received["web_reasoning_effort"] == "xhigh"
    assert state["model_id"] == "claude-flex"
    assert state["reasoning_effort"] == "xhigh"
    rated_state = _service().start_game(
        "claude-flex",
        "white",
        config_path,
        {
            "CLAUDE_CODE_OAUTH_TOKEN": "subscription-token",
            "CLAUDE_MODEL_CATALOG_PATH": str(catalog_path),
        },
        reasoning_effort="xhigh",
        human_profile={
            "username": "Some_Player",
            "rating": 1847,
            "rating_deviation": 73,
            "provisional": False,
        },
    )
    assert rated_state["rated_model_id"] == "claude-flex (xhigh)"


def test_configured_claude_models_offer_separate_effort_choices(tmp_path):
    catalog_path = tmp_path / "claude-models.json"
    catalog_path.write_text(json.dumps({
        "models": ["claude-fable-5-1", "claude-fable-5", "opus", "sonnet", "haiku"],
    }))

    models = _service().list_playable_models(
        __import__("pathlib").Path("config/benchmark.yaml"),
        {
            "CLAUDE_CODE_OAUTH_TOKEN": "subscription-token",
            "CLAUDE_MODEL_CATALOG_PATH": str(catalog_path),
        },
    )

    assert len(models) == 5
    for model in models:
        assert [effort["id"] for effort in model["efforts"]] == [
            "low", "medium", "high", "xhigh", "max",
        ]
        assert model["default_effort"] == "high"


def test_unavailable_effort_is_rejected_for_model(config_path):
    with pytest.raises(_service().ConfigurationError, match="effort"):
        _service().start_game(
            "chat-model",
            "white",
            config_path,
            {"OPENROUTER_API_KEY": "openrouter-key"},
            reasoning_effort="high",
        )


def test_model_list_hides_unverified_claude_models(config_path, tmp_path):
    models = _service().list_playable_models(
        config_path,
        {
            "CODEX_AUTH_JSON_B64": "encoded-codex-login",
            "CLAUDE_CODE_OAUTH_TOKEN": "claude-subscription-token",
            "CLAUDE_MODEL_CATALOG_PATH": str(tmp_path / "missing-catalog.json"),
        },
    )

    assert [model["id"] for model in models] == ["codex-local"]


def test_model_list_adds_only_claude_models_verified_for_subscription(
    config_path,
    tmp_path,
):
    catalog_path = tmp_path / "claude-models.json"
    catalog_path.write_text(json.dumps({
        "models": ["claude-fable-5-1", "opus"],
    }))

    models = _service().list_playable_models(
        config_path,
        {
            "CLAUDE_CODE_OAUTH_TOKEN": "claude-subscription-token",
            "CLAUDE_MODEL_CATALOG_PATH": str(catalog_path),
        },
    )

    assert [model["id"] for model in models] == [
        "claude-fable-5.1",
        "claude-code-local",
    ]


def test_verified_web_only_claude_model_can_start_game(config_path, tmp_path):
    catalog_path = tmp_path / "claude-models.json"
    catalog_path.write_text(json.dumps({"models": ["claude-fable-5-1"]}))

    state = _service().start_game(
        "claude-fable-5.1",
        "white",
        config_path,
        {
            "CLAUDE_CODE_OAUTH_TOKEN": "claude-subscription-token",
            "CLAUDE_MODEL_CATALOG_PATH": str(catalog_path),
        },
        move_provider=lambda *_args: pytest.fail("LLM should not move first"),
    )

    assert state["model_id"] == "claude-fable-5.1"
    assert state["model_name"] == "claude-fable-5-1"


def test_start_game_with_white_pieces_waits_for_human(config_path):
    service = _service()
    state = service.start_game(
        "chat-model",
        "white",
        config_path,
        {"OPENROUTER_API_KEY": "key"},
        move_provider=lambda *_args: pytest.fail("LLM should not move first"),
    )

    assert state == {
        "model_id": "chat-model",
        "model_name": "provider/chat-model",
        "reasoning_effort": "default",
        "human_color": "white",
        "moves": [],
        "status": "active",
        "winner": None,
        "termination": None,
        "llm_illegal_moves": 0,
    }


def test_start_game_preserves_immutable_lichess_snapshot(config_path):
    service = _service()

    state = service.start_game(
        "chat-model",
        "white",
        config_path,
        {"OPENROUTER_API_KEY": "key"},
        move_provider=lambda *_args: pytest.fail("LLM should not move first"),
        human_profile={
            "username": "Some_Player",
            "rating": 1847,
            "rating_deviation": 73,
            "provisional": True,
        },
    )

    assert state["game_id"]
    assert state["started_at"].endswith("+00:00")
    assert state["human_profile"] == {
        "username": "Some_Player",
        "rating": 1847,
        "rating_deviation": 73,
        "provisional": True,
    }
    assert service.game_view(state)["human_profile"] == state["human_profile"]


def test_start_game_with_black_pieces_requests_llm_opening(config_path):
    service = _service()
    calls = []

    state = service.start_game(
        "chat-model",
        "black",
        config_path,
        {"OPENROUTER_API_KEY": "key"},
        move_provider=lambda _config, board, is_retry, previous: (
            calls.append((board.fen(), is_retry, previous)) or "e2e4"
        ),
    )

    assert state["moves"] == ["e2e4"]
    assert state["llm_illegal_moves"] == 0
    assert calls[0][1:] == (False, None)
    assert service.game_view(state)["turn"] == "human"


def test_legal_human_move_and_llm_reply_advance_authoritative_state(config_path):
    service = _service()
    state = service.start_game(
        "chat-model",
        "white",
        config_path,
        {"OPENROUTER_API_KEY": "key"},
        move_provider=lambda *_args: "unused",
    )

    updated = service.play_human_move(
        state,
        "e2e4",
        config_path,
        {"OPENROUTER_API_KEY": "key"},
        move_provider=lambda *_args: "e7e5",
    )

    assert state["moves"] == []
    assert updated["moves"] == ["e2e4", "e7e5"]
    assert service.game_view(updated)["san_moves"] == ["e4", "e5"]
    assert service.game_view(updated)["turn"] == "human"


def test_game_view_exports_a_standard_pgn(config_path):
    service = _service()
    state = service.start_game(
        "chat-model",
        "white",
        config_path,
        {"OPENROUTER_API_KEY": "key"},
        move_provider=lambda *_args: "unused",
    )
    updated = service.play_human_move(
        state,
        "e2e4",
        config_path,
        {"OPENROUTER_API_KEY": "key"},
        move_provider=lambda *_args: "e7e5",
    )

    view = service.game_view(updated)
    exported = chess.pgn.read_game(io.StringIO(view["pgn"]))

    assert exported is not None
    assert exported.headers["Event"] == "Human vs LLM"
    assert exported.headers["White"] == "You"
    assert exported.headers["Black"] == "chat-model"
    assert exported.headers["Result"] == "*"
    assert [move.uci() for move in exported.mainline_moves()] == ["e2e4", "e7e5"]


@pytest.mark.parametrize("move", ["e2e5", "not-a-move", ""])
def test_illegal_human_move_is_rejected_without_mutating_state(config_path, move):
    service = _service()
    state = service.start_game(
        "chat-model",
        "white",
        config_path,
        {"OPENROUTER_API_KEY": "key"},
        move_provider=lambda *_args: "unused",
    )
    original = copy.deepcopy(state)

    with pytest.raises(service.GameStateError, match="legal move"):
        service.play_human_move(
            state,
            move,
            config_path,
            {"OPENROUTER_API_KEY": "key"},
            move_provider=lambda *_args: pytest.fail("LLM should not be called"),
        )

    assert state == original


def test_human_cannot_move_during_llm_turn(config_path):
    service = _service()
    state = {
        "model_id": "chat-model",
        "model_name": "provider/chat-model",
        "human_color": "black",
        "moves": [],
        "status": "active",
        "winner": None,
        "termination": None,
        "llm_illegal_moves": 0,
    }

    with pytest.raises(service.GameStateError, match="not your turn"):
        service.play_human_move(
            state,
            "e7e5",
            config_path,
            {"OPENROUTER_API_KEY": "key"},
            move_provider=lambda *_args: pytest.fail("LLM should not be called"),
        )


def test_promotion_move_is_accepted(config_path):
    service = _service()
    state = {
        "model_id": "chat-model",
        "model_name": "provider/chat-model",
        "human_color": "white",
        "moves": [
            "a2a4", "h7h5", "a4a5", "h5h4",
            "a5a6", "h4h3", "a6b7", "h3g2",
        ],
        "status": "active",
        "winner": None,
        "termination": None,
        "llm_illegal_moves": 0,
    }

    updated = service.play_human_move(
        state,
        "b7a8q",
        config_path,
        {"OPENROUTER_API_KEY": "key"},
        move_provider=lambda *_args: "b8a6",
    )

    assert updated["moves"][-2:] == ["b7a8q", "b8a6"]


def test_llm_checkmate_marks_game_finished(config_path):
    service = _service()
    state = {
        "model_id": "chat-model",
        "model_name": "provider/chat-model",
        "human_color": "white",
        "moves": ["f2f3", "e7e5"],
        "status": "active",
        "winner": None,
        "termination": None,
        "llm_illegal_moves": 0,
    }

    updated = service.play_human_move(
        state,
        "g2g4",
        config_path,
        {"OPENROUTER_API_KEY": "key"},
        move_provider=lambda *_args: "d8h4",
    )

    assert updated["status"] == "finished"
    assert updated["winner"] == "llm"
    assert updated["termination"] == "checkmate"
    view = service.game_view(updated)
    exported = chess.pgn.read_game(io.StringIO(view["pgn"]))
    assert view["turn"] == "finished"
    assert exported is not None
    assert exported.headers["Result"] == "0-1"


def test_llm_gets_one_retry_after_first_illegal_move(config_path):
    service = _service()
    replies = iter(["a1a1", "e2e4"])
    calls = []

    state = service.start_game(
        "chat-model",
        "black",
        config_path,
        {"OPENROUTER_API_KEY": "key"},
        move_provider=lambda _config, _board, is_retry, previous: (
            calls.append((is_retry, previous)) or next(replies)
        ),
    )

    assert state["moves"] == ["e2e4"]
    assert state["llm_illegal_moves"] == 1
    assert calls == [(False, None), (True, "a1a1")]


def test_second_llm_illegal_move_forfeits_without_another_retry(config_path):
    service = _service()
    replies = iter(["a1a1", "e2e4", "a1a1"])
    calls = []
    provider = lambda _config, _board, is_retry, previous: (
        calls.append((is_retry, previous)) or next(replies)
    )
    state = service.start_game(
        "chat-model",
        "black",
        config_path,
        {"OPENROUTER_API_KEY": "key"},
        move_provider=provider,
    )

    updated = service.play_human_move(
        state,
        "e7e5",
        config_path,
        {"OPENROUTER_API_KEY": "key"},
        move_provider=provider,
    )

    assert updated["status"] == "finished"
    assert updated["winner"] == "human"
    assert updated["termination"] == "llm_forfeit_illegal_move"
    assert len(calls) == 3


def test_provider_error_does_not_mutate_game_state(config_path):
    service = _service()
    state = service.start_game(
        "chat-model",
        "white",
        config_path,
        {"OPENROUTER_API_KEY": "key"},
        move_provider=lambda *_args: "unused",
    )
    original = copy.deepcopy(state)

    with pytest.raises(service.ProviderError, match="could not provide"):
        service.play_human_move(
            state,
            "e2e4",
            config_path,
            {"OPENROUTER_API_KEY": "key"},
            move_provider=lambda *_args: (_ for _ in ()).throw(
                TransientAPIError("provider unavailable")
            ),
        )

    assert state == original


def test_direct_gemini_provider_constructs_and_returns_move(monkeypatch):
    service = _service()
    captured = {}

    class FakeGeminiPlayer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def close(self):
            captured["closed"] = True

    async def fake_request(_player, _board, **_kwargs):
        return "e2e4"

    monkeypatch.setattr(service, "GeminiPlayer", FakeGeminiPlayer)
    monkeypatch.setattr(service, "request_llm_move", fake_request)

    move = asyncio.run(service._request_model_move(
        {
            "player_id": "gemini-direct",
            "model_name": "google/gemini-direct",
            "api": "gemini",
        },
        __import__("chess").Board(),
        False,
        None,
        {"GEMINI_API_KEY": "gemini-key"},
    ))

    assert move == "e2e4"
    assert captured["model_name"] == "gemini-direct"
    assert captured["closed"] is True


@pytest.mark.parametrize(
    (
        "backend",
        "backend_field",
        "class_name",
        "model_name",
        "credential",
        "expected_effort",
    ),
    [
        (
            "codex",
            "api",
            "CodexSubagentPlayer",
            "openai/gpt-5.6-sol",
            {"CODEX_AUTH_JSON_B64": "encoded-login"},
            "high",
        ),
        (
            "claude_code",
            "web_api",
            "ClaudeCodePlayer",
            "anthropic/claude-opus-4.7",
            {"CLAUDE_CODE_OAUTH_TOKEN": "subscription-token"},
            "low",
        ),
    ],
)
def test_subscription_cli_provider_constructs_configured_player(
    monkeypatch,
    backend,
    backend_field,
    class_name,
    model_name,
    credential,
    expected_effort,
):
    service = _service()
    captured = {}

    class FakeCLIPlayer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def close(self):
            captured["closed"] = True

    async def fake_request(_player, _board, **_kwargs):
        return "e2e4"

    monkeypatch.setattr(service, class_name, FakeCLIPlayer, raising=False)
    monkeypatch.setattr(service, "request_llm_move", fake_request)

    move = asyncio.run(service._request_model_move(
        dict({
            "player_id": f"{backend}-model",
            "model_name": model_name,
            "reasoning_effort": "high",
            "web_reasoning_effort": "low",
        }, **{backend_field: backend}),
        __import__("chess").Board(),
        False,
        None,
        credential,
    ))

    assert move == "e2e4"
    assert captured["model_name"] == model_name
    assert captured["reasoning_effort"] == expected_effort
    assert captured.get("subscription_only", False) is (backend == "codex")
    assert captured["closed"] is True


def test_claude_code_provider_can_use_a_cli_model_alias(monkeypatch):
    service = _service()
    captured = {}

    class FakeClaudeCodePlayer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def close(self):
            return None

    async def fake_request(_player, _board, **_kwargs):
        return "e2e4"

    monkeypatch.setattr(service, "ClaudeCodePlayer", FakeClaudeCodePlayer)
    monkeypatch.setattr(service, "request_llm_move", fake_request)

    asyncio.run(service._request_model_move(
        {
            "player_id": "latest-opus",
            "model_name": "anthropic/claude-opus-4.7",
            "web_model_name": "opus",
            "web_api": "claude_code",
        },
        __import__("chess").Board(),
        False,
        None,
        {"CLAUDE_CODE_OAUTH_TOKEN": "subscription-token"},
    ))

    assert captured["model_name"] == "opus"
