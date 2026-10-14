"""Draw commands, request capabilities, and provider output boundaries."""

import asyncio
import hashlib

import chess
import pytest

from llm import protocol
from llm.base_llm import BaseLLMPlayer, request_llm_move
from llm.claude_code_client import ClaudeCodePlayer
from llm.codex_subagent_client import CodexSubagentPlayer
from llm.gemini_client import GeminiPlayer
from llm.openrouter_client import OpenRouterPlayer
from llm.openrouter_completion_client import OpenRouterCompletionPlayer
from llm.prompts import build_chess_prompt


def players():
    return [
        OpenRouterPlayer("openrouter", "test", api_key="test"),
        CodexSubagentPlayer("codex", "test", max_concurrent=1),
        ClaudeCodePlayer("claude", "test"),
        OpenRouterCompletionPlayer("completion", "test", api_key="test"),
    ]


@pytest.mark.parametrize("text, expected", [
    ("accept_draw", "accept_draw"),
    (" ACCEPT_DRAW \n", "accept_draw"),
    ("E2E4 OFFER_DRAW", "e2e4 offer_draw"),
    ("a7a8Q offer_draw", "a7a8q offer_draw"),
    ("offer_draw", None), ("e4 offer_draw", None),
    ("a1a1 offer_draw", None), ("0000 offer_draw", None),
    ("e2e9 offer_draw", None), ("e7e8k offer_draw", None),
    ("e2e4 offer_draw please", None), ("I accept_draw", None),
    ("accept_draw e2e4", None), ("accept_draw\nresign", None),
    ("e2e4\noffer_draw", None), ("e2e4 offer_draw e7e5", None),
    ("`accept_draw`", None), ("MOVE: accept_draw", None),
    ("resign", None), ("", None), (None, None),
])
def test_strict_draw_action_protocol(text, expected):
    parse = getattr(protocol, "parse_draw_action", None)
    assert callable(parse), "The shared draw-action parser is missing"
    assert parse(text) == expected


def test_split_draw_offer_preserves_other_actions():
    split = getattr(protocol, "split_draw_offer", None)
    assert callable(split), "The canonical draw-offer splitter is missing"
    assert split("a7a8q offer_draw") == ("a7a8q", True)
    for action in ("e2e4", "accept_draw", "resign", "bad offer_draw"):
        assert split(action) == (action, False)


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("pending", [False, True])
def test_prompt_capabilities_include_offers_and_only_pending_acceptance(retry, pending):
    board = chess.Board()
    common = dict(allow_draw_offers=True, opponent_draw_offer=pending)
    prompts = [build_chess_prompt(board, is_retry=retry, illegal_move="a1a1", **common)]
    for player in players()[1:]:
        if isinstance(player, OpenRouterCompletionPlayer):
            prompts.append(player._build_prompt(board, **common))
        else:
            prompts.append(player._build_prompt(board, retry, "a1a1", **common))
    for prompt in prompts:
        assert "<uci> offer_draw" in prompt
        assert ("accept_draw" in prompt) is pending
        assert "UCI" in prompt


def test_disabled_capability_prompts_match_original_probe_bytes():
    board = chess.Board()
    _, codex, claude, completion = players()
    original = [
        (build_chess_prompt(board), "bcb3ba2b49260efbbb99ade0c1d1c54881643e3b72834ad7d324c3bca55aece2"),
        (codex._build_prompt(board, False, None), "7613d9a692fdeefb2944acdbe7a770ecbf23cd62f1422435e1d6c30da4806c6f"),
        (claude._build_prompt(board, False, None), "9aa7690b271d22a200e20ba8e4ffe51bdbd465cb3746a65149b03856957bdf7e"),
        (completion._build_prompt(board), "d7a02b6699c54689411377390588a179d20d4921d6f3156353f4c6fdfffa9dc7"),
    ]
    for prompt, digest in original:
        assert hashlib.sha256(prompt.encode()).hexdigest() == digest
    assert build_chess_prompt(board, opponent_draw_offer=True) == original[0][0]


@pytest.mark.parametrize("raises", [False, True])
def test_request_scopes_draw_capabilities_and_restores_even_on_failure(raises):
    class Player(BaseLLMPlayer):
        async def select_move(self, board, **kwargs):
            assert self.allow_draw_offers is True
            assert self.opponent_draw_offer is True
            assert kwargs == {"is_retry": True, "last_move_illegal": "a1a1"}
            if raises:
                raise RuntimeError("provider failed")
            return " accept_draw "
        async def close(self):
            pass
    player = Player("test", "test")
    player.allow_draw_offers = False
    player.opponent_draw_offer = "old value"
    async def run():
        return await request_llm_move(player, chess.Board(), is_retry=True,
                                      last_move_illegal="a1a1", allow_resignation=True,
                                      allow_draw_offers=True, opponent_draw_offer=True)
    if raises:
        with pytest.raises(RuntimeError, match="provider failed"):
            asyncio.run(run())
    else:
        assert asyncio.run(run()) == "accept_draw"
    assert player.allow_draw_offers is False
    assert player.opponent_draw_offer == "old value"


@pytest.mark.parametrize("text", ["accept_draw", "e2e4 offer_draw", "a7a8Q OFFER_DRAW"])
def test_every_backend_preserves_canonical_draw_actions(text):
    board = chess.Board()
    openrouter, codex, claude, completion = players()
    expected = text.lower()
    assert openrouter._parse_move(text, board) == expected
    assert GeminiPlayer._parse_move(None, text, board) == expected
    assert codex._parse_move(text, board) == expected
    assert completion._parse_san(text, board) == expected
    async def fake_cli(prompt):
        return text, {}
    claude._run_cli = fake_cli
    assert asyncio.run(claude.select_move(board)) == expected


def test_provider_parsers_do_not_promote_ambiguous_text_to_draw_actions():
    board = chess.Board()
    openrouter, codex, _, completion = players()
    for text in ("MOVE: accept_draw", "I accept_draw", "accept_draw resign", "e2e4 offer_draw please"):
        for parse in (openrouter._parse_move, codex._parse_move, completion._parse_san,
                      lambda text, board: GeminiPlayer._parse_move(None, text, board)):
            result = parse(text, board)
            assert result not in {"accept_draw", "e2e4 offer_draw"}


@pytest.mark.parametrize("text", ["accept_draw", "a7a8q offer_draw"])
def test_completion_returns_draw_action_without_sampling_it_away(text):
    player = OpenRouterCompletionPlayer("completion", "test", api_key="test")
    player.allow_draw_offers = True
    player.opponent_draw_offer = True
    calls = []
    async def session():
        return object()
    async def post(session, headers, payload):
        calls.append(payload)
        return {"choices": [{"text": text}], "usage": {"prompt_tokens": 10, "completion_tokens": 6}}
    player._ensure_session = session
    player._post_with_retries = post
    result = asyncio.run(player.select_move(chess.Board(), is_retry=True, last_move_illegal="a1a1"))
    assert result == text
    assert len(calls) == 1
    assert calls[0]["max_tokens"] >= 16
    assert player.max_tokens == 8
    assert "accept_draw" in calls[0]["prompt"]
    assert player.completion_tokens == 6


def test_completion_probe_retains_original_token_limit():
    player = OpenRouterCompletionPlayer("completion", "test", api_key="test")
    calls = []
    async def session():
        return object()
    async def post(session, headers, payload):
        calls.append(payload)
        return {"choices": [{"text": "e4"}]}
    player._ensure_session = session
    player._post_with_retries = post
    assert asyncio.run(player.select_move(chess.Board())) == "e2e4"
    assert calls[0]["max_tokens"] == 8


def test_truncated_invalid_cli_output_cannot_turn_into_acceptance():
    _, codex, claude, _ = players()
    text = "accept_draw" + " " * 100 + "actually keep playing"
    async def fake_cli(prompt):
        return text, {}
    for player, method in ((codex, "_run_codex"), (claude, "_run_cli")):
        setattr(player, method, fake_cli)
        result = asyncio.run(player.select_move(chess.Board()))
        assert result.strip() != "accept_draw"


def test_request_supports_probe_players_without_new_capability_attributes():
    class Probe:
        allow_resignation = False
        rating_context = None
        async def select_move(self, board, **kwargs):
            assert self.allow_draw_offers is False
            assert self.opponent_draw_offer is False
            return "e2e4"
    player = Probe()
    result = asyncio.run(request_llm_move(player, chess.Board(), is_retry=False,
                                          last_move_illegal=None, allow_resignation=False))
    assert result == "e2e4"
    assert not hasattr(player, "allow_draw_offers")
    assert not hasattr(player, "opponent_draw_offer")


@pytest.mark.parametrize("enabled", [False, True])
def test_openrouter_stream_forwards_game_capabilities_and_preserves_probe_limit(enabled):
    import json
    from types import SimpleNamespace

    player = OpenRouterPlayer("openrouter", "test", api_key="test", max_tokens=8)
    payloads = []
    class Stream:
        async def iter_any(self):
            for part in ("e2e4", " offer_draw"):
                data = {"choices": [{"delta": {"content": part}}]}
                yield ("data: " + json.dumps(data) + "\n\n").encode()
            yield b'data: {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 6}}\n\ndata: [DONE]\n\n'
    class Response:
        status = 200
        content = Stream()
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
    def post(*args, **kwargs):
        payloads.append(kwargs["json"])
        return Response()
    async def session():
        return SimpleNamespace(post=post)
    player._ensure_session = session
    result = asyncio.run(request_llm_move(
        player, chess.Board(), is_retry=True, last_move_illegal="a1a1",
        allow_resignation=True, allow_draw_offers=enabled, opponent_draw_offer=True,
    ))
    assert result == "e2e4 offer_draw"
    assert ("accept_draw" in player.last_prompt) is enabled
    assert payloads[0]["max_tokens"] == (16 if enabled else 8)
    assert player.completion_tokens == 6
    assert player.max_tokens == 8
    assert player.allow_draw_offers is False


@pytest.mark.parametrize("text, expected", [
    ("ACCEPT_DRAW", "accept_draw"),
    ("a7a8Q offer_draw", "a7a8q offer_draw"),
    ("accept_draw" + " " * 100 + "no", "accept_draw..."),
])
def test_gemini_request_preserves_commands_and_does_not_truncate_into_acceptance(text, expected):
    from types import SimpleNamespace

    player = GeminiPlayer.__new__(GeminiPlayer)
    BaseLLMPlayer.__init__(player, "gemini", "gemini-test")
    player.temperature = 0.0
    player.reasoning = False
    player.reasoning_effort = None
    player._genai = SimpleNamespace(types=SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs))
    async def generate(**kwargs):
        return SimpleNamespace(text=text, usage_metadata=None)
    async def close(client):
        pass
    player._create_client = lambda: SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    player._close_client = close
    result = asyncio.run(request_llm_move(
        player, chess.Board(), is_retry=True, last_move_illegal="a1a1",
        allow_resignation=True, allow_draw_offers=True, opponent_draw_offer=True,
    ))
    assert result == expected
    assert "accept_draw" in player.last_prompt
    assert "a1a1" in player.last_prompt
    assert player.allow_draw_offers is False


@pytest.mark.parametrize("limit", [20, 80])
def test_debug_truncation_preserves_non_action_evidence(limit):
    text = "accept_draw" + " " * 100 + "no"
    assert protocol.unparsed_response(text, limit).strip() != "accept_draw"
    assert protocol.unparsed_response("ordinary bad output", limit) == "ordinary bad output"
    assert protocol.unparsed_response(None, limit) == ""
    assert protocol.unparsed_response("accept_draw", limit) == "accept_draw"


def test_completion_draw_prompt_keeps_normal_san_moves_explicit():
    completion = OpenRouterCompletionPlayer("completion", "test", api_key="test")
    prompt = completion._build_prompt(chess.Board(), allow_draw_offers=True)
    assert "SAN move" in prompt
    assert "Use UCI notation for a move with an offer" in prompt


@pytest.mark.parametrize("resignation", [False, True])
def test_pending_acceptance_prompt_lists_acceptance_as_an_action_without_requiring_a_move(resignation):
    prompt = build_chess_prompt(chess.Board(), allow_resignation=resignation,
                                allow_draw_offers=True, opponent_draw_offer=True)
    assert "Choose exactly ONE action for White" in prompt
    assert "- Play exactly ONE legal move" not in prompt
    assert "accept_draw" in prompt
