import chess

from llm.claude_code_client import ClaudeCodePlayer
from llm.codex_subagent_client import CodexSubagentPlayer
from llm.gemini_client import GeminiPlayer
from llm.openrouter_client import OpenRouterPlayer
from llm.openrouter_completion_client import OpenRouterCompletionPlayer
from llm.prompts import build_chess_prompt


def test_chess_prompt_presents_resignation_as_neutral_optional_action():
    prompt = build_chess_prompt(chess.Board(), allow_resignation=True)

    assert "If you independently choose to resign, output exactly: resign" in prompt
    assert "Resignation is optional and should be based only on your assessment" in prompt
    assert "Only a legal move in UCI or the word resign" in prompt


def test_retry_prompt_still_allows_an_exact_resignation():
    prompt = build_chess_prompt(
        chess.Board(),
        is_retry=True,
        illegal_move="a1a1",
        allow_resignation=True,
    )

    assert 'previous attempted move "a1a1" was ILLEGAL' in prompt
    assert "If you independently choose to resign, output exactly: resign" in prompt


def test_provider_parsers_accept_only_unambiguous_resignation():
    board = chess.Board()
    openrouter = OpenRouterPlayer(
        player_id="openrouter-test",
        model_name="provider/test",
        api_key="test-key",
    )
    codex = CodexSubagentPlayer(
        player_id="codex-test",
        model_name="openai/gpt-test",
        reasoning_effort="low",
        max_concurrent=1,
        subscription_only=True,
    )

    parsers = [
        lambda text: openrouter._parse_move(text, board),
        lambda text: GeminiPlayer._parse_move(None, text, board),
        lambda text: codex._parse_move(text, board),
    ]
    for parse in parsers:
        assert parse("resign") == "resign"
        assert parse("RESIGN") == "resign"
        assert parse("I might resign, but I will keep playing.") is None
        assert parse("I considered resigning; e2e4") == "e2e4"


def test_completion_models_receive_and_parse_the_same_resignation_option():
    player = OpenRouterCompletionPlayer(
        player_id="completion-test",
        model_name="provider/completion-test",
        api_key="test-key",
    )
    board = chess.Board()

    prompt = player._build_prompt(board, allow_resignation=True)

    assert "output resign if you independently choose to resign" in prompt
    assert "Resignation is optional and should be based only on your assessment" in prompt
    assert player._parse_san("RESIGN", board) == "resign"
    assert player._parse_san("I might resign", board) is None


def test_position_probe_prompts_do_not_offer_resignation():
    board = chess.Board()
    completion = OpenRouterCompletionPlayer(
        player_id="completion-test",
        model_name="provider/completion-test",
        api_key="test-key",
    )

    assert "resign" not in build_chess_prompt(board).casefold()
    assert "resign" not in completion._build_prompt(board).casefold()


def test_cli_specific_prompts_do_not_override_the_resignation_option():
    board = chess.Board()
    codex = CodexSubagentPlayer(
        player_id="codex-test",
        model_name="openai/gpt-test",
        reasoning_effort="low",
        max_concurrent=1,
        subscription_only=True,
    )
    claude = ClaudeCodePlayer(
        player_id="claude-test",
        model_name="anthropic/claude-test",
        reasoning_effort="low",
    )

    assert "Return exactly one line: <uci> or resign" in codex._build_prompt(
        board, False, None, allow_resignation=True
    )
    assert "Return exactly one line: <uci> or resign" in claude._build_prompt(
        board, False, None, allow_resignation=True
    )


def test_codex_legal_move_hint_keeps_resignation_available():
    player = CodexSubagentPlayer(
        player_id="codex-test",
        model_name="openai/gpt-test",
        reasoning_effort="low",
        max_concurrent=1,
        subscription_only=True,
        include_legal_moves=True,
    )

    prompt = player._build_prompt(
        chess.Board(),
        False,
        None,
        allow_resignation=True,
    )

    assert "If choosing a move, choose one from these legal UCI moves" in prompt


def test_claude_position_command_contains_no_resignation_language():
    player = ClaudeCodePlayer(
        player_id="claude-test",
        model_name="anthropic/claude-test",
        reasoning_effort="low",
    )
    prompt = player._build_prompt(
        chess.Board(),
        False,
        None,
        allow_resignation=False,
    )

    assert "resign" not in " ".join(player._command(prompt)).casefold()
