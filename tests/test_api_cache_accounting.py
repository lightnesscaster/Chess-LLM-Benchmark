import pytest

from llm.openrouter_client import OpenRouterPlayer
from llm.openrouter_completion_client import OpenRouterCompletionPlayer


@pytest.mark.parametrize("client", [OpenRouterPlayer, OpenRouterCompletionPlayer])
def test_api_usage_retains_cache_without_subtracting_runtime(client):
    player = client(player_id="test", model_name="openai/test", api_key="unused")
    player._track_usage({"usage": {"prompt_tokens": 1000, "completion_tokens": 25,
        "total_tokens": 1025, "prompt_tokens_details": {"cached_tokens": 600,
                                                       "cache_write_tokens": 100}}})
    usage = player.get_token_usage()
    assert usage["prompt_tokens"] == 1000
    assert usage["chess_prompt_tokens"] == 1000
    assert usage["chess_cached_input_tokens"] == 600
    assert usage["chess_cache_creation_input_tokens"] == 100
    assert usage["runtime_prompt_tokens"] == 0
    player.reset_token_usage()
    assert player.get_token_usage()["prompt_tokens"] == 0
