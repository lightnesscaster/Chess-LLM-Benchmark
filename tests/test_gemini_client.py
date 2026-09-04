from types import SimpleNamespace
import unittest

import chess

from llm.base_llm import BaseLLMPlayer
from llm.gemini_client import GeminiPlayer


class _FakeConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeModels:
    def __init__(self) -> None:
        self.request = None

    async def generate_content(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(usage_metadata=None, text="e2e4")


class _FakeClient:
    def __init__(self) -> None:
        self.aio = SimpleNamespace(models=_FakeModels())


class GeminiGenerationConfigTests(unittest.IsolatedAsyncioTestCase):
    async def _request_for(self, model_name: str) -> dict:
        player = GeminiPlayer.__new__(GeminiPlayer)
        BaseLLMPlayer.__init__(
            player,
            f"{model_name} (medium)",
            model_name,
        )
        player.temperature = 0.0
        player.reasoning = None
        player.reasoning_effort = "medium"
        player._last_prompt_tokens = 0
        player._last_completion_tokens = 0
        player._genai = SimpleNamespace(
            types=SimpleNamespace(
                GenerateContentConfig=_FakeConfig,
                ThinkingConfig=_FakeConfig,
            )
        )
        client = _FakeClient()
        player._create_client = lambda: client

        async def close_client(_client) -> None:
            return None

        player._close_client = close_client

        move = await player.select_move(chess.Board())

        self.assertEqual(move, "e2e4")
        return client.aio.models.request

    async def test_sampling_parameter_support_changes_at_gemini_36(self) -> None:
        for model_name, supports_temperature in (
            ("gemini-3.5-flash", True),
            ("gemini-3.6-flash", False),
            ("gemini-3.8-flash", False),
        ):
            with self.subTest(model_name=model_name):
                request = await self._request_for(model_name)
                config = request["config"]

                self.assertEqual(request["model"], model_name)
                if supports_temperature:
                    self.assertEqual(config.kwargs["temperature"], 0.0)
                else:
                    self.assertNotIn("temperature", config.kwargs)
                self.assertEqual(
                    config.kwargs["thinking_config"].kwargs,
                    {"thinking_level": "medium"},
                )


if __name__ == "__main__":
    unittest.main()
