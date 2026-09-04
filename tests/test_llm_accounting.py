from types import SimpleNamespace
import unittest

import chess

from llm.base_llm import BaseLLMPlayer
from llm.gemini_client import billed_completion_tokens


class DummyPlayer(BaseLLMPlayer):
    def __init__(self) -> None:
        super().__init__("dummy", "dummy/model")
        self._session = object()
        self._prefetched_response = {"move": "e2e4"}
        self._last_prompt_tokens = 11
        self._last_completion_tokens = 22
        self.last_api_error = "old error"

    async def select_move(
        self,
        board: chess.Board,
        is_retry: bool = False,
        last_move_illegal: str = None,
    ) -> str:
        return "e2e4"

    async def close(self) -> None:
        return None


class ConcurrentPlayerIsolationTests(unittest.TestCase):
    def test_clone_for_game_resets_all_mutable_request_state(self) -> None:
        template = DummyPlayer()
        template.prompt_tokens = 101
        template.completion_tokens = 202
        template.total_tokens = 303
        template.move_times = [1.0, 2.0]
        template.total_move_time = 3.0
        template.last_prompt = "prompt"
        template.last_raw_response = "response"
        template.last_successful_response = "e2e4"

        clone = template.clone_for_game()

        self.assertIsNot(clone, template)
        self.assertIsNone(clone._session)
        self.assertIsNone(clone._prefetched_response)
        self.assertEqual(clone.last_api_error, "")
        self.assertEqual(clone.get_token_usage(), {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        })
        self.assertEqual(clone.get_timing_usage()["move_times"], [])
        self.assertEqual(clone._last_prompt_tokens, 0)
        self.assertEqual(clone._last_completion_tokens, 0)
        self.assertEqual(clone.last_prompt, "")
        self.assertEqual(clone.last_raw_response, "")
        self.assertEqual(clone.last_successful_response, "")

        clone.move_times.append(4.0)
        self.assertEqual(template.move_times, [1.0, 2.0])
        self.assertIsNotNone(template._session)
        self.assertEqual(template._prefetched_response, {"move": "e2e4"})


class GeminiAccountingTests(unittest.TestCase):
    def test_billed_completion_includes_hidden_thought_tokens(self) -> None:
        usage = SimpleNamespace(
            candidates_token_count=7,
            thoughts_token_count=193,
        )

        self.assertEqual(billed_completion_tokens(usage), 200)

    def test_billed_completion_supports_older_usage_metadata(self) -> None:
        usage = SimpleNamespace(candidates_token_count=7)

        self.assertEqual(billed_completion_tokens(usage), 7)


if __name__ == "__main__":
    unittest.main()
