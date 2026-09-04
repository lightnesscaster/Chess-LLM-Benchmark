#!/usr/bin/env python3
"""
Debug script: sends the exact move-33 prompt to gemini-3.1-pro-preview
to test if it's the specific prompt content or accumulated state that causes the 500.
"""
import os
import asyncio
from google import genai

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-3.1-pro-preview"

PROMPT = """You are playing chess as Black.

Move history:
1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Bg5 e6 7. f4 Be7 8. Qf3 Qc7 9. O-O-O Nbd7 10. g4 b5 11. Bxf6 Nxf6 12. g5 Nd7 13. f5 Nc5 14. f6 gxf6 15. gxf6 Bf8 16. Rg1 h5 17. Rg7

>>> White's last move: Rg7 (g1g7) <<<


Current position (FEN):
r1b1kb1r/2q2pR1/p2ppP2/1pn4p/3NP3/2N2Q2/PPP4P/2KR1B2 b kq - 1 17

Board:
8 | r . b . k b . r
7 | . . q . . p R .
6 | p . . p p P . .
5 | . p n . . . . p
4 | . . . N P . . .
3 | . . N . . Q . .
2 | P P P . . . . P
1 | . . K R . B . .
  +-----------------
    a b c d e f g h

Your task:
- Play exactly ONE legal move for Black.
- Use UCI notation only (examples: e2e4, g1f3, e7e8q for promotion).
- Do NOT include any commentary, explanations, or additional text.

Output format:
- Only the move in UCI, e.g.:
b1c3"""


async def test_prompt(thinking_level=None):
    client = genai.Client(
        api_key=API_KEY,
        http_options=genai.types.HttpOptions(timeout=300_000),
    )
    config_kwargs = {"temperature": 0.0}
    if thinking_level:
        config_kwargs["thinking_config"] = genai.types.ThinkingConfig(
            thinking_level=thinking_level,
        )
        config_kwargs["temperature"] = 0.0

    config = genai.types.GenerateContentConfig(**config_kwargs)

    print(f"Testing {MODEL} with thinking_level={thinking_level}...")
    try:
        response = await client.aio.models.generate_content(
            model=MODEL, contents=PROMPT, config=config,
        )
        print(f"  SUCCESS: {response.text}")
        if response.usage_metadata:
            print(f"  Tokens: prompt={response.usage_metadata.prompt_token_count}, "
                  f"completion={response.usage_metadata.candidates_token_count}, "
                  f"total={response.usage_metadata.total_token_count}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
    finally:
        try:
            await client.aio.aclose()
        finally:
            client.close()


async def main():
    # Test 1: With high thinking (what you're doing)
    await test_prompt("high")
    print()
    # Test 2: With low thinking
    await test_prompt("low")
    print()
    # Test 3: With medium thinking
    await test_prompt("medium")


if __name__ == "__main__":
    asyncio.run(main())
