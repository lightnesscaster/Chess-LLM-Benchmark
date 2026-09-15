# Chess-only token cost accounting

Game records retain provider-reported `prompt_tokens`, `completion_tokens`, and
`total_tokens`. Input totals include cached input. Claude input is the sum of
uncached input, cache reads, and cache writes; its `input_tokens` alone is not
total input.

New records additionally retain chess input, runtime input, cache-read/write
counts, optional write TTL counts, and per-request accounting. Direct API chess
requests use reported input without runtime subtraction. Codex and Claude Code
requests estimate the actual chess prompt with `o200k_base`; this is explicitly
an estimate, not a claim to reproduce each model's native tokenizer.
If the tokenizer is unavailable, an explicitly labeled UTF-8-byte estimate
keeps accounting failure from discarding an otherwise valid chess response.
Cached
overlap is estimated with the runtime prefix first, followed by chess input.
Raw counters are never replaced by the estimates. Output, including reported
reasoning tokens, remains chargeable.

The cost formula prices uncached chess input, cached chess input, chess cache
writes, and output separately using `config/pricing.json`. Unknown cache prices
fall back to the ordinary input rate, not zero. Cache rates were sourced from
the OpenRouter model catalogue on 2026-09-15; this change does not reprice base
input/output rates. These are benchmark API-equivalent costs, not subscription
invoices. Other subscription models retain their zero scheduling-budget
overrides; Astra and Fable use token-priced costs.

## Historical limitations

Old agent records did not retain the chess/runtime split. Codex discarded cache
reads; Claude Code discarded cache reads and writes. We cannot reconstruct
those measurements from aggregate totals or a PGN. Their displayed cost is a
**lower bound containing known output only**, with a missing-chess-input count
and explanation. This is not an assertion that chess input was free. The old
fixed 16,000-input-tokens-per-call subtraction is no longer used.

Historical records, results, ratings, and frozen flags are not rewritten by this
code change. Deploying a different cost policy changes the inputs to future
freezing checks; any production frozen-flag recomputation requires its own
preview and apply step. Preserve originals if recovered raw request logs later
allow a verified backfill.

References:
- https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- https://developers.openai.com/api/docs/models/gpt-6-astra
- https://openrouter.ai/api/v1/models
