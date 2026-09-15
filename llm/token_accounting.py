"""Auditable chess input estimates; o200k_base is not a native model tokenizer."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def chess_usage(usage: dict, prompt: str, *, runtime: bool = True) -> dict:
    """Retain raw counts and allocate cache to the runtime prefix before chess.

    CLI chess counts and cache overlaps are estimates, bounded by reported input.
    API callers without a runtime prefix can use their reported input directly.
    """
    total = int(usage.get("prompt_tokens", 0) or 0)
    output = int(usage.get("completion_tokens", 0) or 0)
    read = int(usage.get("cached_input_tokens", 0) or 0)
    write = int(usage.get("cache_creation_input_tokens", 0) or 0)
    estimated = total
    method = "provider_reported_chess_input"
    if runtime:
        try:
            estimated = len(_encoding().encode(prompt, disallowed_special=()))
            method = "estimated_o200k_base_runtime_prefix_cache"
        except Exception:
            # Accounting must not discard an already completed model move if
            # tokenizer installation or first-use vocabulary retrieval fails.
            estimated = (len(prompt.encode("utf-8")) + 3) // 4
            method = "estimated_utf8_quarter_runtime_prefix_cache"
    chess = min(max(0, total), estimated)
    overhead = max(0, total - chess)
    chess_read = min(chess, max(0, read - overhead))
    chess_write = min(chess - chess_read, max(0, write - max(0, overhead - read)))
    result = {
        "prompt_tokens": total,
        "completion_tokens": output,
        "total_tokens": total + output,
        "cached_input_tokens": read,
        "cache_creation_input_tokens": write,
        "chess_prompt_tokens": chess,
        "chess_cached_input_tokens": chess_read,
        "chess_cache_creation_input_tokens": chess_write,
        "runtime_prompt_tokens": overhead,
        "input_accounting_method": method,
        "cache_accounting_known": bool(usage.get(
            "cache_accounting_known", "cached_input_tokens" in usage
        )),
    }
    # TTL order inside a cached prefix is unknown. Allocate the cheaper 5m
    # writes to overhead first so the estimated chess cost stays conservative.
    remaining_runtime = max(0, overhead - read)
    remaining_chess_write = chess_write
    for ttl in ("5m", "1h"):
        key = f"cache_creation_{ttl}_input_tokens"
        if key in usage:
            raw = int(usage[key] or 0)
            result[key] = raw
            overlap = min(remaining_chess_write, max(0, raw - remaining_runtime))
            result[f"chess_{key}"] = overlap
            remaining_runtime = max(0, remaining_runtime - raw)
            remaining_chess_write -= overlap
    return result
