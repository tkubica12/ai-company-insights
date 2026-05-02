from __future__ import annotations

from typing import Any

from ai_company_insights.models import TokenUsage


def merge_token_usage(*items: TokenUsage | None) -> TokenUsage:
    return TokenUsage(
        input_tokens=sum(item.input_tokens for item in items if item),
        cached_input_tokens=sum(item.cached_input_tokens for item in items if item),
        output_tokens=sum(item.output_tokens for item in items if item),
    )


def extract_token_usage(value: Any) -> TokenUsage:
    usage = _get(value, "usage")
    if usage is None:
        usage = value
    details = (
        _get(usage, "input_tokens_details")
        or _get(usage, "prompt_tokens_details")
        or _get(usage, "input_token_details")
        or {}
    )
    input_tokens = _int_value(
        _get(usage, "input_tokens"),
        _get(usage, "prompt_tokens"),
        _get(usage, "total_input_tokens"),
    )
    cached_input_tokens = _int_value(
        _get(details, "cached_tokens"),
        _get(details, "cached_input_tokens"),
        _get(usage, "cached_input_tokens"),
        _get(usage, "cache_read_input_tokens"),
    )
    output_tokens = _int_value(
        _get(usage, "output_tokens"),
        _get(usage, "completion_tokens"),
        _get(usage, "total_output_tokens"),
    )
    return TokenUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
    )


def _get(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _int_value(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0
