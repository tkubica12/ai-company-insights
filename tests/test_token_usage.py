from types import SimpleNamespace

from ai_company_insights.models import TokenUsage
from ai_company_insights.token_usage import extract_token_usage, merge_token_usage


def test_extract_token_usage_from_openai_response_shape() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=120,
            input_tokens_details=SimpleNamespace(cached_tokens=45),
            output_tokens=30,
        )
    )

    usage = extract_token_usage(response)

    assert usage.input_tokens == 120
    assert usage.cached_input_tokens == 45
    assert usage.output_tokens == 30


def test_merge_token_usage_sums_counts() -> None:
    usage = merge_token_usage(
        TokenUsage(input_tokens=10, cached_input_tokens=2, output_tokens=3),
        TokenUsage(input_tokens=5, cached_input_tokens=1, output_tokens=7),
    )

    assert usage.input_tokens == 15
    assert usage.cached_input_tokens == 3
    assert usage.output_tokens == 10
