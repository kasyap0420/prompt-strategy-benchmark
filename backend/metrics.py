import re
from typing import Literal, Mapping

BenchmarkStatus = Literal[
    "success",
    "timeout",
    "rate_limit",
    "invalid_request",
    "authentication_error",
    "api_error",
    "validation_error",
    "parsing_error",
    "unexpected_error",
]

TOKEN_METRIC_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def calculate_latency_ms(start_time: float, end_time: float) -> int:
    return max(0, int(round((end_time - start_time) * 1000)))


def calculate_response_length(response_text: str | None) -> int:
    return len(response_text or "")


def calculate_word_count(response_text: str | None) -> int:
    return len(re.findall(r"\S+", response_text or ""))


def format_benchmark_metrics(
    *,
    start_time: float,
    end_time: float,
    response_text: str | None,
    status: BenchmarkStatus,
    token_usage: Mapping[str, int | None] | None = None,
) -> dict[str, int | str | None]:
    metrics: dict[str, int | str | None] = {
        "latency_ms": calculate_latency_ms(start_time, end_time),
        "response_length": calculate_response_length(response_text),
        "word_count": calculate_word_count(response_text),
        "status": status,
    }

    for key in TOKEN_METRIC_KEYS:
        metrics[key] = token_usage.get(key) if token_usage else None

    return metrics
