from backend.metrics import (
    calculate_latency_ms,
    calculate_response_length,
    calculate_word_count,
    format_benchmark_metrics,
)


def test_calculate_latency_ms_uses_wall_clock_delta() -> None:
    assert calculate_latency_ms(10.0, 11.2344) == 1234


def test_response_length_and_word_count() -> None:
    response = "Docker packages apps into containers."

    assert calculate_response_length(response) == len(response)
    assert calculate_word_count(response) == 5


def test_format_benchmark_metrics_without_token_usage_returns_null_tokens() -> None:
    metrics = format_benchmark_metrics(
        start_time=1.0,
        end_time=1.5,
        response_text="hello world",
        status="success",
        token_usage=None,
    )

    assert metrics == {
        "latency_ms": 500,
        "response_length": 11,
        "word_count": 2,
        "status": "success",
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }


def test_format_benchmark_metrics_uses_real_token_metadata_when_available() -> None:
    metrics = format_benchmark_metrics(
        start_time=1.0,
        end_time=1.1,
        response_text="hello",
        status="success",
        token_usage={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
    )

    assert metrics["input_tokens"] == 3
    assert metrics["output_tokens"] == 4
    assert metrics["total_tokens"] == 7