import asyncio

import pytest

from backend.benchmark_engine import BenchmarkEngine
from backend.gemini_client import (
    GeminiAPIRequestError,
    GeminiAuthenticationError,
    GeminiClient,
    GeminiGeneration,
    GeminiInvalidRequestError,
    GeminiParsingError,
    GeminiRateLimitError,
    GeminiTimeoutError,
)
from backend.schemas import BenchmarkRequest


class FakeAPIError:
    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message


class FlakyGeminiClient(GeminiClient):
    def __init__(self) -> None:
        super().__init__(api_key="test-key", max_retries=2, retry_base_delay_seconds=0)
        self.call_count = 0

    async def _generate_content_once_async(self, cleaned_prompt: str) -> GeminiGeneration:
        self.call_count += 1
        if self.call_count < 3:
            raise GeminiTimeoutError("Temporary timeout.")
        return GeminiGeneration(text="Recovered", input_tokens=None, output_tokens=None)


class RateLimitedGeminiClient(GeminiClient):
    def __init__(self) -> None:
        super().__init__(api_key="test-key", max_retries=2, retry_base_delay_seconds=0)
        self.call_count = 0

    async def _generate_content_once_async(self, cleaned_prompt: str) -> GeminiGeneration:
        self.call_count += 1
        raise GeminiRateLimitError("Gemini rate limit exceeded.", status_code=429)


class MissingTokenMetadataResponse:
    text = "Response without token metadata"


class EmptyTextResponse:
    text = ""
    usage_metadata = None


class SuccessfulGeminiClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_content_async(self, prompt: str) -> GeminiGeneration:
        self.prompts.append(prompt)
        return GeminiGeneration(text="ok")


class ReliabilityFailingGeminiClient:
    def __init__(self) -> None:
        self.call_count = 0

    async def generate_content_async(self, prompt: str) -> GeminiGeneration:
        self.call_count += 1
        if self.call_count == 1:
            raise GeminiTimeoutError("Timed out.")
        if self.call_count == 2:
            raise GeminiRateLimitError("Rate limited.", status_code=429)
        if self.call_count == 3:
            raise GeminiInvalidRequestError("Bad prompt.", status_code=400)
        if self.call_count == 4:
            raise GeminiParsingError("Gemini returned an empty response.")
        return GeminiGeneration(text="success")


def test_transient_errors_are_retryable_and_can_recover() -> None:
    client = FlakyGeminiClient()

    generation = asyncio.run(client.generate_content_async("hello"))

    assert generation.text == "Recovered"
    assert client.call_count == 3


def test_rate_limit_errors_do_not_retry() -> None:
    client = RateLimitedGeminiClient()

    with pytest.raises(GeminiRateLimitError):
        asyncio.run(client.generate_content_async("hello"))

    assert client.call_count == 1


def test_api_error_mapping_distinguishes_rate_limit_invalid_and_authentication() -> None:
    client = GeminiClient(api_key="test-key")

    rate_limit = client._map_api_error(FakeAPIError(429, "quota exceeded"))  # type: ignore[arg-type]
    invalid_request = client._map_api_error(FakeAPIError(400, "prompt rejected"))  # type: ignore[arg-type]
    authentication = client._map_api_error(FakeAPIError(403, "forbidden"))  # type: ignore[arg-type]
    server_error = client._map_api_error(FakeAPIError(503, "unavailable"))  # type: ignore[arg-type]

    assert isinstance(rate_limit, GeminiRateLimitError)
    assert rate_limit.status_code == 429
    assert rate_limit.retryable is False
    assert isinstance(invalid_request, GeminiInvalidRequestError)
    assert invalid_request.status_code == 400
    assert isinstance(authentication, GeminiAuthenticationError)
    assert authentication.status_code == 403
    assert isinstance(server_error, GeminiAPIRequestError)
    assert server_error.retryable is True


def test_missing_token_metadata_is_safe() -> None:
    generation = GeminiClient(api_key="test-key")._extract_generation(MissingTokenMetadataResponse())

    assert generation.text == "Response without token metadata"
    assert generation.input_tokens is None
    assert generation.output_tokens is None
    assert generation.total_tokens is None


def test_malformed_empty_response_raises_parsing_error() -> None:
    with pytest.raises(GeminiParsingError, match="empty response"):
        GeminiClient(api_key="test-key")._extract_generation(EmptyTextResponse())


def test_complete_benchmark_success_attempts_all_strategies() -> None:
    gemini_client = SuccessfulGeminiClient()
    engine = BenchmarkEngine(gemini_client=gemini_client)  # type: ignore[arg-type]

    response = asyncio.run(engine.run(BenchmarkRequest(user_input="Explain containers")))

    assert len(response.results) == 5
    assert len(gemini_client.prompts) == 5
    assert all(result.metrics.status == "success" for result in response.results)
    assert all(result.error_type is None for result in response.results)


def test_partial_benchmark_failure_preserves_successes_and_error_reasons() -> None:
    engine = BenchmarkEngine(gemini_client=ReliabilityFailingGeminiClient())  # type: ignore[arg-type]

    response = asyncio.run(engine.run(BenchmarkRequest(user_input="Explain containers")))

    assert [result.metrics.status for result in response.results] == [
        "timeout",
        "rate_limit",
        "invalid_request",
        "parsing_error",
        "success",
    ]
    assert response.results[1].error_type == "GeminiRateLimitError"
    assert response.results[2].error_message == "Bad prompt."
    assert response.results[4].response == "success"
