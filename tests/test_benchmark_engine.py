import asyncio

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.benchmark_engine import BenchmarkEngine
from backend.gemini_client import GeminiAPIRequestError, GeminiGeneration, GeminiTimeoutError
from backend.main import app
from backend.prompt_strategies import list_strategy_names
from backend.schemas import BenchmarkRequest, BenchmarkResponse


class SuccessfulGeminiClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_content_async(self, prompt: str) -> GeminiGeneration:
        self.prompts.append(prompt)
        return GeminiGeneration(
            text=f"Response for: {prompt}",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )


class PartiallyFailingGeminiClient:
    def __init__(self) -> None:
        self.call_count = 0

    async def generate_content_async(self, prompt: str) -> GeminiGeneration:
        self.call_count += 1
        if self.call_count == 1:
            raise GeminiTimeoutError("Request timed out.")
        if self.call_count == 2:
            raise GeminiAPIRequestError("Rate limit exceeded.")
        if self.call_count == 3:
            raise RuntimeError("Unexpected failure.")
        return GeminiGeneration(text="Recovered response")


def test_successful_benchmark_execution_returns_five_results() -> None:
    gemini_client = SuccessfulGeminiClient()
    engine = BenchmarkEngine(gemini_client=gemini_client)  # type: ignore[arg-type]

    response = asyncio.run(engine.run(BenchmarkRequest(user_input="Explain Docker")))

    assert isinstance(response, BenchmarkResponse)
    assert len(response.results) == 5
    assert len(gemini_client.prompts) == 5
    assert [result.strategy_name for result in response.results] == list_strategy_names()
    assert all(result.response for result in response.results)
    assert all(result.metrics.status == "success" for result in response.results)


def test_benchmark_request_rejects_empty_input() -> None:
    with pytest.raises(ValidationError):
        BenchmarkRequest(user_input="   ")


def test_benchmark_metrics_are_generated_for_successful_calls() -> None:
    engine = BenchmarkEngine(gemini_client=SuccessfulGeminiClient())  # type: ignore[arg-type]

    response = asyncio.run(engine.run(BenchmarkRequest(user_input="Explain Docker")))
    first_result = response.results[0]

    assert first_result.metrics.latency_ms >= 0
    assert first_result.metrics.response_length == len(first_result.response or "")
    assert first_result.metrics.word_count > 0
    assert first_result.metrics.input_tokens == 10
    assert first_result.metrics.output_tokens == 20
    assert first_result.metrics.total_tokens == 30


def test_failed_gemini_calls_are_recorded_and_execution_continues() -> None:
    engine = BenchmarkEngine(gemini_client=PartiallyFailingGeminiClient())  # type: ignore[arg-type]

    response = asyncio.run(engine.run(BenchmarkRequest(user_input="Explain Docker")))

    assert len(response.results) == 5
    assert [result.metrics.status for result in response.results[:3]] == [
        "timeout",
        "api_error",
        "unexpected_error",
    ]
    assert response.results[0].error_type == "GeminiTimeoutError"
    assert response.results[1].error_message == "Rate limit exceeded."
    assert response.results[0].response is None
    assert response.results[1].response is None
    assert response.results[2].response is None
    assert response.results[3].metrics.status == "success"
    assert response.results[4].metrics.status == "success"


def test_benchmark_response_schema_validation() -> None:
    response = BenchmarkResponse.model_validate(
        {
            "results": [
                {
                    "strategy_name": "Zero-Shot",
                    "prompt": "Task:\nExplain Docker",
                    "response": "Docker packages applications into containers.",
                    "metrics": {
                        "latency_ms": 12,
                        "response_length": 45,
                        "word_count": 5,
                        "status": "success",
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                    },
                }
            ]
        }
    )

    assert response.results[0].metrics.status == "success"


def test_benchmark_endpoint_returns_structured_results(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generate_content_async(self: object, prompt: str) -> GeminiGeneration:
        return GeminiGeneration(text="Endpoint response")

    monkeypatch.setattr(
        "backend.gemini_client.GeminiClient.generate_content_async",
        fake_generate_content_async,
    )
    client = TestClient(app)

    response = client.post("/benchmark", json={"user_input": "Explain Docker"})

    assert response.status_code == 200
    body = response.json()
    assert {"run_id", "created_at", "user_input", "results"} <= set(body.keys())
    assert len(body["results"]) == 5
    assert body["user_input"] == "Explain Docker"
    assert body["results"][0]["metrics"]["status"] == "success"
    assert body["results"][0]["metrics"]["input_tokens"] is None


def test_benchmark_endpoint_rejects_empty_input() -> None:
    client = TestClient(app)

    response = client.post("/benchmark", json={"user_input": "   "})

    assert response.status_code == 422
