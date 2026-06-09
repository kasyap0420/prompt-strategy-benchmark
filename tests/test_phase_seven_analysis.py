import asyncio

import pytest
from fastapi.testclient import TestClient

from backend.benchmark_engine import (
    detect_benchmark_winners,
    generate_benchmark_summary,
    rank_benchmark_results,
)
from backend.gemini_client import GeminiGeneration
from backend.main import app
from backend.schemas import BenchmarkMetrics, BenchmarkRequest, BenchmarkResult
from backend.benchmark_engine import BenchmarkEngine


def make_result(
    strategy_name: str,
    *,
    status: str = "success",
    latency_ms: int = 100,
    response_length: int = 100,
    word_count: int = 20,
    input_tokens: int | None = 10,
    output_tokens: int | None = 20,
    total_tokens: int | None = 30,
) -> BenchmarkResult:
    return BenchmarkResult(
        strategy_name=strategy_name,
        prompt=f"Prompt for {strategy_name}",
        response="x" * response_length if status == "success" else None,
        metrics=BenchmarkMetrics(
            status=status,  # type: ignore[arg-type]
            latency_ms=latency_ms,
            response_length=response_length,
            word_count=word_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        ),
    )


def test_ranking_prioritizes_success_then_latency_then_detail_then_tokens() -> None:
    results = [
        make_result("Failed Fast", status="api_error", latency_ms=1),
        make_result("Detailed", latency_ms=50, response_length=500, word_count=80),
        make_result("Fastest", latency_ms=25, response_length=100, word_count=20),
        make_result("Efficient", latency_ms=50, response_length=500, word_count=80, output_tokens=30),
    ]

    ranking = rank_benchmark_results(results)

    assert [item.strategy_name for item in ranking] == [
        "Fastest",
        "Efficient",
        "Detailed",
        "Failed Fast",
    ]
    assert ranking[-1].status == "api_error"


def test_winner_detection_and_summary_generation() -> None:
    results = [
        make_result("Fastest", latency_ms=10, response_length=100, word_count=15),
        make_result("Detailed", latency_ms=20, response_length=400, word_count=90),
        make_result("Efficient", latency_ms=30, response_length=120, word_count=20, output_tokens=50),
    ]
    ranking = rank_benchmark_results(results)

    winners = detect_benchmark_winners(results, ranking)
    summary = generate_benchmark_summary(winners)

    assert winners.overall_winner == "Fastest"
    assert winners.fastest_strategy == "Fastest"
    assert winners.most_detailed_strategy == "Detailed"
    assert winners.most_token_efficient_strategy == "Efficient"
    assert "Overall winner: Fastest." in summary


class PhaseSevenGeminiClient:
    async def generate_content_async(self, prompt: str) -> GeminiGeneration:
        if prompt.startswith("Task:"):
            return GeminiGeneration(text="quick", input_tokens=10, output_tokens=5, total_tokens=15)
        if "clear structure" in prompt:
            return GeminiGeneration(
                text="one two three four five six seven eight nine ten",
                input_tokens=10,
                output_tokens=25,
                total_tokens=35,
            )
        return GeminiGeneration(text="medium response text", input_tokens=10, output_tokens=10, total_tokens=20)


def test_benchmark_response_includes_ranking_winners_and_summary() -> None:
    engine = BenchmarkEngine(gemini_client=PhaseSevenGeminiClient())  # type: ignore[arg-type]

    response = asyncio.run(engine.run(BenchmarkRequest(user_input="Explain indexes")))

    assert len(response.ranking) == 5
    assert response.winners.overall_winner is not None
    assert response.benchmark_summary


def test_persistence_retrieval_and_exports_include_phase_seven_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_generate_content_async(self: object, prompt: str) -> GeminiGeneration:
        if prompt.startswith("Task:"):
            return GeminiGeneration(text="quick", input_tokens=10, output_tokens=5, total_tokens=15)
        return GeminiGeneration(
            text="medium response text with enough detail",
            input_tokens=10,
            output_tokens=12,
            total_tokens=22,
        )

    monkeypatch.setattr(
        "backend.gemini_client.GeminiClient.generate_content_async",
        fake_generate_content_async,
    )
    client = TestClient(app)

    response = client.post("/benchmark", json={"user_input": "Explain indexes"})

    assert response.status_code == 200
    body = response.json()
    run_id = body["run_id"]
    assert body["ranking"]
    assert body["winners"]["overall_winner"]
    assert body["benchmark_summary"]

    retrieved = client.get(f"/benchmark/{run_id}").json()
    assert retrieved["ranking"] == body["ranking"]
    assert retrieved["winners"] == body["winners"]
    assert retrieved["benchmark_summary"] == body["benchmark_summary"]

    json_export = client.get(f"/benchmark/{run_id}/export/json").json()
    assert json_export["ranking"] == body["ranking"]
    assert json_export["winners"] == body["winners"]

    csv_export = client.get(f"/benchmark/{run_id}/export/csv").text
    assert "metadata_key,metadata_value" in csv_export
    assert "overall_winner" in csv_export
    assert "benchmark_summary" in csv_export
