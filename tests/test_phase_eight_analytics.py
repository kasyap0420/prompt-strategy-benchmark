from fastapi.testclient import TestClient

from backend.database import create_benchmark_run, get_analytics_summary, get_strategy_performance
from backend.main import app
from backend.schemas import (
    BenchmarkMetrics,
    BenchmarkResponse,
    BenchmarkResult,
    BenchmarkWinners,
)


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


def seed_analytics_runs() -> None:
    create_benchmark_run(
        "First run",
        BenchmarkResponse(
            run_id="run-1",
            created_at="2026-06-09T10:00:00+00:00",
            results=[
                make_result("Zero-Shot", latency_ms=100, response_length=100, word_count=20),
                make_result(
                    "Role Prompting",
                    status="api_error",
                    latency_ms=200,
                    response_length=0,
                    word_count=0,
                    input_tokens=None,
                    output_tokens=None,
                    total_tokens=None,
                ),
            ],
            winners=BenchmarkWinners(overall_winner="Zero-Shot"),
            benchmark_summary="Overall winner: Zero-Shot.",
        ),
    )
    create_benchmark_run(
        "Second run",
        BenchmarkResponse(
            run_id="run-2",
            created_at="2026-06-09T11:00:00+00:00",
            results=[
                make_result(
                    "Zero-Shot",
                    latency_ms=300,
                    response_length=50,
                    word_count=10,
                    input_tokens=20,
                    output_tokens=10,
                    total_tokens=30,
                ),
                make_result(
                    "Role Prompting",
                    latency_ms=150,
                    response_length=200,
                    word_count=40,
                    input_tokens=10,
                    output_tokens=30,
                    total_tokens=40,
                ),
            ],
            winners=BenchmarkWinners(overall_winner="Role Prompting"),
            benchmark_summary="Overall winner: Role Prompting.",
        ),
    )


def test_empty_analytics_endpoints_return_valid_payloads() -> None:
    client = TestClient(app)

    summary = client.get("/analytics/summary")
    performance = client.get("/analytics/strategy-performance")
    history = client.get("/analytics/history")

    assert summary.status_code == 200
    assert summary.json() == {
        "total_runs": 0,
        "total_results": 0,
        "total_successes": 0,
        "total_failures": 0,
        "overall_success_rate": 0.0,
    }
    assert performance.status_code == 200
    assert performance.json() == {"strategies": []}
    assert history.status_code == 200
    assert history.json() == {"runs": []}


def test_analytics_summary_and_strategy_performance_aggregate_persisted_data() -> None:
    seed_analytics_runs()

    summary = get_analytics_summary()
    performance = get_strategy_performance()
    by_strategy = {item.strategy_name: item for item in performance.strategies}

    assert summary.total_runs == 2
    assert summary.total_results == 4
    assert summary.total_successes == 3
    assert summary.total_failures == 1
    assert summary.overall_success_rate == 0.75

    zero_shot = by_strategy["Zero-Shot"]
    assert zero_shot.total_runs == 2
    assert zero_shot.wins == 1
    assert zero_shot.win_rate == 0.5
    assert zero_shot.success_rate == 1.0
    assert zero_shot.avg_latency_ms == 200
    assert zero_shot.avg_response_length == 75
    assert zero_shot.avg_word_count == 15

    role_prompting = by_strategy["Role Prompting"]
    assert role_prompting.total_runs == 2
    assert role_prompting.wins == 1
    assert role_prompting.success_rate == 0.5
    assert role_prompting.failure_rate == 0.5
    assert role_prompting.avg_output_tokens == 30


def test_analytics_history_endpoint_returns_run_level_summaries() -> None:
    seed_analytics_runs()
    client = TestClient(app)

    response = client.get("/analytics/history")

    assert response.status_code == 200
    runs = response.json()["runs"]
    assert [run["run_id"] for run in runs] == ["run-1", "run-2"]
    assert runs[0]["total_results"] == 2
    assert runs[0]["successes"] == 1
    assert runs[0]["failures"] == 1
    assert runs[0]["success_rate"] == 0.5
    assert runs[1]["overall_winner"] == "Role Prompting"
