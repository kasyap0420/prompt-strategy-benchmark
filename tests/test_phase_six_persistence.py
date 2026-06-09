import csv
import io

import pytest
from fastapi.testclient import TestClient

from backend.database import get_connection
from backend.gemini_client import GeminiGeneration
from backend.main import app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    async def fake_generate_content_async(self: object, prompt: str) -> GeminiGeneration:
        return GeminiGeneration(
            text=f"Saved response for: {prompt}",
            input_tokens=4,
            output_tokens=8,
            total_tokens=12,
        )

    monkeypatch.setattr(
        "backend.gemini_client.GeminiClient.generate_content_async",
        fake_generate_content_async,
    )
    return TestClient(app)


def create_saved_benchmark(client: TestClient, user_input: str = "Explain Docker") -> dict:
    response = client.post("/benchmark", json={"user_input": user_input})
    assert response.status_code == 200
    return response.json()


def test_benchmark_persists_run_and_results(client: TestClient) -> None:
    body = create_saved_benchmark(client)

    with get_connection() as connection:
        run_count = connection.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0]
        result_count = connection.execute("SELECT COUNT(*) FROM benchmark_results").fetchone()[0]
        first_result = connection.execute(
            """
            SELECT strategy_name, status, input_tokens, output_tokens, total_tokens
            FROM benchmark_results
            WHERE run_id = ?
            ORDER BY id ASC
            LIMIT 1
            """,
            (body["run_id"],),
        ).fetchone()

    assert run_count == 1
    assert result_count == 5
    assert first_result["strategy_name"] == "Zero-Shot"
    assert first_result["status"] == "success"
    assert first_result["input_tokens"] == 4
    assert first_result["output_tokens"] == 8
    assert first_result["total_tokens"] == 12


def test_benchmark_history_endpoint_returns_saved_runs(client: TestClient) -> None:
    first = create_saved_benchmark(client, "Explain Docker")
    second = create_saved_benchmark(client, "Explain Kubernetes")

    response = client.get("/benchmark/history")

    assert response.status_code == 200
    runs = response.json()["runs"]
    assert [run["run_id"] for run in runs] == [second["run_id"], first["run_id"]]
    assert runs[0]["user_input"] == "Explain Kubernetes"
    assert runs[0]["result_count"] == 5


def test_benchmark_retrieval_endpoint_returns_full_run(client: TestClient) -> None:
    saved_run = create_saved_benchmark(client)

    response = client.get(f"/benchmark/{saved_run['run_id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == saved_run["run_id"]
    assert body["user_input"] == "Explain Docker"
    assert len(body["results"]) == 5
    assert body["results"][0]["response"].startswith("Saved response for:")


def test_benchmark_retrieval_endpoint_returns_404_for_missing_run(client: TestClient) -> None:
    response = client.get("/benchmark/missing-run-id")

    assert response.status_code == 404


def test_json_export_returns_saved_run(client: TestClient) -> None:
    saved_run = create_saved_benchmark(client)

    response = client.get(f"/benchmark/{saved_run['run_id']}/export/json")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "attachment" in response.headers["content-disposition"]
    body = response.json()
    assert body["run_id"] == saved_run["run_id"]
    assert len(body["results"]) == 5


def test_csv_export_returns_flattened_results(client: TestClient) -> None:
    saved_run = create_saved_benchmark(client)

    response = client.get(f"/benchmark/{saved_run['run_id']}/export/csv")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]

    lines = response.text.splitlines()
    result_header_index = lines.index(
        "run_id,created_at,user_input,strategy_name,prompt,response,error_type,error_message,status,"
        "latency_ms,response_length,word_count,input_tokens,output_tokens,total_tokens"
    )
    result_csv = "\n".join(lines[result_header_index:])
    rows = list(csv.DictReader(io.StringIO(result_csv)))
    assert len(rows) == 5
    assert rows[0]["run_id"] == saved_run["run_id"]
    assert rows[0]["strategy_name"] == "Zero-Shot"
    assert rows[0]["status"] == "success"
    assert rows[0]["input_tokens"] == "4"
