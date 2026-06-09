import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.gemini_client import GeminiGeneration
from backend.main import app


def client_headers(client_id: int) -> dict[str, str]:
    return {"X-Forwarded-For": f"203.0.113.{client_id}"}


def test_benchmark_rate_limit_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generate_content_async(self: object, prompt: str) -> GeminiGeneration:
        return GeminiGeneration(text="ok")

    monkeypatch.setattr(
        "backend.gemini_client.GeminiClient.generate_content_async",
        fake_generate_content_async,
    )
    client = TestClient(app)

    statuses = [
        client.post(
            "/benchmark",
            json={"user_input": f"Prompt {index}"},
            headers=client_headers(1),
        ).status_code
        for index in range(4)
    ]

    assert statuses == [200, 200, 200, 429]


def test_test_gemini_rate_limit_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generate_response_async(self: object, prompt: str) -> str:
        return "ok"

    monkeypatch.setattr(
        "backend.gemini_client.GeminiClient.generate_response_async",
        fake_generate_response_async,
    )
    client = TestClient(app)

    statuses = [
        client.post(
            "/test-gemini",
            json={"prompt": f"Prompt {index}"},
            headers=client_headers(2),
        ).status_code
        for index in range(11)
    ]

    assert statuses[:10] == [200] * 10
    assert statuses[10] == 429


def test_cache_hit_returns_cached_result_without_gemini_call(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    async def fake_generate_content_async(self: object, prompt: str) -> GeminiGeneration:
        nonlocal call_count
        call_count += 1
        return GeminiGeneration(text="cached response")

    monkeypatch.setattr(
        "backend.gemini_client.GeminiClient.generate_content_async",
        fake_generate_content_async,
    )
    client = TestClient(app)

    first = client.post(
        "/benchmark",
        json={"user_input": "Repeatable prompt"},
        headers=client_headers(3),
    )
    second = client.post(
        "/benchmark",
        json={"user_input": "Repeatable prompt"},
        headers=client_headers(3),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert second.json()["run_id"] == first.json()["run_id"]
    assert call_count == 5


def test_cache_miss_executes_gemini_for_new_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    async def fake_generate_content_async(self: object, prompt: str) -> GeminiGeneration:
        nonlocal call_count
        call_count += 1
        return GeminiGeneration(text="fresh response")

    monkeypatch.setattr(
        "backend.gemini_client.GeminiClient.generate_content_async",
        fake_generate_content_async,
    )
    client = TestClient(app)

    first = client.post(
        "/benchmark",
        json={"user_input": "First unique prompt"},
        headers=client_headers(4),
    )
    second = client.post(
        "/benchmark",
        json={"user_input": "Second unique prompt"},
        headers=client_headers(4),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["cached"] is False
    assert second.json()["cached"] is False
    assert call_count == 10


def test_daily_budget_enforcement_blocks_new_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    async def fake_generate_content_async(self: object, prompt: str) -> GeminiGeneration:
        nonlocal call_count
        call_count += 1
        return GeminiGeneration(text="budgeted response")

    monkeypatch.setattr(settings, "benchmark_daily_limit", 1)
    monkeypatch.setattr(
        "backend.gemini_client.GeminiClient.generate_content_async",
        fake_generate_content_async,
    )
    client = TestClient(app)

    first = client.post(
        "/benchmark",
        json={"user_input": "Allowed prompt"},
        headers=client_headers(5),
    )
    second = client.post(
        "/benchmark",
        json={"user_input": "Blocked prompt"},
        headers=client_headers(5),
    )

    assert first.status_code == 200
    assert first.json()["daily_budget"]["used"] == 1
    assert second.status_code == 429
    assert second.json()["detail"]["message"] == "Daily benchmark request budget reached. Try again tomorrow."
    assert call_count == 5


def test_daily_budget_does_not_block_cached_result(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    async def fake_generate_content_async(self: object, prompt: str) -> GeminiGeneration:
        nonlocal call_count
        call_count += 1
        return GeminiGeneration(text="cached budget response")

    monkeypatch.setattr(settings, "benchmark_daily_limit", 1)
    monkeypatch.setattr(
        "backend.gemini_client.GeminiClient.generate_content_async",
        fake_generate_content_async,
    )
    client = TestClient(app)

    first = client.post(
        "/benchmark",
        json={"user_input": "Budget cache prompt"},
        headers=client_headers(6),
    )
    second = client.post(
        "/benchmark",
        json={"user_input": "Budget cache prompt"},
        headers=client_headers(6),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert second.json()["daily_budget"]["remaining"] == 0
    assert call_count == 5
