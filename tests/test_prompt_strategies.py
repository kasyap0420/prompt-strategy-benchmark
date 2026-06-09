import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.prompt_strategies import (
    build_prompt,
    generate_prompt_variants,
    list_strategy_names,
)


EXPECTED_STRATEGY_NAMES = [
    "Zero-Shot",
    "Role Prompting",
    "Structured Prompting",
    "Expert Prompting",
    "Reasoning-Oriented Prompting",
]


def test_list_strategy_names_returns_phase_three_strategies() -> None:
    assert list_strategy_names() == EXPECTED_STRATEGY_NAMES


def test_generate_prompt_variants_uses_user_input() -> None:
    user_input = "Explain Docker in simple terms"

    variants = generate_prompt_variants(user_input)

    assert [variant.strategy_name for variant in variants] == EXPECTED_STRATEGY_NAMES
    assert len({variant.prompt for variant in variants}) == len(EXPECTED_STRATEGY_NAMES)
    assert all(user_input in variant.prompt for variant in variants)


def test_build_prompt_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown prompt strategy"):
        build_prompt("Unknown Strategy", "Explain Docker")


def test_generate_prompt_variants_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="User input cannot be empty"):
        generate_prompt_variants("   ")


def test_generate_strategies_endpoint_returns_generated_prompts_only(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_if_gemini_is_called(*args: object, **kwargs: object) -> str:
        raise AssertionError("Gemini should not be called by /generate-strategies")

    monkeypatch.setattr(
        "backend.gemini_client.GeminiClient.generate_response_async",
        fail_if_gemini_is_called,
    )
    client = TestClient(app)

    response = client.post(
        "/generate-strategies",
        json={"user_input": "Explain Docker in simple terms"},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"prompts"}
    assert [item["strategy_name"] for item in body["prompts"]] == EXPECTED_STRATEGY_NAMES
    assert all("prompt" in item for item in body["prompts"])


def test_generate_strategies_endpoint_rejects_empty_input() -> None:
    client = TestClient(app)

    response = client.post("/generate-strategies", json={"user_input": "   "})

    assert response.status_code == 422