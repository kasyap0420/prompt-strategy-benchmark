from typing import Any


def calculate_latency_ms(start_time: float, end_time: float) -> float:
    return round((end_time - start_time) * 1000, 3)


def extract_token_usage(response_metadata: dict[str, Any]) -> dict[str, int] | None:
    """Return token usage from provider metadata when available."""
    token_usage = response_metadata.get("token_usage")
    return token_usage if isinstance(token_usage, dict) else None


def check_format_compliance(response_text: str, required_sections: list[str]) -> bool | None:
    if not required_sections:
        return None
    return all(section.lower() in response_text.lower() for section in required_sections)


def check_constraint_satisfaction(response_text: str, constraints: list[str]) -> bool | None:
    if not constraints:
        return None
    raise NotImplementedError("Constraint satisfaction checks will be implemented in Phase 2.")
