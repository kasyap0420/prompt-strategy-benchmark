from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str


class BenchmarkRequest(BaseModel):
    user_input: str = Field(..., min_length=1, description="Prompt or task to benchmark.")
    strategies: list[str] | None = Field(
        default=None,
        description="Optional list of prompt strategies to evaluate later.",
    )


class BenchmarkResult(BaseModel):
    strategy_name: str
    prompt: str
    response_text: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResponse(BaseModel):
    run_id: str
    status: str
    results: list[BenchmarkResult] = Field(default_factory=list)
