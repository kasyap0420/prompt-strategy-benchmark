from typing import Any

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    app_name: str


class GeminiTestRequest(BaseModel):
    prompt: str = Field(..., description="Prompt sent to Gemini for connectivity testing.")


class GeminiTestResponse(BaseModel):
    success: bool
    response: str | None = None
    error: str | None = None


class GenerateStrategiesRequest(BaseModel):
    user_input: str = Field(..., description="User input used to generate prompt strategy variants.")

    @field_validator("user_input")
    @classmethod
    def validate_user_input(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("User input cannot be empty.")
        return cleaned_value


class GeneratedPrompt(BaseModel):
    strategy_name: str
    prompt: str


class GenerateStrategiesResponse(BaseModel):
    prompts: list[GeneratedPrompt]


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