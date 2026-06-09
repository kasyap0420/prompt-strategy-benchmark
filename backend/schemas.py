from typing import Literal

from pydantic import BaseModel, Field, field_validator

BenchmarkStatus = Literal[
    "success",
    "timeout",
    "api_error",
    "validation_error",
    "unexpected_error",
]


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
    user_input: str = Field(..., description="Prompt or task to benchmark.")

    @field_validator("user_input")
    @classmethod
    def validate_user_input(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("User input cannot be empty.")
        return cleaned_value


class BenchmarkMetrics(BaseModel):
    latency_ms: int = Field(..., ge=0)
    response_length: int = Field(..., ge=0)
    word_count: int = Field(..., ge=0)
    status: BenchmarkStatus
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class BenchmarkResult(BaseModel):
    strategy_name: str
    prompt: str
    response: str | None = None
    metrics: BenchmarkMetrics


class BenchmarkResponse(BaseModel):
    results: list[BenchmarkResult]