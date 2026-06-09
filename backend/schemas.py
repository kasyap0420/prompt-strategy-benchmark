from typing import Literal

from pydantic import BaseModel, Field, field_validator

BenchmarkStatus = Literal[
    "success",
    "timeout",
    "rate_limit",
    "invalid_request",
    "authentication_error",
    "api_error",
    "validation_error",
    "parsing_error",
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
    error_type: str | None = None
    error_message: str | None = None
    metrics: BenchmarkMetrics


class BenchmarkRankingItem(BaseModel):
    rank: int = Field(..., ge=1)
    strategy_name: str
    status: BenchmarkStatus
    latency_ms: int = Field(..., ge=0)
    response_length: int = Field(..., ge=0)
    word_count: int = Field(..., ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    token_efficiency: float | None = Field(default=None, ge=0)


class BenchmarkWinners(BaseModel):
    overall_winner: str | None = None
    fastest_strategy: str | None = None
    most_detailed_strategy: str | None = None
    most_token_efficient_strategy: str | None = None


class BenchmarkResponse(BaseModel):
    run_id: str | None = None
    created_at: str | None = None
    user_input: str | None = None
    results: list[BenchmarkResult]
    ranking: list[BenchmarkRankingItem] = Field(default_factory=list)
    winners: BenchmarkWinners = Field(default_factory=BenchmarkWinners)
    benchmark_summary: str | None = None


class BenchmarkHistoryItem(BaseModel):
    run_id: str
    created_at: str
    user_input: str
    result_count: int


class BenchmarkHistoryResponse(BaseModel):
    runs: list[BenchmarkHistoryItem]


class BenchmarkRunResponse(BaseModel):
    run_id: str
    created_at: str
    user_input: str
    results: list[BenchmarkResult]
    ranking: list[BenchmarkRankingItem] = Field(default_factory=list)
    winners: BenchmarkWinners = Field(default_factory=BenchmarkWinners)
    benchmark_summary: str | None = None


class AnalyticsSummaryResponse(BaseModel):
    total_runs: int = Field(..., ge=0)
    total_results: int = Field(..., ge=0)
    total_successes: int = Field(..., ge=0)
    total_failures: int = Field(..., ge=0)
    overall_success_rate: float = Field(..., ge=0)


class StrategyPerformanceItem(BaseModel):
    strategy_name: str
    total_runs: int = Field(..., ge=0)
    wins: int = Field(..., ge=0)
    win_rate: float = Field(..., ge=0)
    avg_latency_ms: float | None = None
    avg_response_length: float | None = None
    avg_word_count: float | None = None
    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    avg_total_tokens: float | None = None
    success_rate: float = Field(..., ge=0)
    failure_rate: float = Field(..., ge=0)


class StrategyPerformanceResponse(BaseModel):
    strategies: list[StrategyPerformanceItem]


class AnalyticsHistoryItem(BaseModel):
    run_id: str
    created_at: str
    user_input: str
    total_results: int = Field(..., ge=0)
    successes: int = Field(..., ge=0)
    failures: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0)
    overall_winner: str | None = None
    benchmark_summary: str | None = None


class AnalyticsHistoryResponse(BaseModel):
    runs: list[AnalyticsHistoryItem]
