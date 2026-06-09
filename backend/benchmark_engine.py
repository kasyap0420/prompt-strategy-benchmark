import time

from backend.gemini_client import (
    GeminiAPIRequestError,
    GeminiAuthenticationError,
    GeminiClient,
    GeminiClientError,
    GeminiConfigurationError,
    GeminiNetworkError,
    GeminiTimeoutError,
    GeminiUnexpectedError,
    GeminiValidationError,
)
from backend.metrics import BenchmarkStatus, format_benchmark_metrics
from backend.prompt_strategies import PromptVariant, generate_prompt_variants
from backend.schemas import BenchmarkMetrics, BenchmarkRequest, BenchmarkResponse, BenchmarkResult


class BenchmarkEngine:
    """Sequentially generates prompts, calls Gemini, and assembles objective metrics."""

    def __init__(self, gemini_client: GeminiClient | None = None) -> None:
        self.gemini_client = gemini_client or GeminiClient()

    async def run(self, request: BenchmarkRequest) -> BenchmarkResponse:
        prompt_variants = self.prepare_prompts(request.user_input)
        results: list[BenchmarkResult] = []

        for variant in prompt_variants:
            results.append(await self._execute_variant(variant))

        return BenchmarkResponse(results=results)

    def prepare_prompts(self, user_input: str) -> list[PromptVariant]:
        return generate_prompt_variants(user_input)

    async def _execute_variant(self, variant: PromptVariant) -> BenchmarkResult:
        start_time = time.perf_counter()
        response_text: str | None = None
        token_usage: dict[str, int | None] | None = None
        status: BenchmarkStatus = "success"

        try:
            generation = await self.gemini_client.generate_content_async(variant.prompt)
            response_text = generation.text
            token_usage = generation.token_usage()
        except GeminiClientError as exc:
            status = self._status_from_gemini_error(exc)
        except Exception:
            status = "unexpected_error"

        end_time = time.perf_counter()
        metrics = format_benchmark_metrics(
            start_time=start_time,
            end_time=end_time,
            response_text=response_text,
            status=status,
            token_usage=token_usage,
        )

        return BenchmarkResult(
            strategy_name=variant.strategy_name,
            prompt=variant.prompt,
            response=response_text,
            metrics=BenchmarkMetrics(**metrics),
        )

    def _status_from_gemini_error(self, exc: GeminiClientError) -> BenchmarkStatus:
        if isinstance(exc, GeminiTimeoutError):
            return "timeout"
        if isinstance(exc, (GeminiConfigurationError, GeminiValidationError)):
            return "validation_error"
        if isinstance(
            exc,
            (
                GeminiAPIRequestError,
                GeminiAuthenticationError,
                GeminiNetworkError,
            ),
        ):
            return "api_error"
        if isinstance(exc, GeminiUnexpectedError):
            return "unexpected_error"
        return "unexpected_error"

    def persist_results(self, response: BenchmarkResponse) -> None:
        raise NotImplementedError("Benchmark persistence is outside the Phase 4/5 scope.")