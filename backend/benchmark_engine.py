import logging
import time

from backend.database import create_benchmark_run
from backend.gemini_client import (
    GeminiAPIRequestError,
    GeminiAuthenticationError,
    GeminiClient,
    GeminiClientError,
    GeminiConfigurationError,
    GeminiInvalidRequestError,
    GeminiNetworkError,
    GeminiParsingError,
    GeminiRateLimitError,
    GeminiTimeoutError,
    GeminiUnexpectedError,
    GeminiValidationError,
)
from backend.metrics import BenchmarkStatus, format_benchmark_metrics
from backend.prompt_strategies import PromptVariant, generate_prompt_variants, list_strategy_names
from backend.schemas import (
    BenchmarkMetrics,
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkResult,
    BenchmarkRunResponse,
)

logger = logging.getLogger(__name__)


class BenchmarkEngine:
    """Sequentially generates prompts, calls Gemini, and assembles objective metrics."""

    def __init__(self, gemini_client: GeminiClient | None = None) -> None:
        self.gemini_client = gemini_client or GeminiClient()

    async def run(self, request: BenchmarkRequest) -> BenchmarkResponse:
        prompt_variants = self.prepare_prompts(request.user_input)
        results: list[BenchmarkResult] = []

        for variant in prompt_variants:
            results.append(await self._execute_variant(variant))

        response = BenchmarkResponse(results=results)
        stored_run = self.persist_results(request.user_input, response)
        return BenchmarkResponse(
            run_id=stored_run.run_id,
            created_at=stored_run.created_at,
            user_input=stored_run.user_input,
            results=stored_run.results,
        )

    def prepare_prompts(self, user_input: str) -> list[PromptVariant]:
        return generate_prompt_variants(user_input)

    async def _execute_variant(self, variant: PromptVariant) -> BenchmarkResult:
        start_time = time.perf_counter()
        response_text: str | None = None
        token_usage: dict[str, int | None] | None = None
        status: BenchmarkStatus = "success"
        error_type: str | None = None
        error_message: str | None = None

        try:
            self._validate_variant(variant)
            generation = await self.gemini_client.generate_content_async(variant.prompt)
            response_text = generation.text
            token_usage = generation.token_usage()
        except GeminiClientError as exc:
            status = self._status_from_gemini_error(exc)
            error_type = type(exc).__name__
            error_message = str(exc)
            self._log_variant_failure(variant, exc)
        except Exception as exc:
            status = "unexpected_error"
            error_type = type(exc).__name__
            error_message = "Unexpected benchmark execution error."
            self._log_variant_failure(variant, exc)

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
            error_type=error_type,
            error_message=error_message,
            metrics=BenchmarkMetrics(**metrics),
        )

    def _validate_variant(self, variant: PromptVariant) -> None:
        if variant.strategy_name not in list_strategy_names():
            raise GeminiValidationError(f"Unknown prompt strategy: {variant.strategy_name}")
        if not variant.prompt or not variant.prompt.strip():
            raise GeminiValidationError(f"Prompt is empty for strategy: {variant.strategy_name}")

    def _log_variant_failure(self, variant: PromptVariant, exc: Exception) -> None:
        logger.exception(
            "Benchmark strategy failed: strategy=%s exception_type=%s status_code=%s message=%s",
            variant.strategy_name,
            type(exc).__name__,
            getattr(exc, "status_code", None),
            str(exc),
        )

    def _status_from_gemini_error(self, exc: GeminiClientError) -> BenchmarkStatus:
        if isinstance(exc, GeminiTimeoutError):
            return "timeout"
        if isinstance(exc, GeminiRateLimitError):
            return "rate_limit"
        if isinstance(exc, GeminiInvalidRequestError):
            return "invalid_request"
        if isinstance(exc, GeminiAuthenticationError):
            return "authentication_error"
        if isinstance(exc, GeminiParsingError):
            return "parsing_error"
        if isinstance(exc, (GeminiConfigurationError, GeminiValidationError)):
            return "invalid_request"
        if isinstance(exc, (GeminiAPIRequestError, GeminiNetworkError)):
            return "api_error"
        if isinstance(exc, GeminiUnexpectedError):
            return "unexpected_error"
        return "unexpected_error"

    def persist_results(self, user_input: str, response: BenchmarkResponse) -> BenchmarkRunResponse:
        return create_benchmark_run(user_input, response)
