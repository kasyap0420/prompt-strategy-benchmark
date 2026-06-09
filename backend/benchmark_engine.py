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
    BenchmarkRankingItem,
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkResult,
    BenchmarkRunResponse,
    BenchmarkWinners,
)

logger = logging.getLogger(__name__)


def calculate_token_efficiency(result: BenchmarkResult) -> float | None:
    input_tokens = result.metrics.input_tokens
    output_tokens = result.metrics.output_tokens
    if not input_tokens or output_tokens is None:
        return None
    return output_tokens / input_tokens


def _token_efficiency_sort_value(result: BenchmarkResult) -> float:
    return calculate_token_efficiency(result) or 0.0


def rank_benchmark_results(results: list[BenchmarkResult]) -> list[BenchmarkRankingItem]:
    sorted_results = sorted(
        results,
        key=lambda result: (
            result.metrics.status != "success",
            result.metrics.latency_ms,
            -result.metrics.response_length,
            -result.metrics.word_count,
            -_token_efficiency_sort_value(result),
            result.strategy_name,
        ),
    )

    return [
        BenchmarkRankingItem(
            rank=index,
            strategy_name=result.strategy_name,
            status=result.metrics.status,
            latency_ms=result.metrics.latency_ms,
            response_length=result.metrics.response_length,
            word_count=result.metrics.word_count,
            total_tokens=result.metrics.total_tokens,
            token_efficiency=calculate_token_efficiency(result),
        )
        for index, result in enumerate(sorted_results, start=1)
    ]


def detect_benchmark_winners(
    results: list[BenchmarkResult],
    ranking: list[BenchmarkRankingItem],
) -> BenchmarkWinners:
    successful_results = [result for result in results if result.metrics.status == "success"]
    if not successful_results:
        return BenchmarkWinners()

    fastest = min(
        successful_results,
        key=lambda result: (result.metrics.latency_ms, result.strategy_name),
    )
    most_detailed = max(
        successful_results,
        key=lambda result: (
            result.metrics.word_count,
            result.metrics.response_length,
            -result.metrics.latency_ms,
            result.strategy_name,
        ),
    )

    with_token_efficiency = [
        result for result in successful_results if calculate_token_efficiency(result) is not None
    ]
    if with_token_efficiency:
        most_efficient = max(
            with_token_efficiency,
            key=lambda result: (
                calculate_token_efficiency(result) or 0.0,
                -result.metrics.latency_ms,
                result.strategy_name,
            ),
        )
    else:
        most_efficient = min(
            successful_results,
            key=lambda result: (
                result.metrics.response_length,
                result.metrics.word_count,
                result.metrics.latency_ms,
                result.strategy_name,
            ),
        )

    overall_winner = next(
        (item.strategy_name for item in ranking if item.status == "success"),
        None,
    )

    return BenchmarkWinners(
        overall_winner=overall_winner,
        fastest_strategy=fastest.strategy_name,
        most_detailed_strategy=most_detailed.strategy_name,
        most_token_efficient_strategy=most_efficient.strategy_name,
    )


def generate_benchmark_summary(winners: BenchmarkWinners) -> str:
    if not winners.overall_winner:
        return "No strategy completed successfully. Review the error details for each strategy."

    return (
        f"{winners.most_detailed_strategy} produced the most detailed answer.\n"
        f"{winners.fastest_strategy} was the fastest.\n"
        f"{winners.most_token_efficient_strategy} was the most token-efficient.\n"
        f"Overall winner: {winners.overall_winner}."
    )


class BenchmarkEngine:
    """Sequentially generates prompts, calls Gemini, and assembles objective metrics."""

    def __init__(self, gemini_client: GeminiClient | None = None) -> None:
        self.gemini_client = gemini_client or GeminiClient()

    async def run(self, request: BenchmarkRequest) -> BenchmarkResponse:
        prompt_variants = self.prepare_prompts(request.user_input)
        results: list[BenchmarkResult] = []

        for variant in prompt_variants:
            results.append(await self._execute_variant(variant))

        ranking = rank_benchmark_results(results)
        winners = detect_benchmark_winners(results, ranking)
        benchmark_summary = generate_benchmark_summary(winners)
        response = BenchmarkResponse(
            results=results,
            ranking=ranking,
            winners=winners,
            benchmark_summary=benchmark_summary,
        )
        stored_run = self.persist_results(request.user_input, response)
        return BenchmarkResponse(
            run_id=stored_run.run_id,
            created_at=stored_run.created_at,
            user_input=stored_run.user_input,
            results=stored_run.results,
            ranking=stored_run.ranking,
            winners=stored_run.winners,
            benchmark_summary=stored_run.benchmark_summary,
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
