import logging

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.benchmark_engine import BenchmarkEngine
from backend.config import settings
from backend.database import (
    consume_daily_budget,
    get_analytics_history,
    get_analytics_summary,
    get_benchmark_run,
    get_cached_benchmark_run,
    get_daily_budget_usage,
    get_strategy_performance,
    init_db,
    list_benchmark_runs,
)
from backend.export_service import benchmark_run_to_csv, benchmark_run_to_json
from backend.gemini_client import GeminiClient, GeminiClientError
from backend.prompt_strategies import generate_prompt_variants
from backend.schemas import (
    AnalyticsHistoryResponse,
    AnalyticsSummaryResponse,
    BenchmarkHistoryResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkRunResponse,
    GenerateStrategiesRequest,
    GenerateStrategiesResponse,
    GeneratedPrompt,
    GeminiTestRequest,
    GeminiTestResponse,
    HealthResponse,
    StrategyPerformanceResponse,
)

logger = logging.getLogger(__name__)


def get_rate_limit_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",", maxsplit=1)[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=get_rate_limit_key)


app = FastAPI(
    title=settings.app_name,
    version="0.6.0",
    description="End-to-end API for benchmarking prompt strategies.",
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please wait before trying again.",
            "error": str(exc),
        },
    )


@app.on_event("startup")
def on_startup() -> None:
    """Prepare local application resources required at startup."""
    init_db()


@app.get("/", tags=["Root"])
def read_root() -> dict[str, str]:
    return {
        "app": settings.app_name,
        "environment": settings.app_env,
        "status": "ready",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", app_name=settings.app_name)


@app.get(
    "/analytics/summary",
    response_model=AnalyticsSummaryResponse,
    tags=["Analytics"],
)
def analytics_summary() -> AnalyticsSummaryResponse:
    return get_analytics_summary()


@app.get(
    "/analytics/strategy-performance",
    response_model=StrategyPerformanceResponse,
    tags=["Analytics"],
)
def analytics_strategy_performance() -> StrategyPerformanceResponse:
    return get_strategy_performance()


@app.get(
    "/analytics/history",
    response_model=AnalyticsHistoryResponse,
    tags=["Analytics"],
)
def analytics_history() -> AnalyticsHistoryResponse:
    return get_analytics_history()


@app.post(
    "/generate-strategies",
    response_model=GenerateStrategiesResponse,
    tags=["Prompt Strategies"],
)
def generate_strategies(request: GenerateStrategiesRequest) -> GenerateStrategiesResponse:
    prompt_variants = generate_prompt_variants(request.user_input)
    return GenerateStrategiesResponse(
        prompts=[
            GeneratedPrompt(strategy_name=variant.strategy_name, prompt=variant.prompt)
            for variant in prompt_variants
        ]
    )


@app.post("/benchmark", response_model=BenchmarkResponse, tags=["Benchmark"])
@limiter.limit("3/minute")
async def run_benchmark(request: Request, benchmark_request: BenchmarkRequest) -> BenchmarkResponse:
    cached_run = get_cached_benchmark_run(benchmark_request.user_input)
    current_budget = get_daily_budget_usage(settings.benchmark_daily_limit)
    if cached_run is not None:
        return BenchmarkResponse(
            run_id=cached_run.run_id,
            created_at=cached_run.created_at,
            user_input=cached_run.user_input,
            results=cached_run.results,
            ranking=cached_run.ranking,
            winners=cached_run.winners,
            benchmark_summary=cached_run.benchmark_summary,
            cached=True,
            daily_budget=current_budget,
        )

    daily_budget = consume_daily_budget(settings.benchmark_daily_limit)
    if daily_budget is None:
        budget = get_daily_budget_usage(settings.benchmark_daily_limit)
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Daily benchmark request budget reached. Try again tomorrow.",
                "daily_budget": budget.model_dump(),
            },
        )

    engine = BenchmarkEngine()
    response = await engine.run(benchmark_request)
    response.daily_budget = daily_budget
    return response


@app.get(
    "/benchmark/history",
    response_model=BenchmarkHistoryResponse,
    tags=["Benchmark"],
)
def benchmark_history() -> BenchmarkHistoryResponse:
    return BenchmarkHistoryResponse(runs=list_benchmark_runs())


@app.get(
    "/benchmark/{run_id}",
    response_model=BenchmarkRunResponse,
    tags=["Benchmark"],
)
def get_benchmark(run_id: str) -> BenchmarkRunResponse:
    run = get_benchmark_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Benchmark run not found.")
    return run


@app.get("/benchmark/{run_id}/export/json", tags=["Benchmark"])
def export_benchmark_json(run_id: str) -> Response:
    run = get_benchmark_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Benchmark run not found.")
    return Response(
        content=benchmark_run_to_json(run),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'},
    )


@app.get("/benchmark/{run_id}/export/csv", tags=["Benchmark"])
def export_benchmark_csv(run_id: str) -> Response:
    run = get_benchmark_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Benchmark run not found.")
    return Response(
        content=benchmark_run_to_csv(run),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'},
    )


@app.post(
    "/test-gemini",
    response_model=GeminiTestResponse,
    response_model_exclude_none=True,
    tags=["Gemini"],
)
@limiter.limit("10/minute")
async def test_gemini(request: Request, gemini_request: GeminiTestRequest) -> GeminiTestResponse:
    client = GeminiClient()
    try:
        response_text = await client.generate_response_async(gemini_request.prompt)
        return GeminiTestResponse(success=True, response=response_text)
    except GeminiClientError as exc:
        logger.exception(
            "Gemini connectivity test failed: exception_type=%s status_code=%s message=%s",
            type(exc).__name__,
            getattr(exc, "status_code", None),
            str(exc),
        )
        return GeminiTestResponse(success=False, error=str(exc))
    except Exception as exc:
        logger.exception("Unexpected Gemini connectivity test failure: %s", exc)
        return GeminiTestResponse(
            success=False,
            error="Unexpected error while testing Gemini connectivity.",
        )
