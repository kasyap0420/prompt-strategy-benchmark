import logging

from fastapi import FastAPI, HTTPException, Response

from backend.benchmark_engine import BenchmarkEngine
from backend.config import settings
from backend.database import get_benchmark_run, init_db, list_benchmark_runs
from backend.export_service import benchmark_run_to_csv, benchmark_run_to_json
from backend.gemini_client import GeminiClient, GeminiClientError
from backend.prompt_strategies import generate_prompt_variants
from backend.schemas import (
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
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="0.6.0",
    description="End-to-end API for benchmarking prompt strategies.",
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
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    engine = BenchmarkEngine()
    return await engine.run(request)


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
async def test_gemini(request: GeminiTestRequest) -> GeminiTestResponse:
    client = GeminiClient()
    try:
        response_text = await client.generate_response_async(request.prompt)
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
