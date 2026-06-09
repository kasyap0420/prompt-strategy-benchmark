from fastapi import FastAPI

from backend.benchmark_engine import BenchmarkEngine
from backend.config import settings
from backend.database import init_db
from backend.gemini_client import GeminiClient, GeminiClientError
from backend.prompt_strategies import generate_prompt_variants
from backend.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    GenerateStrategiesRequest,
    GenerateStrategiesResponse,
    GeneratedPrompt,
    GeminiTestRequest,
    GeminiTestResponse,
    HealthResponse,
)

app = FastAPI(
    title=settings.app_name,
    version="0.5.0",
    description="API foundation for the Prompt Strategy Benchmark application.",
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
        return GeminiTestResponse(success=False, error=str(exc))
    except Exception:
        return GeminiTestResponse(
            success=False,
            error="Unexpected error while testing Gemini connectivity.",
        )