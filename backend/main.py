from fastapi import FastAPI

from backend.config import settings
from backend.database import init_db
from backend.schemas import HealthResponse

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
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
