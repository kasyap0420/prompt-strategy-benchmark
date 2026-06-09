from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - compatibility with Pydantic v1
    from pydantic import BaseSettings  # type: ignore[no-redef]

    SettingsConfigDict = None  # type: ignore[assignment]


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    app_name: str = Field(default="Prompt Strategy Benchmark", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_api_version: str = Field(default="v1", alias="GEMINI_API_VERSION")
    gemini_timeout_seconds: float = Field(default=30.0, alias="GEMINI_TIMEOUT_SECONDS")
    database_url: str = Field(default="sqlite:///benchmark.db", alias="DATABASE_URL")
    benchmark_daily_limit: int = Field(default=100, alias="BENCHMARK_DAILY_LIMIT")

    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")
    else:

        class Config:
            env_file = BASE_DIR / ".env"
            extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
