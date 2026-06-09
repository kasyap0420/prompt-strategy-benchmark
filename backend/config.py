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
    database_url: str = Field(default="sqlite:///benchmark.db", alias="DATABASE_URL")

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
