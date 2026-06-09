from dataclasses import dataclass

from backend.config import settings


class GeminiClientError(RuntimeError):
    """Raised when Gemini client configuration or calls fail."""


@dataclass(slots=True)
class GeminiClient:
    api_key: str | None = settings.gemini_api_key
    model_name: str = "gemini-pro"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def ensure_configured(self) -> None:
        if not self.is_configured:
            raise GeminiClientError("GEMINI_API_KEY is not configured.")

    def generate_response(self, prompt: str) -> str:
        """Future integration point for live Gemini API calls."""
        self.ensure_configured()
        raise NotImplementedError("Live Gemini API calls will be implemented in Phase 2.")
