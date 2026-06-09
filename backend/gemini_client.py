import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from backend.config import settings

logger = logging.getLogger(__name__)


class GeminiClientError(RuntimeError):
    """Base error for Gemini client failures safe to return to API callers."""

    status_code: int | None = None
    retryable: bool = False

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GeminiConfigurationError(GeminiClientError):
    """Raised when required Gemini configuration is missing or invalid."""


class GeminiValidationError(GeminiClientError):
    """Raised when a Gemini request is invalid before it reaches the API."""


class GeminiAuthenticationError(GeminiClientError):
    """Raised when Gemini rejects the configured API key or permissions."""


class GeminiTimeoutError(GeminiClientError):
    """Raised when a Gemini request exceeds the configured timeout."""

    retryable = True


class GeminiRateLimitError(GeminiClientError):
    """Raised when Gemini rate-limits the request."""

    retryable = True


class GeminiInvalidRequestError(GeminiClientError):
    """Raised when Gemini rejects the request payload."""


class GeminiNetworkError(GeminiClientError):
    """Raised when the Gemini API cannot be reached."""

    retryable = True


class GeminiAPIRequestError(GeminiClientError):
    """Raised when Gemini returns an API-level error."""


class GeminiParsingError(GeminiClientError):
    """Raised when a Gemini response cannot be parsed safely."""


class GeminiUnexpectedError(GeminiClientError):
    """Raised when an unexpected Gemini integration error occurs."""


@dataclass(frozen=True, slots=True)
class GeminiGeneration:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def token_usage(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class GeminiClient:
    api_key: str | None = settings.gemini_api_key
    model_name: str = settings.gemini_model
    api_version: str = settings.gemini_api_version
    timeout_seconds: float = settings.gemini_timeout_seconds
    max_retries: int = 2
    retry_base_delay_seconds: float = 0.5

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def ensure_configured(self) -> None:
        if not self.is_configured:
            raise GeminiConfigurationError("GEMINI_API_KEY is not configured.")

    def generate_response(self, prompt: str) -> str:
        """Generate content with the synchronous Gemini SDK client."""
        return self.generate_content(prompt).text

    def generate_content(self, prompt: str) -> GeminiGeneration:
        """Generate content and return text plus real provider token metadata when available."""
        cleaned_prompt = self._validate_prompt(prompt)
        return self._generate_content_with_retries(cleaned_prompt)

    async def generate_response_async(self, prompt: str) -> str:
        """Generate content with a wall-clock timeout around the async SDK call."""
        generation = await self.generate_content_async(prompt)
        return generation.text

    async def generate_content_async(self, prompt: str) -> GeminiGeneration:
        """Generate content and return text plus real provider token metadata when available."""
        cleaned_prompt = self._validate_prompt(prompt)
        return await self._generate_content_with_retries_async(cleaned_prompt)

    def _generate_content_with_retries(self, cleaned_prompt: str) -> GeminiGeneration:
        attempts = self.max_retries + 1
        for attempt_index in range(attempts):
            try:
                return self._generate_content_once(cleaned_prompt)
            except GeminiClientError as exc:
                if not self._should_retry(exc, attempt_index):
                    raise
                self._log_retry(exc, attempt_index)
                time_to_sleep = self._retry_delay(attempt_index)
                if time_to_sleep > 0:
                    import time

                    time.sleep(time_to_sleep)

        raise GeminiUnexpectedError("Gemini retry loop exited unexpectedly.")

    async def _generate_content_with_retries_async(self, cleaned_prompt: str) -> GeminiGeneration:
        attempts = self.max_retries + 1
        for attempt_index in range(attempts):
            try:
                return await self._generate_content_once_async(cleaned_prompt)
            except GeminiClientError as exc:
                if not self._should_retry(exc, attempt_index):
                    raise
                self._log_retry(exc, attempt_index)
                delay = self._retry_delay(attempt_index)
                if delay > 0:
                    await asyncio.sleep(delay)

        raise GeminiUnexpectedError("Gemini retry loop exited unexpectedly.")

    def _generate_content_once(self, cleaned_prompt: str) -> GeminiGeneration:
        client = self._create_client()
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=cleaned_prompt,
            )
            return self._extract_generation(response)
        except Exception as exc:
            self._raise_client_error(exc)
        finally:
            with suppress(Exception):
                client.close()

    async def _generate_content_once_async(self, cleaned_prompt: str) -> GeminiGeneration:
        client = self._create_client()
        async_client = client.aio
        try:
            response = await asyncio.wait_for(
                async_client.models.generate_content(
                    model=self.model_name,
                    contents=cleaned_prompt,
                ),
                timeout=self.timeout_seconds,
            )
            return self._extract_generation(response)
        except asyncio.TimeoutError as exc:
            raise GeminiTimeoutError(
                f"Gemini request timed out after {self.timeout_seconds:g} seconds."
            ) from exc
        except Exception as exc:
            self._raise_client_error(exc)
        finally:
            with suppress(Exception):
                await async_client.aclose()

    def _should_retry(self, exc: GeminiClientError, attempt_index: int) -> bool:
        return exc.retryable and attempt_index < self.max_retries

    def _retry_delay(self, attempt_index: int) -> float:
        return self.retry_base_delay_seconds * (2**attempt_index)

    def _log_retry(self, exc: GeminiClientError, attempt_index: int) -> None:
        logger.warning(
            "Retrying Gemini request after %s on attempt %s/%s: status_code=%s message=%s",
            type(exc).__name__,
            attempt_index + 1,
            self.max_retries + 1,
            exc.status_code,
            str(exc),
        )

    def _create_client(self) -> genai.Client:
        self.ensure_configured()
        api_key = self.api_key.strip() if self.api_key else ""
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                api_version=self.api_version,
                timeout=int(self.timeout_seconds * 1000),
            ),
        )

    def _validate_prompt(self, prompt: str) -> str:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise GeminiValidationError("Prompt cannot be empty.")
        return cleaned_prompt

    def _extract_generation(self, response: Any) -> GeminiGeneration:
        text = self._extract_text(response)
        usage_metadata = getattr(response, "usage_metadata", None)
        return GeminiGeneration(
            text=text,
            input_tokens=self._read_token_count(usage_metadata, "prompt_token_count"),
            output_tokens=self._read_token_count(usage_metadata, "candidates_token_count"),
            total_tokens=self._read_token_count(usage_metadata, "total_token_count"),
        )

    def _extract_text(self, response: Any) -> str:
        try:
            text = getattr(response, "text", None)
        except Exception as exc:
            raise GeminiParsingError("Gemini response text could not be read.") from exc
        if not text:
            raise GeminiParsingError("Gemini returned an empty response.")
        return str(text)

    def _read_token_count(self, usage_metadata: Any, field_name: str) -> int | None:
        value = getattr(usage_metadata, field_name, None)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value if value >= 0 else None

    def _raise_client_error(self, exc: Exception) -> None:
        if isinstance(exc, GeminiClientError):
            raise exc
        if isinstance(exc, errors.APIError):
            raise self._map_api_error(exc) from exc
        if isinstance(exc, httpx.TimeoutException):
            raise GeminiTimeoutError(
                f"Gemini request timed out after {self.timeout_seconds:g} seconds."
            ) from exc
        if isinstance(exc, httpx.RequestError | ConnectionError | OSError):
            raise GeminiNetworkError("Network error while contacting the Gemini API.") from exc
        raise GeminiUnexpectedError("Unexpected error while contacting the Gemini API.") from exc

    def _map_api_error(self, exc: errors.APIError) -> GeminiClientError:
        code = getattr(exc, "code", None) or getattr(exc, "status", None)
        message = self._safe_error_message(getattr(exc, "message", str(exc)))
        lower_message = message.lower()

        if code in {401, 403} or (code == 400 and "api key" in lower_message):
            return GeminiAuthenticationError(
                "Gemini API key is invalid or not authorized.",
                status_code=code,
            )
        if code == 429:
            return GeminiRateLimitError(
                f"Gemini rate limit exceeded: {message}",
                status_code=code,
            )
        if code == 400:
            return GeminiInvalidRequestError(
                f"Gemini invalid request: {message}",
                status_code=code,
            )
        if code and 500 <= code <= 599:
            error = GeminiAPIRequestError(f"Gemini API error ({code}): {message}", status_code=code)
            error.retryable = True
            return error
        if code:
            return GeminiAPIRequestError(f"Gemini API error ({code}): {message}", status_code=code)
        return GeminiAPIRequestError(f"Gemini API error: {message}")

    def _safe_error_message(self, message: str) -> str:
        cleaned = " ".join(message.split())
        if self.api_key:
            cleaned = cleaned.replace(self.api_key, "[redacted]")
        return cleaned
