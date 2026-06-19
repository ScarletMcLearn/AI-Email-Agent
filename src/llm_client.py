"""Hosted LLM adapters with retries, metadata, and provider fallback."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel

from src.config import Settings

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
SERVER_ERROR_STATUS = 500


class LLMError(RuntimeError):
    """Raised after an LLM operation exhausts its retries."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 1,
        fallback_allowed: bool = True,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.fallback_allowed = fallback_allowed


class DailyQuotaExceeded(LLMError):
    """Raised when the final provider's daily request quota is exhausted."""


@dataclass(frozen=True)
class CallMetadata:
    """Provider details for one successful LLM operation."""

    provider: str
    model: str
    attempts: int
    fallback_used: bool = False


@dataclass(frozen=True)
class LLMResponse(Generic[T]):
    """A generated value and the provider metadata that produced it."""

    value: T
    metadata: CallMetadata


@dataclass(frozen=True)
class RetryOutcome(Generic[T]):
    """Successful retry result with the number of attempts used."""

    value: T
    attempts: int


class LLMClient(Protocol):
    """Interface used by generation and evaluation."""

    provider_name: str

    def generate_text(
        self, prompt: str, *, model: str, temperature: float
    ) -> LLMResponse[str]:
        """Generate a non-empty text response."""

    def generate_structured(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        response_model: type[T],
        validator: Callable[[T], T] | None = None,
    ) -> LLMResponse[T]:
        """Generate, parse, and validate a structured response."""


def retry_call_with_attempts(
    operation: Callable[[], T],
    *,
    max_attempts: int,
    base_delay_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    operation_name: str = "LLM operation",
    retry_delay_fn: Callable[[Exception], float | None] | None = None,
    should_retry: Callable[[Exception], bool] | None = None,
) -> RetryOutcome[T]:
    """Run an operation with exponential backoff and return attempt metadata."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return RetryOutcome(operation(), attempt)
        except Exception as exc:  # API and parsing failures are classified below.
            last_error = exc
            if is_daily_quota_error(exc):
                raise DailyQuotaExceeded(
                    f"{operation_name} stopped because the model's daily free-tier "
                    f"request quota is exhausted: {exc}",
                    attempts=attempt,
                ) from exc
            retryable = should_retry(exc) if should_retry else True
            if not retryable:
                raise LLMError(
                    f"{operation_name} failed with a non-retryable provider error: "
                    f"{exc}",
                    attempts=attempt,
                    fallback_allowed=False,
                ) from exc
            if attempt == max_attempts:
                break
            exponential_delay = base_delay_seconds * (2 ** (attempt - 1))
            recommended_delay = retry_delay_fn(exc) if retry_delay_fn else None
            delay = max(exponential_delay, recommended_delay or 0.0)
            LOGGER.warning(
                "%s failed on attempt %s/%s: %s. Retrying in %.1fs.",
                operation_name,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            sleep_fn(delay)
    raise LLMError(
        f"{operation_name} failed after {max_attempts} attempts: {last_error}",
        attempts=max_attempts,
    ) from last_error


def retry_call(
    operation: Callable[[], T],
    *,
    max_attempts: int,
    base_delay_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    operation_name: str = "LLM operation",
    retry_delay_fn: Callable[[Exception], float | None] | None = None,
    should_retry: Callable[[Exception], bool] | None = None,
) -> T:
    """Backward-compatible retry helper returning only the successful value."""
    return retry_call_with_attempts(
        operation,
        max_attempts=max_attempts,
        base_delay_seconds=base_delay_seconds,
        sleep_fn=sleep_fn,
        operation_name=operation_name,
        retry_delay_fn=retry_delay_fn,
        should_retry=should_retry,
    ).value


def is_daily_quota_error(error: Exception) -> bool:
    """Return whether Gemini identified the per-model requests-per-day quota."""
    message = str(error)
    return "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in message or (
        "generate_content_free_tier_requests" in message
        and "per day" in message.lower()
    )


def _status_code(error: Exception) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
        if callable(value):
            called = value()
            if isinstance(called, int):
                return called
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_retryable_provider_error(error: Exception) -> bool:
    """Classify hosted-provider errors without depending on one SDK's classes."""
    if isinstance(error, (ValueError, json.JSONDecodeError)):
        return True
    status = _status_code(error)
    if status in {400, 401, 403, 404, 413}:
        return False
    if status in {408, 409, 422, 429, 498} or (
        status is not None and status >= SERVER_ERROR_STATUS
    ):
        return True
    class_name = type(error).__name__.lower()
    return (
        any(
            marker in class_name
            for marker in ("connection", "timeout", "ratelimit", "internalserver")
        )
        or status is None
    )


def extract_server_retry_delay(error: Exception) -> float | None:
    """Extract a provider's recommended wait from headers or its error message."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if isinstance(headers, Mapping):
        header_value = headers.get("retry-after")
        if header_value is not None:
            try:
                return float(header_value) + 0.5
            except (TypeError, ValueError):
                pass

    message = str(error)
    seconds_match = re.search(
        r"(?:retry in\s+|retryDelay['\":\s]+)(\d+(?:\.\d+)?)s",
        message,
        flags=re.IGNORECASE,
    )
    if seconds_match:
        return float(seconds_match.group(1)) + 0.5
    milliseconds_match = re.search(
        r"(?:retry in\s+|retryDelay['\":\s]+)(\d+(?:\.\d+)?)ms",
        message,
        flags=re.IGNORECASE,
    )
    if milliseconds_match:
        return float(milliseconds_match.group(1)) / 1000 + 0.5
    return None


class _PacedClient:
    provider_name: str

    def __init__(
        self,
        settings: Settings,
        *,
        min_request_interval_seconds: float,
        sleep_fn: Callable[[float], None],
        clock_fn: Callable[[], float],
    ) -> None:
        self.settings = settings
        self.min_request_interval_seconds = min_request_interval_seconds
        self.sleep_fn = sleep_fn
        self.clock_fn = clock_fn
        self._last_request_at: float | None = None

    def _pace_request(self) -> None:
        now = self.clock_fn()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            remaining = self.min_request_interval_seconds - elapsed
            if remaining > 0:
                LOGGER.info(
                    "Pacing %s request for %.1fs.", self.provider_name, remaining
                )
                self.sleep_fn(remaining)
        self._last_request_at = self.clock_fn()


class GeminiClient(_PacedClient):
    """Official Google GenAI SDK implementation."""

    provider_name = "gemini"

    def __init__(
        self,
        settings: Settings,
        *,
        sdk_client: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(
            settings,
            min_request_interval_seconds=settings.min_request_interval_seconds,
            sleep_fn=sleep_fn,
            clock_fn=clock_fn,
        )
        if sdk_client is not None:
            self._client = sdk_client
            self._types = None
            return

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "The google-genai package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc

        self._client = genai.Client(
            api_key=settings.require_provider_api_key(self.provider_name),
            http_options=types.HttpOptions(
                timeout=int(settings.request_timeout_seconds * 1000)
            ),
        )
        self._types = types

    def _config(self, **kwargs: Any) -> Any:
        if self._types is None:
            return kwargs
        return self._types.GenerateContentConfig(**kwargs)

    def generate_text(
        self, prompt: str, *, model: str, temperature: float
    ) -> LLMResponse[str]:
        """Generate plain text and reject empty responses."""

        def operation() -> str:
            self._pace_request()
            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=self._config(temperature=temperature),
            )
            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise ValueError("Gemini returned an empty text response.")
            return text.strip()

        outcome = retry_call_with_attempts(
            operation,
            max_attempts=self.settings.max_attempts,
            base_delay_seconds=self.settings.retry_base_delay_seconds,
            sleep_fn=self.sleep_fn,
            operation_name="Gemini email generation",
            retry_delay_fn=extract_server_retry_delay,
            should_retry=is_retryable_provider_error,
        )
        return LLMResponse(
            outcome.value,
            CallMetadata(self.provider_name, model, outcome.attempts),
        )

    def generate_structured(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        response_model: type[T],
        validator: Callable[[T], T] | None = None,
    ) -> LLMResponse[T]:
        """Request JSON, parse it with Pydantic, and apply semantic validation."""
        _validate_response_model(response_model)

        def operation() -> T:
            self._pace_request()
            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=self._config(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_json_schema=response_model.model_json_schema(),
                ),
            )
            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise ValueError("Gemini returned an empty structured response.")
            parsed = response_model.model_validate_json(text)
            return validator(parsed) if validator else parsed

        outcome = retry_call_with_attempts(
            operation,
            max_attempts=self.settings.max_attempts,
            base_delay_seconds=self.settings.retry_base_delay_seconds,
            sleep_fn=self.sleep_fn,
            operation_name="Gemini structured judge evaluation",
            retry_delay_fn=extract_server_retry_delay,
            should_retry=is_retryable_provider_error,
        )
        return LLMResponse(
            outcome.value,
            CallMetadata(self.provider_name, model, outcome.attempts),
        )


class GroqClient(_PacedClient):
    """Official Groq SDK implementation using hosted open-weight models."""

    provider_name = "groq"

    def __init__(
        self,
        settings: Settings,
        *,
        sdk_client: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(
            settings,
            min_request_interval_seconds=settings.groq_min_request_interval_seconds,
            sleep_fn=sleep_fn,
            clock_fn=clock_fn,
        )
        if sdk_client is not None:
            self._client = sdk_client
            return
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError(
                "The groq package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
        self._client = Groq(
            api_key=settings.require_provider_api_key(self.provider_name),
            timeout=settings.request_timeout_seconds,
        )

    def _completion(self, **kwargs: Any) -> Any:
        self._pace_request()
        return self._client.chat.completions.create(**kwargs)

    @staticmethod
    def _content(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("Groq returned no completion choices.")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not content or not content.strip():
            raise ValueError("Groq returned an empty response.")
        return content.strip()

    def generate_text(
        self, prompt: str, *, model: str, temperature: float
    ) -> LLMResponse[str]:
        """Generate plain text through GroqCloud."""

        def operation() -> str:
            response = self._completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return self._content(response)

        outcome = retry_call_with_attempts(
            operation,
            max_attempts=self.settings.max_attempts,
            base_delay_seconds=self.settings.retry_base_delay_seconds,
            sleep_fn=self.sleep_fn,
            operation_name="Groq email generation",
            retry_delay_fn=extract_server_retry_delay,
            should_retry=is_retryable_provider_error,
        )
        return LLMResponse(
            outcome.value,
            CallMetadata(self.provider_name, model, outcome.attempts),
        )

    def generate_structured(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        response_model: type[T],
        validator: Callable[[T], T] | None = None,
    ) -> LLMResponse[T]:
        """Use Groq strict structured output and validate the result."""
        _validate_response_model(response_model)

        def operation() -> T:
            response = self._completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": response_model.model_json_schema(),
                    },
                },
            )
            parsed = response_model.model_validate_json(self._content(response))
            return validator(parsed) if validator else parsed

        outcome = retry_call_with_attempts(
            operation,
            max_attempts=self.settings.max_attempts,
            base_delay_seconds=self.settings.retry_base_delay_seconds,
            sleep_fn=self.sleep_fn,
            operation_name="Groq structured judge evaluation",
            retry_delay_fn=extract_server_retry_delay,
            should_retry=is_retryable_provider_error,
        )
        return LLMResponse(
            outcome.value,
            CallMetadata(self.provider_name, model, outcome.attempts),
        )


def _validate_response_model(response_model: type[Any]) -> None:
    if not isinstance(response_model, type) or not issubclass(
        response_model, BaseModel
    ):
        raise TypeError("response_model must be a Pydantic BaseModel class.")


class ProviderRouter:
    """Try configured providers in order and record successful fallback."""

    def __init__(
        self,
        clients: list[LLMClient],
        *,
        generation_models: dict[str, str],
        judge_models: dict[str, str],
    ) -> None:
        if not clients:
            raise ValueError("At least one provider client is required.")
        self.clients = clients
        self.generation_models = generation_models
        self.judge_models = judge_models

    def _run(
        self,
        operation: Callable[[LLMClient, str], LLMResponse[T]],
        models: dict[str, str],
        operation_name: str,
    ) -> LLMResponse[T]:
        errors: list[str] = []
        attempts = 0
        last_error: LLMError | None = None
        for index, client in enumerate(self.clients):
            provider = client.provider_name
            try:
                response = operation(client, models[provider])
                return replace(
                    response,
                    metadata=replace(
                        response.metadata,
                        attempts=attempts + response.metadata.attempts,
                        fallback_used=index > 0,
                    ),
                )
            except LLMError as exc:
                attempts += exc.attempts
                last_error = exc
                errors.append(f"{provider}: {exc}")
                if not exc.fallback_allowed:
                    raise
                if index + 1 < len(self.clients):
                    LOGGER.warning(
                        "%s failed through %s; switching to %s.",
                        operation_name,
                        provider,
                        self.clients[index + 1].provider_name,
                    )

        message = f"{operation_name} failed across all providers: {' | '.join(errors)}"
        if isinstance(last_error, DailyQuotaExceeded):
            raise DailyQuotaExceeded(message, attempts=attempts) from last_error
        raise LLMError(message, attempts=attempts) from last_error

    def generate_text(
        self, prompt: str, *, model: str, temperature: float
    ) -> LLMResponse[str]:
        """Generate text using the configured provider order."""
        primary = self.clients[0].provider_name
        models = {**self.generation_models, primary: model}
        return self._run(
            lambda client, provider_model: client.generate_text(
                prompt,
                model=provider_model,
                temperature=temperature,
            ),
            models,
            "Email generation",
        )

    def generate_structured(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        response_model: type[T],
        validator: Callable[[T], T] | None = None,
    ) -> LLMResponse[T]:
        """Generate structured output using the configured provider order."""
        primary = self.clients[0].provider_name
        models = {**self.judge_models, primary: model}
        return self._run(
            lambda client, provider_model: client.generate_structured(
                prompt,
                model=provider_model,
                temperature=temperature,
                response_model=response_model,
                validator=validator,
            ),
            models,
            "Structured judge evaluation",
        )


def create_llm_client(settings: Settings) -> ProviderRouter:
    """Build the configured hosted-provider route."""
    settings.require_configured_provider_keys()
    factories: dict[str, Callable[[Settings], LLMClient]] = {
        "groq": GroqClient,
        "gemini": GeminiClient,
    }
    clients = [factories[provider](settings) for provider in settings.provider_order]
    return ProviderRouter(
        clients,
        generation_models={
            provider: settings.generation_model_for(provider)
            for provider in settings.provider_order
        },
        judge_models={
            provider: settings.judge_model_for(provider)
            for provider in settings.provider_order
        },
    )
