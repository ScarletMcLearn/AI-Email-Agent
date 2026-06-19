"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS_PATH = PROJECT_ROOT / "data" / "scenarios.json"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings for generation and evaluation."""

    api_key: str | None
    groq_api_key: str | None = None
    generation_model: str = "gemini-2.5-flash"
    judge_model: str = "gemini-2.5-flash-lite"
    groq_generation_model: str = "llama-3.3-70b-versatile"
    groq_judge_model: str = "openai/gpt-oss-20b"
    provider_order: tuple[str, ...] = ("groq", "gemini")
    generation_temperature: float = 0.4
    judge_temperature: float = 0.0
    max_attempts: int = 3
    retry_base_delay_seconds: float = 1.0
    min_request_interval_seconds: float = 13.0
    groq_min_request_interval_seconds: float = 6.0
    request_timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls, env_file: Path | str | None = None) -> Settings:
        """Load settings from .env and the process environment."""
        load_dotenv(dotenv_path=env_file or PROJECT_ROOT / ".env")
        generation_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        judge_model = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.5-flash-lite").strip()
        api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip() or None
        groq_generation_model = os.getenv(
            "GROQ_GENERATION_MODEL", "llama-3.3-70b-versatile"
        ).strip()
        groq_judge_model = os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b").strip()
        provider_order_value = os.getenv("LLM_PROVIDER_ORDER", "groq,gemini").strip()
        interval_value = os.getenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "13").strip()
        groq_interval_value = os.getenv(
            "GROQ_MIN_REQUEST_INTERVAL_SECONDS", "6"
        ).strip()
        timeout_value = os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "120").strip()
        try:
            min_request_interval_seconds = float(interval_value)
            groq_min_request_interval_seconds = float(groq_interval_value)
            request_timeout_seconds = float(timeout_value)
        except ValueError as exc:
            raise ConfigurationError(
                "Provider interval and timeout settings must be numbers."
            ) from exc
        if min_request_interval_seconds < 0:
            raise ConfigurationError(
                "GEMINI_MIN_REQUEST_INTERVAL_SECONDS cannot be negative."
            )
        if groq_min_request_interval_seconds < 0:
            raise ConfigurationError(
                "GROQ_MIN_REQUEST_INTERVAL_SECONDS cannot be negative."
            )
        if request_timeout_seconds <= 0:
            raise ConfigurationError(
                "GEMINI_REQUEST_TIMEOUT_SECONDS must be greater than zero."
            )
        provider_order = tuple(
            provider.strip().lower()
            for provider in provider_order_value.split(",")
            if provider.strip()
        )
        supported_providers = {"groq", "gemini"}
        if not provider_order:
            raise ConfigurationError(
                "LLM_PROVIDER_ORDER must name at least one provider."
            )
        if len(provider_order) != len(set(provider_order)):
            raise ConfigurationError("LLM_PROVIDER_ORDER cannot contain duplicates.")
        unsupported = set(provider_order) - supported_providers
        if unsupported:
            message = (
                "LLM_PROVIDER_ORDER contains unsupported providers: "
                f"{', '.join(sorted(unsupported))}."
            )
            raise ConfigurationError(message)
        return cls(
            api_key=api_key,
            groq_api_key=groq_api_key,
            generation_model=generation_model,
            judge_model=judge_model,
            groq_generation_model=groq_generation_model,
            groq_judge_model=groq_judge_model,
            provider_order=provider_order,
            min_request_interval_seconds=min_request_interval_seconds,
            groq_min_request_interval_seconds=groq_min_request_interval_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )

    def require_api_key(self) -> str:
        """Return the configured API key or raise an actionable error."""
        if not self.api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not configured. Copy .env.example to .env, "
                "add a Google AI Studio API key, and run the command again."
            )
        return self.api_key

    @property
    def primary_provider(self) -> str:
        """Return the first configured provider."""
        return self.provider_order[0]

    @property
    def primary_generation_model(self) -> str:
        """Return the generation model for the primary provider."""
        return self.generation_model_for(self.primary_provider)

    @property
    def primary_judge_model(self) -> str:
        """Return the judge model for the primary provider."""
        return self.judge_model_for(self.primary_provider)

    def generation_model_for(self, provider: str) -> str:
        """Return the configured generation model for a provider."""
        if provider == "groq":
            return self.groq_generation_model
        if provider == "gemini":
            return self.generation_model
        message = f"Unsupported LLM provider: {provider}."
        raise ConfigurationError(message)

    def judge_model_for(self, provider: str) -> str:
        """Return the configured judge model for a provider."""
        if provider == "groq":
            return self.groq_judge_model
        if provider == "gemini":
            return self.judge_model
        message = f"Unsupported LLM provider: {provider}."
        raise ConfigurationError(message)

    def require_provider_api_key(self, provider: str) -> str:
        """Return one provider key or raise an actionable configuration error."""
        if provider == "gemini":
            return self.require_api_key()
        if provider == "groq":
            if not self.groq_api_key:
                raise ConfigurationError(
                    "GROQ_API_KEY is not configured. Add a GroqCloud API key "
                    "to .env and run the command again."
                )
            return self.groq_api_key
        message = f"Unsupported LLM provider: {provider}."
        raise ConfigurationError(message)

    def require_configured_provider_keys(self) -> None:
        """Validate keys for every configured primary or fallback provider."""
        for provider in self.provider_order:
            self.require_provider_api_key(provider)

    def is_configured_model(self, provider: str, model: str, *, judge: bool) -> bool:
        """Return whether a provider/model remains part of the active route."""
        if provider not in self.provider_order:
            return False
        configured = (
            self.judge_model_for(provider)
            if judge
            else self.generation_model_for(provider)
        )
        return model == configured
