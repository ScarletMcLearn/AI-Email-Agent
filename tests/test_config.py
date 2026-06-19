from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from src.config import ConfigurationError, Settings

if TYPE_CHECKING:
    from pathlib import Path


def test_settings_from_env_uses_documented_defaults(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmp_path: Path,
) -> None:
    for name in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_JUDGE_MODEL",
        "GEMINI_MIN_REQUEST_INTERVAL_SECONDS",
        "GEMINI_REQUEST_TIMEOUT_SECONDS",
        "GROQ_API_KEY",
        "GROQ_GENERATION_MODEL",
        "GROQ_JUDGE_MODEL",
        "GROQ_MIN_REQUEST_INTERVAL_SECONDS",
        "LLM_PROVIDER_ORDER",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env(workspace_tmp_path / "missing.env")

    assert settings.api_key is None
    assert settings.groq_api_key is None
    assert settings.generation_model == "gemini-2.5-flash"
    assert settings.judge_model == "gemini-2.5-flash-lite"
    assert settings.groq_generation_model == "llama-3.3-70b-versatile"
    assert settings.groq_judge_model == "openai/gpt-oss-20b"
    assert settings.provider_order == ("groq", "gemini")
    assert settings.min_request_interval_seconds == 13
    assert settings.groq_min_request_interval_seconds == 6
    assert settings.request_timeout_seconds == 120


def test_settings_from_env_strips_and_converts_values(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmp_path: Path,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", " secret ")
    monkeypatch.setenv("GEMINI_MODEL", " generator ")
    monkeypatch.setenv("GEMINI_JUDGE_MODEL", " judge ")
    monkeypatch.setenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", " 0.5 ")
    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT_SECONDS", " 30 ")
    monkeypatch.setenv("GROQ_API_KEY", " groq-secret ")
    monkeypatch.setenv("GROQ_GENERATION_MODEL", " groq-generator ")
    monkeypatch.setenv("GROQ_JUDGE_MODEL", " groq-judge ")
    monkeypatch.setenv("GROQ_MIN_REQUEST_INTERVAL_SECONDS", " 0.25 ")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", " gemini, groq ")

    settings = Settings.from_env(workspace_tmp_path / "missing.env")

    assert settings.api_key == "secret"
    assert settings.groq_api_key == "groq-secret"
    assert settings.generation_model == "generator"
    assert settings.judge_model == "judge"
    assert settings.groq_generation_model == "groq-generator"
    assert settings.groq_judge_model == "groq-judge"
    assert settings.provider_order == ("gemini", "groq")
    assert settings.min_request_interval_seconds == 0.5
    assert settings.groq_min_request_interval_seconds == 0.25
    assert settings.request_timeout_seconds == 30


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "soon", "must be numbers"),
        ("GROQ_MIN_REQUEST_INTERVAL_SECONDS", "soon", "must be numbers"),
        ("GEMINI_REQUEST_TIMEOUT_SECONDS", "later", "must be numbers"),
        ("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "-1", "cannot be negative"),
        ("GROQ_MIN_REQUEST_INTERVAL_SECONDS", "-1", "cannot be negative"),
        ("GEMINI_REQUEST_TIMEOUT_SECONDS", "0", "greater than zero"),
    ],
)
def test_settings_rejects_invalid_numeric_configuration(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmp_path: Path,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message):
        Settings.from_env(workspace_tmp_path / "missing.env")


def test_require_api_key_returns_key_or_explains_setup() -> None:
    assert Settings(api_key="key").require_api_key() == "key"
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        Settings(api_key="").require_api_key()


def test_provider_configuration_helpers_validate_keys_models_and_order(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmp_path: Path,
) -> None:
    settings = Settings(
        api_key="gemini-key",
        groq_api_key="groq-key",
        generation_model="gemini-generation",
        judge_model="gemini-judge",
        groq_generation_model="groq-generation",
        groq_judge_model="groq-judge",
    )
    settings.require_configured_provider_keys()
    assert settings.primary_provider == "groq"
    assert settings.primary_generation_model == "groq-generation"
    assert settings.primary_judge_model == "groq-judge"
    assert settings.is_configured_model("gemini", "gemini-judge", judge=True)
    assert not settings.is_configured_model("gemini", "wrong", judge=True)
    assert not settings.is_configured_model("unconfigured", "model", judge=False)
    for method in (
        settings.generation_model_for,
        settings.judge_model_for,
        settings.require_provider_api_key,
    ):
        with pytest.raises(ConfigurationError, match="Unsupported"):
            method("unknown")

    monkeypatch.setenv("LLM_PROVIDER_ORDER", "groq,groq")
    with pytest.raises(ConfigurationError, match="duplicates"):
        Settings.from_env(workspace_tmp_path / "missing.env")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "groq,unknown")
    with pytest.raises(ConfigurationError, match="unsupported"):
        Settings.from_env(workspace_tmp_path / "missing.env")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", ",")
    with pytest.raises(ConfigurationError, match="at least one"):
        Settings.from_env(workspace_tmp_path / "missing.env")


def test_missing_groq_key_is_actionable() -> None:
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        Settings(api_key="gemini", groq_api_key=None).require_configured_provider_keys()
