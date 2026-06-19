from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import src.llm_client as llm_module
from src.config import Settings
from src.llm_client import (
    CallMetadata,
    DailyQuotaExceeded,
    GeminiClient,
    GroqClient,
    LLMError,
    LLMResponse,
    ProviderRouter,
    create_llm_client,
    extract_server_retry_delay,
    is_daily_quota_error,
    is_retryable_provider_error,
    retry_call,
)
from src.models import JudgeResult


@dataclass
class FakeResponse:
    text: str


class FakeModels:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.call_count = 0

    def generate_content(self, **kwargs):
        self.call_count += 1
        return FakeResponse(next(self.responses))


class FakeSDKClient:
    def __init__(self, responses: list[str]) -> None:
        self.models = FakeModels(responses)


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeGroqResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return FakeGroqResponse(response)


class FakeChat:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.completions = FakeCompletions(responses)


class FakeGroqSDK:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.chat = FakeChat(responses)


class StatusError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def valid_judge_json() -> str:
    return (
        '{"fact_assessments":[{"fact_index":0,"status":"accurate",'
        '"rationale":"Included."}],"tone_score":5,'
        '"tone_rationale":"Matches.","professional_quality_score":4,'
        '"professional_quality_rationale":"Strong."}'
    )


def test_structured_generation_retries_malformed_json() -> None:
    sdk = FakeSDKClient(["not-json", valid_judge_json()])
    settings = Settings(api_key=None, max_attempts=3, retry_base_delay_seconds=0)
    client = GeminiClient(settings, sdk_client=sdk, sleep_fn=lambda _: None)
    result = client.generate_structured(
        "judge",
        model="test",
        temperature=0.0,
        response_model=JudgeResult,
    )
    assert result.value.tone_score == 5
    assert result.metadata.provider == "gemini"
    assert sdk.models.call_count == 2


def test_retry_call_raises_after_three_attempts() -> None:
    attempts = 0

    def fail() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("failure")

    with pytest.raises(LLMError, match="after 3 attempts"):
        retry_call(
            fail,
            max_attempts=3,
            base_delay_seconds=0,
            sleep_fn=lambda _: None,
        )
    assert attempts == 3


def test_server_retry_delay_parsing() -> None:
    error = RuntimeError("Please retry in 31.692s.")
    assert extract_server_retry_delay(error) == pytest.approx(32.192)


def test_server_retry_delay_uses_retry_after_header() -> None:
    error = RuntimeError("limited")
    error.response = SimpleNamespace(headers={"retry-after": "3"})
    assert extract_server_retry_delay(error) == 3.5
    error.response = SimpleNamespace(headers={"retry-after": "not-a-number"})
    assert extract_server_retry_delay(error) is None


def test_retry_call_honors_server_delay() -> None:
    attempts = 0
    delays: list[float] = []

    def eventually_succeed() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Please retry in 8s.")
        return "ok"

    result = retry_call(
        eventually_succeed,
        max_attempts=3,
        base_delay_seconds=1,
        sleep_fn=delays.append,
        retry_delay_fn=extract_server_retry_delay,
    )
    assert result == "ok"
    assert delays == [8.5]


def test_daily_quota_stops_without_retry() -> None:
    attempts = 0

    def daily_quota() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    with pytest.raises(DailyQuotaExceeded):
        retry_call(
            daily_quota,
            max_attempts=3,
            base_delay_seconds=1,
            sleep_fn=lambda _: None,
        )
    assert attempts == 1
    assert is_daily_quota_error(
        RuntimeError("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    )


def test_plain_text_generation_returns_trimmed_content_and_sdk_arguments() -> None:
    sdk = FakeSDKClient(["  generated email  "])
    client = GeminiClient(
        Settings(api_key=None, min_request_interval_seconds=0),
        sdk_client=sdk,
        sleep_fn=lambda _: None,
    )
    result = client.generate_text("prompt", model="model", temperature=0.4)
    assert result.value == "generated email"
    assert result.metadata == CallMetadata("gemini", "model", 1)


@pytest.mark.parametrize("response", ["", "   "])
def test_plain_text_generation_retries_empty_responses(response: str) -> None:
    sdk = FakeSDKClient([response, response])
    client = GeminiClient(
        Settings(api_key=None, max_attempts=2, retry_base_delay_seconds=0),
        sdk_client=sdk,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(LLMError, match="empty text"):
        client.generate_text("prompt", model="model", temperature=0.4)


def test_structured_generation_validates_response_model_and_semantics() -> None:
    client = GeminiClient(Settings(api_key=None), sdk_client=FakeSDKClient([]))
    with pytest.raises(TypeError, match="Pydantic"):
        client.generate_structured(
            "prompt", model="model", temperature=0, response_model=dict
        )

    sdk = FakeSDKClient([valid_judge_json(), valid_judge_json()])
    semantic_client = GeminiClient(
        Settings(api_key=None, max_attempts=2, retry_base_delay_seconds=0),
        sdk_client=sdk,
        sleep_fn=lambda _: None,
    )
    calls = 0

    def reject_once(result: JudgeResult) -> JudgeResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("semantic failure")
        return result

    result = semantic_client.generate_structured(
        "prompt",
        model="model",
        temperature=0,
        response_model=JudgeResult,
        validator=reject_once,
    )
    assert result.value.tone_score == 5
    assert calls == 2


def test_gemini_structured_generation_retries_empty_response() -> None:
    client = GeminiClient(
        Settings(api_key=None, max_attempts=1),
        sdk_client=FakeSDKClient([""]),
        sleep_fn=lambda _: None,
    )
    with pytest.raises(LLMError, match="empty structured"):
        client.generate_structured(
            "prompt",
            model="model",
            temperature=0,
            response_model=JudgeResult,
        )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("retryDelay: 250ms", 0.75),
        ("ordinary error", None),
    ],
)
def test_retry_delay_parses_milliseconds_or_returns_none(
    message: str, expected: float | None
) -> None:
    assert extract_server_retry_delay(RuntimeError(message)) == expected


def test_request_pacing_waits_only_for_remaining_interval() -> None:
    times = iter([10.0, 10.0, 12.0, 15.0, 20.0])
    delays: list[float] = []
    sdk = FakeSDKClient(["one", "two"])
    client = GeminiClient(
        Settings(api_key=None, min_request_interval_seconds=5),
        sdk_client=sdk,
        sleep_fn=delays.append,
        clock_fn=lambda: next(times),
    )
    client.generate_text("one", model="model", temperature=0)
    client.generate_text("two", model="model", temperature=0)
    assert delays == [3.0]


def test_alternate_daily_quota_signature_is_recognized() -> None:
    assert is_daily_quota_error(
        RuntimeError("generate_content_free_tier_requests limit per day")
    )


def test_groq_text_and_strict_structured_generation() -> None:
    sdk = FakeGroqSDK(["  generated email  ", valid_judge_json()])
    settings = Settings(
        api_key=None,
        groq_api_key=None,
        max_attempts=1,
        groq_min_request_interval_seconds=0,
    )
    client = GroqClient(settings, sdk_client=sdk, sleep_fn=lambda _: None)

    text = client.generate_text("prompt", model="groq-text", temperature=0.4)
    structured = client.generate_structured(
        "judge",
        model="openai/gpt-oss-20b",
        temperature=0,
        response_model=JudgeResult,
    )

    assert text.value == "generated email"
    assert text.metadata == CallMetadata("groq", "groq-text", 1)
    assert structured.value.tone_score == 5
    structured_call = sdk.chat.completions.calls[1]
    response_format = structured_call["response_format"]
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False


def test_groq_rejects_response_without_choices() -> None:
    response = SimpleNamespace(choices=[])
    with pytest.raises(ValueError, match="no completion choices"):
        GroqClient._content(response)  # noqa: SLF001


def test_groq_retries_transient_and_invalid_output_errors() -> None:
    sdk = FakeGroqSDK(
        [
            StatusError("rate limited, retry in 2s", 429),
            "",
            "success",
        ]
    )
    delays: list[float] = []
    client = GroqClient(
        Settings(
            api_key=None,
            max_attempts=3,
            retry_base_delay_seconds=0,
            groq_min_request_interval_seconds=0,
        ),
        sdk_client=sdk,
        sleep_fn=delays.append,
    )
    response = client.generate_text("prompt", model="model", temperature=0.4)
    assert response.value == "success"
    assert response.metadata.attempts == 3
    assert delays == [2.5, 0.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413])
def test_non_retryable_provider_errors_fail_immediately(status: int) -> None:
    sdk = FakeGroqSDK([StatusError("bad request", status)])
    client = GroqClient(
        Settings(api_key=None, max_attempts=3, groq_min_request_interval_seconds=0),
        sdk_client=sdk,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(LLMError) as caught:
        client.generate_text("prompt", model="model", temperature=0.4)
    assert caught.value.attempts == 1
    assert caught.value.fallback_allowed is False
    assert len(sdk.chat.completions.calls) == 1


@pytest.mark.parametrize("status", [408, 409, 422, 429, 498, 500, 503])
def test_transient_provider_statuses_are_retryable(status: int) -> None:
    assert is_retryable_provider_error(StatusError("temporary", status))


def test_provider_error_classification_supports_sdk_shapes() -> None:
    class CodeError(RuntimeError):
        def code(self) -> int:
            return 401

    class ResponseError(RuntimeError):
        response = SimpleNamespace(status_code=503)

    class APIConnectionError(RuntimeError):
        pass

    assert not is_retryable_provider_error(CodeError("auth"))
    assert is_retryable_provider_error(ResponseError("unavailable"))
    assert is_retryable_provider_error(APIConnectionError("network"))
    assert is_retryable_provider_error(ValueError("invalid model output"))


class FixedClient:
    def __init__(
        self,
        provider_name: str,
        *,
        response: str | None = None,
        error: LLMError | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.response = response
        self.error = error
        self.models: list[str] = []

    def generate_text(self, prompt: str, *, model: str, temperature: float):
        self.models.append(model)
        if self.error:
            raise self.error
        return LLMResponse(
            self.response or "ok",
            CallMetadata(self.provider_name, model, 1),
        )

    def generate_structured(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        response_model,
        validator=None,
    ):
        self.models.append(model)
        if self.error:
            raise self.error
        parsed = response_model.model_validate_json(valid_judge_json())
        value = validator(parsed) if validator else parsed
        return LLMResponse(value, CallMetadata(self.provider_name, model, 1))


def test_router_falls_back_with_provider_specific_models_and_total_attempts() -> None:
    groq = FixedClient(
        "groq",
        error=LLMError("quota", attempts=3),
    )
    gemini = FixedClient("gemini", response="fallback")
    router = ProviderRouter(
        [groq, gemini],
        generation_models={"groq": "groq-gen", "gemini": "gemini-gen"},
        judge_models={"groq": "groq-judge", "gemini": "gemini-judge"},
    )

    result = router.generate_text("prompt", model="groq-gen", temperature=0.4)
    judge = router.generate_structured(
        "judge",
        model="groq-judge",
        temperature=0,
        response_model=JudgeResult,
    )

    assert result.value == "fallback"
    assert result.metadata == CallMetadata(
        "gemini", "gemini-gen", 4, fallback_used=True
    )
    assert judge.metadata.provider == "gemini"
    assert groq.models == ["groq-gen", "groq-judge"]
    assert gemini.models == ["gemini-gen", "gemini-judge"]


def test_router_does_not_fallback_for_configuration_errors() -> None:
    groq = FixedClient(
        "groq",
        error=LLMError("invalid key", attempts=1, fallback_allowed=False),
    )
    gemini = FixedClient("gemini", response="unused")
    router = ProviderRouter(
        [groq, gemini],
        generation_models={"groq": "groq-gen", "gemini": "gemini-gen"},
        judge_models={"groq": "groq-judge", "gemini": "gemini-judge"},
    )
    with pytest.raises(LLMError, match="invalid key"):
        router.generate_text("prompt", model="groq-gen", temperature=0.4)
    assert gemini.models == []


def test_router_requires_a_client_and_reports_non_quota_combined_failure() -> None:
    with pytest.raises(ValueError, match="At least one"):
        ProviderRouter([], generation_models={}, judge_models={})

    groq = FixedClient("groq", error=LLMError("groq unavailable", attempts=1))
    router = ProviderRouter(
        [groq],
        generation_models={"groq": "groq-gen"},
        judge_models={"groq": "groq-judge"},
    )
    with pytest.raises(LLMError, match="all providers") as caught:
        router.generate_text("prompt", model="groq-gen", temperature=0.4)
    assert not isinstance(caught.value, DailyQuotaExceeded)


def test_router_reports_combined_failures_and_preserves_daily_quota() -> None:
    groq = FixedClient("groq", error=LLMError("groq unavailable", attempts=2))
    gemini = FixedClient(
        "gemini",
        error=DailyQuotaExceeded("gemini daily quota", attempts=1),
    )
    router = ProviderRouter(
        [groq, gemini],
        generation_models={"groq": "groq-gen", "gemini": "gemini-gen"},
        judge_models={"groq": "groq-judge", "gemini": "gemini-judge"},
    )
    with pytest.raises(DailyQuotaExceeded, match="all providers") as caught:
        router.generate_text("prompt", model="groq-gen", temperature=0.4)
    assert caught.value.attempts == 3


def test_create_client_requires_every_configured_key() -> None:
    with pytest.raises(Exception, match="GROQ_API_KEY"):
        create_llm_client(Settings(api_key="gemini", groq_api_key=None))


def test_create_client_builds_real_offline_sdk_clients() -> None:
    settings = Settings(
        api_key="gemini-key",
        groq_api_key="groq-key",
        min_request_interval_seconds=0,
        groq_min_request_interval_seconds=0,
    )
    router = create_llm_client(settings)
    assert [client.provider_name for client in router.clients] == ["groq", "gemini"]
    gemini = router.clients[1]
    assert gemini._config(temperature=0).temperature == 0  # noqa: SLF001


def test_create_client_uses_configured_factory_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []

    def fake_groq(_settings: Settings) -> FixedClient:
        created.append("groq")
        return FixedClient("groq", response="groq")

    def fake_gemini(_settings: Settings) -> FixedClient:
        created.append("gemini")
        return FixedClient("gemini", response="gemini")

    monkeypatch.setattr(llm_module, "GroqClient", fake_groq)
    monkeypatch.setattr(llm_module, "GeminiClient", fake_gemini)
    settings = Settings(
        api_key="gemini-key",
        groq_api_key="groq-key",
        provider_order=("gemini", "groq"),
    )
    router = llm_module.create_llm_client(settings)
    assert created == ["gemini", "groq"]
    assert router.generation_models["groq"] == settings.groq_generation_model
