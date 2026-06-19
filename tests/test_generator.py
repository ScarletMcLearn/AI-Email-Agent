from __future__ import annotations

import pytest
from src.config import Settings
from src.generator import EmailGenerator
from src.llm_client import CallMetadata, LLMResponse
from src.models import Scenario, Strategy


class RecordingClient:
    provider_name = "gemini"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_text(
        self, prompt: str, *, model: str, temperature: float
    ) -> LLMResponse[str]:
        self.calls.append(
            {"prompt": prompt, "model": model, "temperature": temperature}
        )
        return LLMResponse(
            "Subject: Recorded",
            CallMetadata(self.provider_name, model, 1),
        )


@pytest.fixture
def generator() -> EmailGenerator:
    return EmailGenerator(
        RecordingClient(),
        Settings(
            api_key=None,
            generation_model="example-model",
            provider_order=("gemini",),
        ),
    )


@pytest.mark.parametrize(
    ("intent", "facts", "tone", "message"),
    [
        (" ", ["A fact"], "formal", "Intent"),
        ("Follow up", ["", "  "], "formal", "key fact"),
        ("Follow up", ["A fact"], " ", "Tone"),
    ],
)
def test_generate_rejects_incomplete_requests(
    generator: EmailGenerator,
    intent: str,
    facts: list[str],
    tone: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        generator.generate(intent, facts, tone)


def test_generate_cleans_facts_and_delegates_model_configuration(
    generator: EmailGenerator,
) -> None:
    result = generator.generate(
        " Follow up ",
        [" Deadline is Friday. ", "", " Alex owns it. "],
        " formal ",
        Strategy.BASELINE,
    )

    call = generator.client.calls[0]
    assert result == "Subject: Recorded"
    assert call["model"] == "example-model"
    assert call["temperature"] == 0.4
    assert "- Deadline is Friday." in str(call["prompt"])
    assert "senior executive communication assistant" not in str(call["prompt"])


def test_generate_scenario_uses_advanced_strategy_by_default(
    generator: EmailGenerator,
) -> None:
    scenario = Scenario(
        id="example",
        intent="Send a project update",
        key_facts=["Launch is Monday."],
        tone="concise",
        human_reference_email="Subject: Update\n\nHello,\n\nLaunch is Monday.",
    )

    generator.generate_scenario(scenario)

    assert "senior executive communication assistant" in str(
        generator.client.calls[0]["prompt"]
    )
