"""Email generation service."""

from __future__ import annotations

from src.config import Settings
from src.llm_client import LLMClient, LLMResponse
from src.models import Scenario, Strategy
from src.prompts import build_generation_prompt


class EmailGenerator:
    """Generate emails through one of the two prompt strategies."""

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def generate(
        self,
        intent: str,
        key_facts: list[str],
        tone: str,
        strategy: Strategy = Strategy.ADVANCED,
    ) -> str:
        """Generate an email while preserving the original string-returning API."""
        return self.generate_with_metadata(intent, key_facts, tone, strategy).value

    def generate_with_metadata(
        self,
        intent: str,
        key_facts: list[str],
        tone: str,
        strategy: Strategy = Strategy.ADVANCED,
    ) -> LLMResponse[str]:
        """Generate an email and return provider/model call metadata."""
        if not intent.strip():
            raise ValueError("Intent cannot be blank.")
        cleaned_facts = [fact.strip() for fact in key_facts if fact.strip()]
        if not cleaned_facts:
            raise ValueError("At least one key fact is required.")
        if not tone.strip():
            raise ValueError("Tone cannot be blank.")

        prompt = build_generation_prompt(intent, cleaned_facts, tone, strategy)
        return self.client.generate_text(
            prompt,
            model=self.settings.primary_generation_model,
            temperature=self.settings.generation_temperature,
        )

    def generate_scenario(
        self,
        scenario: Scenario,
        strategy: Strategy = Strategy.ADVANCED,
    ) -> str:
        return self.generate(
            scenario.intent,
            scenario.key_facts,
            scenario.tone,
            strategy,
        )

    def generate_scenario_with_metadata(
        self,
        scenario: Scenario,
        strategy: Strategy = Strategy.ADVANCED,
    ) -> LLMResponse[str]:
        """Generate one scenario and retain provider/model metadata."""
        return self.generate_with_metadata(
            scenario.intent,
            scenario.key_facts,
            scenario.tone,
            strategy,
        )
