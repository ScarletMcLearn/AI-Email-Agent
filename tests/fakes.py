"""Reusable deterministic test doubles."""

from __future__ import annotations

import re
from typing import Any

from src.llm_client import CallMetadata, LLMResponse
from src.models import FactAssessment, JudgeResult


class FakeLLMClient:
    """Deterministic client that preserves the production client contract."""

    provider_name = "gemini"

    def __init__(self, *, fail_baseline_once: bool = False) -> None:
        self.fail_baseline_once = fail_baseline_once
        self._failed = False
        self.text_calls = 0
        self.structured_calls = 0

    def generate_text(
        self, prompt: str, *, model: str, temperature: float
    ) -> LLMResponse[str]:
        self.text_calls += 1
        advanced = "senior executive communication assistant" in prompt
        if self.fail_baseline_once and not advanced and not self._failed:
            self._failed = True
            raise RuntimeError("Simulated generation failure")
        marker = "ADVANCED-EMAIL" if advanced else "BASELINE-EMAIL"
        return LLMResponse(
            (
                f"Subject: Test Email\n\nHello,\n\n{marker}: This deterministic "
                "fixture represents a generated email.\n\nBest regards,\n[Your Name]"
            ),
            CallMetadata(self.provider_name, model, 1),
        )

    def generate_structured(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        response_model: type[Any],
        validator: Any = None,
    ) -> LLMResponse[Any]:
        self.structured_calls += 1
        fact_block = prompt.split("Required facts:\n", 1)[1].split(
            "\n\nHUMAN REFERENCE EMAIL", 1
        )[0]
        fact_indexes = [
            int(match.group(1)) for match in re.finditer(r"(?m)^(\d+)\. ", fact_block)
        ]
        advanced = "ADVANCED-EMAIL" in prompt
        raw_score = 5 if advanced else 3
        result = JudgeResult(
            fact_assessments=[
                FactAssessment(
                    fact_index=index,
                    status="accurate",
                    rationale="The fixture marks this required fact as present.",
                )
                for index in fact_indexes
            ],
            tone_score=raw_score,
            tone_rationale="Deterministic fixture score.",
            professional_quality_score=raw_score,
            professional_quality_rationale="Deterministic fixture score.",
        )
        parsed = response_model.model_validate(result.model_dump())
        value = validator(parsed) if validator else parsed
        return LLMResponse(value, CallMetadata(self.provider_name, model, 1))
