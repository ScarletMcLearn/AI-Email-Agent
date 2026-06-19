"""Validated domain and evaluation models."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Strategy(str, Enum):
    """Available prompting strategies."""

    ADVANCED = "A"
    BASELINE = "B"

    @property
    def label(self) -> str:
        return (
            "Strategy A (advanced)"
            if self is Strategy.ADVANCED
            else "Strategy B (baseline)"
        )

    @classmethod
    def from_user_input(cls, value: str) -> Strategy:
        normalized = value.strip().lower()
        aliases = {
            "a": cls.ADVANCED,
            "advanced": cls.ADVANCED,
            "strategy a": cls.ADVANCED,
            "b": cls.BASELINE,
            "baseline": cls.BASELINE,
            "strategy b": cls.BASELINE,
        }
        if normalized not in aliases:
            raise ValueError("Strategy must be A/advanced or B/baseline.")
        return aliases[normalized]


class Scenario(BaseModel):
    """One email-generation scenario."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(min_length=1)
    intent: str = Field(min_length=3)
    key_facts: list[str] = Field(min_length=1)
    tone: str = Field(min_length=2)
    human_reference_email: str = Field(min_length=20)

    @field_validator("key_facts")
    @classmethod
    def validate_key_facts(cls, facts: list[str]) -> list[str]:
        cleaned = [fact.strip() for fact in facts]
        if any(not fact for fact in cleaned):
            raise ValueError("Key facts cannot contain blank entries.")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("Key facts must be unique within a scenario.")
        return cleaned


FactStatus = Literal["accurate", "partial", "missing", "contradicted"]


class FactAssessment(BaseModel):
    """Judge assessment of one required fact."""

    model_config = ConfigDict(extra="forbid")

    fact_index: int = Field(ge=0)
    status: FactStatus
    rationale: str = Field(min_length=1)


class JudgeResult(BaseModel):
    """Strict structured response returned by the LLM judge."""

    model_config = ConfigDict(extra="forbid")

    fact_assessments: list[FactAssessment]
    tone_score: int = Field(ge=1, le=5)
    tone_rationale: str = Field(min_length=1)
    professional_quality_score: int = Field(ge=1, le=5)
    professional_quality_rationale: str = Field(min_length=1)


class EvaluationScores(BaseModel):
    """Calculated normalized scores for one generated email."""

    fact_coverage_score: float = Field(ge=0.0, le=1.0)
    tone_raw_score: int = Field(ge=1, le=5)
    tone_match_score: float = Field(ge=0.0, le=1.0)
    professional_quality_raw_score: int = Field(ge=1, le=5)
    professional_quality_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
