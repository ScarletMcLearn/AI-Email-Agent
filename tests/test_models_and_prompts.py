from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from src.evaluate import load_scenarios, normalize_five_point_score
from src.models import FactAssessment, JudgeResult, Scenario, Strategy
from src.prompts import build_generation_prompt, build_judge_prompt, format_key_facts


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a", Strategy.ADVANCED),
        (" ADVANCED ", Strategy.ADVANCED),
        ("strategy a", Strategy.ADVANCED),
        ("B", Strategy.BASELINE),
        ("baseline", Strategy.BASELINE),
        ("strategy b", Strategy.BASELINE),
    ],
)
def test_strategy_user_aliases_are_documented(value: str, expected: Strategy) -> None:
    assert Strategy.from_user_input(value) is expected


def test_strategy_labels_and_invalid_input() -> None:
    assert Strategy.ADVANCED.label == "Strategy A (advanced)"
    assert Strategy.BASELINE.label == "Strategy B (baseline)"
    with pytest.raises(ValueError, match="A/advanced"):
        Strategy.from_user_input("experimental")


def test_scenario_cleans_fields_and_facts() -> None:
    scenario = Scenario(
        id=" example ",
        intent=" Send an update ",
        key_facts=[" First fact. ", "Second fact."],
        tone=" formal ",
        human_reference_email="Subject: Update\n\nA sufficiently complete reference.",
    )
    assert scenario.id == "example"
    assert scenario.key_facts == ["First fact.", "Second fact."]


@pytest.mark.parametrize("facts", [[""], ["same", "same"], ["same", " same "]])
def test_scenario_rejects_blank_or_duplicate_facts(facts: list[str]) -> None:
    with pytest.raises(ValidationError):
        Scenario(
            id="example",
            intent="Send an update",
            key_facts=facts,
            tone="formal",
            human_reference_email=(
                "Subject: Update\n\nA sufficiently complete reference."
            ),
        )


def test_judge_models_forbid_unknown_fields_and_invalid_scores() -> None:
    with pytest.raises(ValidationError):
        FactAssessment(
            fact_index=0,
            status="accurate",
            rationale="Present.",
            unexpected=True,
        )
    with pytest.raises(ValidationError):
        JudgeResult(
            fact_assessments=[],
            tone_score=6,
            tone_rationale="Too high.",
            professional_quality_score=5,
            professional_quality_rationale="Good.",
        )


def test_fact_and_judge_prompt_format_is_unambiguous() -> None:
    scenario = Scenario(
        id="example",
        intent="Share the decision",
        key_facts=["Budget is $10.", "Launch is Friday."],
        tone="direct",
        human_reference_email=(
            "Subject: Decision\n\nThe budget and launch are confirmed."
        ),
    )
    assert format_key_facts(scenario.key_facts) == (
        "- Budget is $10.\n- Launch is Friday."
    )
    prompt = build_judge_prompt(scenario, "Subject: Decision\n\nGenerated content.")
    assert "0. Budget is $10." in prompt
    assert "1. Launch is Friday." in prompt
    assert scenario.human_reference_email in prompt
    assert "Generated content." in prompt


def test_scenario_file_contains_ten_valid_unique_scenarios() -> None:
    scenarios = load_scenarios()
    assert len(scenarios) == 10
    assert len({scenario.id for scenario in scenarios}) == 10
    assert all(scenario.key_facts for scenario in scenarios)
    assert all(
        scenario.human_reference_email.startswith("Subject:") for scenario in scenarios
    )


def test_load_scenarios_rejects_wrong_count(workspace_tmp_path) -> None:
    path = workspace_tmp_path / "scenarios.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 10"):
        load_scenarios(path)


@pytest.mark.parametrize("content", ["not json", '{"id": "not-a-list"}'])
def test_load_scenarios_explains_invalid_file_content(
    workspace_tmp_path, content: str
) -> None:
    path = workspace_tmp_path / "scenarios.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_scenarios(path)


def test_load_scenarios_rejects_duplicate_ids(workspace_tmp_path) -> None:
    source = json.loads(
        (workspace_tmp_path.parents[2] / "data" / "scenarios.json").read_text(
            encoding="utf-8"
        )
    )
    source[1]["id"] = source[0]["id"]
    path = workspace_tmp_path / "scenarios.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_scenarios(path)


def test_load_scenarios_wraps_model_validation_errors(workspace_tmp_path) -> None:
    path = workspace_tmp_path / "scenarios.json"
    path.write_text(json.dumps([{"id": "incomplete"}] * 10), encoding="utf-8")
    with pytest.raises(ValueError, match="Scenario validation failed"):
        load_scenarios(path)


def test_advanced_prompt_contains_required_engineering_features() -> None:
    prompt = build_generation_prompt(
        "Follow up",
        ["The deadline is Friday.", "Alex owns the update."],
        "formal",
        Strategy.ADVANCED,
    )
    assert "senior executive communication assistant" in prompt
    assert "Example 1" in prompt and "Example 2" in prompt
    assert "Subject:" in prompt
    assert "Do not invent" in prompt
    assert "chain-of-thought" in prompt
    assert "- The deadline is Friday." in prompt


def test_baseline_prompt_omits_few_shot_and_strict_structure() -> None:
    prompt = build_generation_prompt(
        "Follow up",
        ["The deadline is Friday."],
        "formal",
        Strategy.BASELINE,
    )
    assert "Example 1" not in prompt
    assert "senior executive communication assistant" not in prompt
    assert "exact high-level structure" not in prompt
    assert "The deadline is Friday." in prompt


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(1, 0.0), (2, 0.25), (3, 0.5), (4, 0.75), (5, 1.0)],
)
def test_score_normalization(raw: int, expected: float) -> None:
    assert normalize_five_point_score(raw) == expected


def test_score_normalization_rejects_out_of_range() -> None:
    for raw in (0, 6):
        with pytest.raises(ValueError):
            normalize_five_point_score(raw)
