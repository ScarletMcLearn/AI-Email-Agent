from __future__ import annotations

import csv
import json

import pytest
from src.config import Settings
from src.evaluate import (
    aggregate_results,
    calculate_automated_quality_score,
    calculate_scores,
    evaluate_email,
    run_evaluation,
    validate_judge_result,
    write_evaluation_outputs,
)
from src.models import FactAssessment, JudgeResult, Scenario

from tests.fakes import FakeLLMClient


def judge_result() -> JudgeResult:
    return JudgeResult(
        fact_assessments=[
            FactAssessment(fact_index=0, status="accurate", rationale="Present."),
            FactAssessment(fact_index=1, status="partial", rationale="Incomplete."),
            FactAssessment(fact_index=2, status="missing", rationale="Absent."),
            FactAssessment(fact_index=3, status="contradicted", rationale="Conflicts."),
        ],
        tone_score=4,
        tone_rationale="Mostly matches.",
        professional_quality_score=5,
        professional_quality_rationale="Excellent.",
    )


def scoring_scenario() -> Scenario:
    return Scenario(
        id="scoring",
        intent="Send an update",
        key_facts=["Launch is Friday."],
        tone="formal",
        human_reference_email=(
            "Subject: Update\n\nHello team,\n\nLaunch is Friday.\n\n"
            "Best regards,\n[Your Name]"
        ),
    )


def test_calculate_scores_uses_defined_metric_logic() -> None:
    scenario = scoring_scenario()
    scores = calculate_scores(
        judge_result(),
        scenario,
        scenario.human_reference_email,
    )
    assert scores.fact_coverage_score == pytest.approx(0.375)
    assert scores.tone_match_score == pytest.approx(0.75)
    assert scores.professional_structure_score == pytest.approx(1.0)
    assert scores.professional_placeholder_score == pytest.approx(1.0)
    assert scores.professional_concision_score == pytest.approx(1.0)
    assert scores.professional_quality_automated_score == pytest.approx(1.0)
    assert scores.professional_quality_score == pytest.approx(1.0)
    assert scores.overall_score == pytest.approx((0.375 + 0.75 + 1.0) / 3)


def test_automated_quality_penalizes_extra_placeholders_and_verbosity() -> None:
    scenario = scoring_scenario()
    generated = (
        "Subject: Update\n\nDear [Recipient Name],\n\n"
        + "This is a deliberately verbose sentence. " * 20
        + "\n\nBest regards,\n[Your Name]\n[Your Title]"
    )
    assert calculate_automated_quality_score(scenario, generated) <= 0.6


def test_validate_judge_result_rejects_missing_fact_index() -> None:
    result = judge_result()
    result.fact_assessments.pop()
    with pytest.raises(ValueError, match="exactly"):
        validate_judge_result(result, 4)


@pytest.mark.parametrize("indexes", [[0, 0], [0, 2], [0, 1, 2, 4]])
def test_validate_judge_result_requires_each_zero_based_index_once(
    indexes: list[int],
) -> None:
    result = judge_result()
    result.fact_assessments = [
        FactAssessment(fact_index=index, status="accurate", rationale="Present.")
        for index in indexes
    ]
    with pytest.raises(ValueError, match="exactly"):
        validate_judge_result(result, 4)


def test_calculate_scores_requires_at_least_one_fact() -> None:
    result = judge_result()
    result.fact_assessments = []
    with pytest.raises(ValueError, match="At least one"):
        calculate_scores(
            result,
            scoring_scenario(),
            scoring_scenario().human_reference_email,
        )


def test_evaluate_email_documents_judge_configuration() -> None:
    client = FakeLLMClient()
    settings = Settings(
        api_key=None,
        judge_model="judge-example",
        provider_order=("gemini",),
    )
    scenario = Scenario(
        id="example",
        intent="Send an update",
        key_facts=["Launch is Friday."],
        tone="formal",
        human_reference_email="Subject: Update\n\nLaunch is Friday as planned.",
    )
    result, scores, metadata = evaluate_email(
        client, settings, scenario, "ADVANCED-EMAIL"
    )
    assert result.fact_assessments[0].fact_index == 0
    assert scores.overall_score == pytest.approx(17 / 18)
    assert metadata.provider == "gemini"


def test_aggregate_excludes_failed_records() -> None:
    records = [
        {
            "strategy": "A",
            "scores": {
                "fact_coverage_score": 1.0,
                "tone_match_score": 0.5,
                "professional_quality_score": 0.75,
            },
        },
        {"strategy": "A", "scores": None},
        {
            "strategy": "B",
            "scores": {
                "fact_coverage_score": 0.5,
                "tone_match_score": 0.25,
                "professional_quality_score": 0.5,
            },
        },
    ]
    summary = aggregate_results(records)
    assert summary["A"]["evaluated_count"] == 1
    assert summary["A"]["error_count"] == 1
    assert summary["A"]["overall_average_score"] == pytest.approx(0.75)


def test_aggregate_documents_empty_and_expected_count_behavior() -> None:
    summary = aggregate_results([], expected_per_strategy=10)
    for strategy in ("A", "B"):
        assert summary[strategy]["recorded_count"] == 0
        assert summary[strategy]["evaluated_count"] == 0
        assert summary[strategy]["error_count"] == 10
        assert summary[strategy]["overall_average_score"] is None


def test_offline_end_to_end_evaluation_writes_all_outputs(workspace_tmp_path) -> None:
    client = FakeLLMClient()
    settings = Settings(
        api_key=None,
        generation_model="fake-model",
        judge_model="fake-judge",
        provider_order=("gemini",),
    )
    result = run_evaluation(client, settings, output_dir=workspace_tmp_path)

    assert len(result["records"]) == 20
    assert result["strategy_summary"]["A"]["evaluated_count"] == 10
    assert result["strategy_summary"]["B"]["evaluated_count"] == 10
    assert (
        result["strategy_summary"]["A"]["overall_average_score"]
        > result["strategy_summary"]["B"]["overall_average_score"]
    )
    assert (workspace_tmp_path / "generated_emails.json").exists()
    assert (workspace_tmp_path / "evaluation_results.json").exists()
    assert (workspace_tmp_path / "evaluation_results.csv").exists()

    payload = json.loads(
        (workspace_tmp_path / "evaluation_results.json").read_text(encoding="utf-8")
    )
    assert len(payload["records"]) == 20
    assert payload["records"][0]["generation_provider"] == "gemini"
    assert payload["records"][0]["judge_provider"] == "gemini"
    with (workspace_tmp_path / "evaluation_results.csv").open(
        encoding="utf-8-sig", newline=""
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 20
    assert rows[0]["generation_provider"] == "gemini"
    assert rows[0]["judge_provider"] == "gemini"


def test_evaluation_continues_after_individual_generation_failure(
    workspace_tmp_path,
) -> None:
    client = FakeLLMClient(fail_baseline_once=True)
    settings = Settings(
        api_key=None,
        generation_model="fake",
        judge_model="fake",
        provider_order=("gemini",),
    )
    result = run_evaluation(client, settings, output_dir=workspace_tmp_path)
    assert len(result["records"]) == 20
    assert (
        sum(record["generation_error"] is not None for record in result["records"]) == 1
    )
    assert sum(record["scores"] is not None for record in result["records"]) == 19


def test_resume_reuses_completed_records_and_failed_generation(
    workspace_tmp_path,
) -> None:
    settings = Settings(
        api_key=None,
        generation_model="fake",
        judge_model="fake",
        min_request_interval_seconds=0,
        provider_order=("gemini",),
    )
    first_client = FakeLLMClient(fail_baseline_once=True)
    first = run_evaluation(first_client, settings, output_dir=workspace_tmp_path)
    assert sum(record["scores"] is not None for record in first["records"]) == 19

    resumed_client = FakeLLMClient()
    resumed = run_evaluation(
        resumed_client,
        settings,
        output_dir=workspace_tmp_path,
        resume=True,
    )
    assert sum(record["scores"] is not None for record in resumed["records"]) == 20
    assert resumed_client.text_calls == 1
    assert resumed_client.structured_calls == 1


def test_resume_checkpoint_keeps_later_completed_records(workspace_tmp_path) -> None:
    settings = Settings(
        api_key=None,
        generation_model="fake",
        judge_model="fake",
        provider_order=("gemini",),
    )
    complete = run_evaluation(FakeLLMClient(), settings, output_dir=workspace_tmp_path)
    payload_path = workspace_tmp_path / "evaluation_results.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["records"][0]["scores"] = None
    payload["records"][0]["evaluation_error"] = "retry this record"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    resumed_client = FakeLLMClient()
    resumed = run_evaluation(
        resumed_client,
        settings,
        output_dir=workspace_tmp_path,
        resume=True,
    )
    assert len(resumed["records"]) == len(complete["records"]) == 20
    assert all(record["scores"] is not None for record in resumed["records"])


def test_resume_with_new_judge_reuses_generation_and_rejudges_all(
    workspace_tmp_path,
) -> None:
    original_settings = Settings(
        api_key=None,
        generation_model="fake-generator",
        judge_model="old-judge",
        provider_order=("gemini",),
    )
    run_evaluation(FakeLLMClient(), original_settings, output_dir=workspace_tmp_path)

    resumed_client = FakeLLMClient()
    new_settings = Settings(
        api_key=None,
        generation_model="fake-generator",
        judge_model="new-judge",
        provider_order=("gemini",),
    )
    resumed = run_evaluation(
        resumed_client,
        new_settings,
        output_dir=workspace_tmp_path,
        resume=True,
    )
    assert resumed_client.text_calls == 0
    assert resumed_client.structured_calls == 20
    assert all(record["judge_model"] == "new-judge" for record in resumed["records"])


def test_resume_reruns_records_from_a_different_generation_model(
    workspace_tmp_path,
) -> None:
    original = Settings(
        api_key=None,
        generation_model="old",
        judge_model="judge",
        provider_order=("gemini",),
    )
    run_evaluation(FakeLLMClient(), original, output_dir=workspace_tmp_path)
    changed = Settings(
        api_key=None,
        generation_model="new",
        judge_model="judge",
        provider_order=("gemini",),
    )
    resumed_client = FakeLLMClient()
    resumed = run_evaluation(
        resumed_client, changed, output_dir=workspace_tmp_path, resume=True
    )
    assert resumed_client.text_calls == 20
    assert resumed_client.structured_calls == 20
    assert all(record["generation_model"] == "new" for record in resumed["records"])


class QuotaClient(FakeLLMClient):
    def generate_text(self, prompt: str, *, model: str, temperature: float) -> str:
        from src.llm_client import DailyQuotaExceeded

        raise DailyQuotaExceeded("daily quota")


class JudgeQuotaClient(FakeLLMClient):
    def generate_structured(self, *args: object, **kwargs: object) -> object:
        from src.llm_client import DailyQuotaExceeded

        raise DailyQuotaExceeded("judge daily quota")


def test_daily_quota_stops_and_persists_a_resumable_checkpoint(
    workspace_tmp_path,
) -> None:
    result = run_evaluation(
        QuotaClient(),
        Settings(
            api_key=None,
            generation_model="fake",
            judge_model="fake",
            provider_order=("gemini",),
        ),
        output_dir=workspace_tmp_path,
    )
    assert result["run_status"] == "stopped_daily_quota"
    assert len(result["records"]) == 1
    assert (workspace_tmp_path / "evaluation_results.json").exists()


def test_judge_daily_quota_preserves_the_generated_email(workspace_tmp_path) -> None:
    result = run_evaluation(
        JudgeQuotaClient(),
        Settings(
            api_key=None,
            generation_model="fake",
            judge_model="fake",
            provider_order=("gemini",),
        ),
        output_dir=workspace_tmp_path,
    )
    assert result["run_status"] == "stopped_daily_quota"
    assert result["records"][0]["generated_email"]
    assert result["records"][0]["evaluation_error"] == "judge daily quota"


def test_output_writer_serializes_failed_records_to_json_and_csv(
    workspace_tmp_path,
) -> None:
    result = {
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "generation_model": "generator",
        "records": [
            {
                "scenario_id": "example",
                "strategy": "A",
                "strategy_label": "Strategy A (advanced)",
                "intent": "Send update",
                "tone": "formal",
                "generated_email": None,
                "generation_provider": None,
                "generation_model": "generator",
                "generation_attempts": None,
                "generation_fallback_used": False,
                "judge_provider": None,
                "judge_model": "judge",
                "judge_attempts": None,
                "judge_fallback_used": False,
                "generation_error": "failed",
                "evaluation_error": None,
                "scores": None,
                "judge_result": None,
            }
        ],
    }
    write_evaluation_outputs(result, workspace_tmp_path)
    generated = json.loads(
        (workspace_tmp_path / "generated_emails.json").read_text(encoding="utf-8")
    )
    assert generated["emails"][0]["error"] == "failed"
    with (workspace_tmp_path / "evaluation_results.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        row = next(csv.DictReader(stream))
    assert row["fact_coverage_score"] == ""
    assert row["generation_error"] == "failed"


class FallbackClient(FakeLLMClient):
    def generate_text(self, prompt: str, *, model: str, temperature: float):
        response = super().generate_text(
            prompt, model="gemini-fallback-generation", temperature=temperature
        )
        return response.__class__(
            response.value,
            response.metadata.__class__(
                "gemini",
                "gemini-fallback-generation",
                4,
                fallback_used=True,
            ),
        )

    def generate_structured(self, *args: object, **kwargs: object):
        kwargs["model"] = "gemini-fallback-judge"
        response = super().generate_structured(*args, **kwargs)
        return response.__class__(
            response.value,
            response.metadata.__class__(
                "gemini",
                "gemini-fallback-judge",
                4,
                fallback_used=True,
            ),
        )


def test_fallback_calls_mark_run_and_records_as_mixed_provider(
    workspace_tmp_path,
) -> None:
    settings = Settings(
        api_key=None,
        generation_model="gemini-fallback-generation",
        judge_model="gemini-fallback-judge",
        groq_generation_model="groq-generation",
        groq_judge_model="groq-judge",
    )
    result = run_evaluation(
        FallbackClient(),
        settings,
        output_dir=workspace_tmp_path,
    )
    assert result["run_status"] == "completed_with_fallback"
    assert result["fallback_used"] is True
    assert "not a fully controlled" in result["comparison_warning"]
    assert all(record["generation_fallback_used"] for record in result["records"])
    assert all(record["judge_fallback_used"] for record in result["records"])


def test_resume_infers_legacy_gemini_metadata(workspace_tmp_path) -> None:
    settings = Settings(
        api_key=None,
        generation_model="legacy-generator",
        judge_model="legacy-judge",
        provider_order=("gemini",),
    )
    run_evaluation(FakeLLMClient(), settings, output_dir=workspace_tmp_path)
    payload_path = workspace_tmp_path / "evaluation_results.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    for record in payload["records"]:
        for field in (
            "generation_provider",
            "generation_attempts",
            "generation_fallback_used",
            "judge_provider",
            "judge_attempts",
            "judge_fallback_used",
        ):
            record.pop(field, None)
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    resumed_client = FakeLLMClient()
    resumed = run_evaluation(
        resumed_client,
        settings,
        output_dir=workspace_tmp_path,
        resume=True,
    )
    assert resumed_client.text_calls == 0
    assert resumed_client.structured_calls == 0
    assert all(
        record["generation_provider"] == "gemini" for record in resumed["records"]
    )
