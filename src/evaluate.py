"""Run the controlled Strategy A versus Strategy B evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import ValidationError

from src.config import (
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_SCENARIOS_PATH,
    ConfigurationError,
    Settings,
)
from src.generator import EmailGenerator
from src.llm_client import (
    CallMetadata,
    DailyQuotaExceeded,
    LLMClient,
    LLMError,
    create_llm_client,
)
from src.models import EvaluationScores, JudgeResult, Scenario, Strategy
from src.prompts import build_judge_prompt

LOGGER = logging.getLogger(__name__)
SCORING_VERSION = 3

METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "fact_coverage": {
        "name": "Fact Coverage Score",
        "definition": "Measures whether every required key fact is included accurately.",
        "logic": (
            "The LLM judge labels each fact accurate, partial, missing, or contradicted. "
            "Python assigns credits of 1.0, 0.5, 0.0, and 0.0 respectively, then "
            "averages the credits across all required facts."
        ),
    },
    "tone_match": {
        "name": "Tone Match Score",
        "definition": "Measures how consistently the email matches the requested tone.",
        "logic": (
            "The LLM judge applies a 1-5 rubric, where 1 is completely wrong and "
            "5 is a strong match. Python normalizes the score with (raw - 1) / 4."
        ),
    },
    "professional_quality": {
        "name": "Professional Email Quality Score",
        "definition": (
            "Measures professionalism, clarity, concision, formatting, and fluency."
        ),
        "logic": (
            "Python averages two components: the normalized 1-5 LLM quality rubric "
            "and an automated email-quality score. The automated score equally "
            "weights structural completeness (subject, greeting, closing), "
            "placeholder discipline, and concision relative to the human reference. "
            "Extra bracketed placeholders beyond [Your Name] reduce the placeholder "
            "component by 0.1 each, to a 0.5 floor. Concision receives 1.0, 0.75, "
            "0.5, 0.25, or 0.0 when generated/reference word-count ratio is at most "
            "1.25, 1.5, 1.75, 2.0, or above 2.0 respectively."
        ),
    },
}

FACT_CREDITS = {
    "accurate": 1.0,
    "partial": 0.5,
    "missing": 0.0,
    "contradicted": 0.0,
}


def load_scenarios(path: Path | str = DEFAULT_SCENARIOS_PATH) -> list[Scenario]:
    """Load and validate exactly 10 unique evaluation scenarios."""
    scenario_path = Path(path)
    try:
        payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not load scenarios from {scenario_path}: {exc}"
        ) from exc

    if not isinstance(payload, list):
        raise ValueError("Scenario file must contain a JSON list.")
    try:
        scenarios = [Scenario.model_validate(item) for item in payload]
    except ValidationError as exc:
        raise ValueError(f"Scenario validation failed: {exc}") from exc
    if len(scenarios) != 10:
        raise ValueError(f"Expected exactly 10 scenarios, found {len(scenarios)}.")
    ids = [scenario.id for scenario in scenarios]
    if len(set(ids)) != len(ids):
        raise ValueError("Scenario IDs must be unique.")
    return scenarios


def normalize_five_point_score(raw_score: int) -> float:
    """Normalize a 1-5 rubric score to 0-1."""
    if raw_score < 1 or raw_score > 5:
        raise ValueError("Raw rubric score must be between 1 and 5.")
    return (raw_score - 1) / 4


def validate_judge_result(result: JudgeResult, fact_count: int) -> JudgeResult:
    """Ensure the judge returned exactly one assessment for every fact."""
    indexes = [assessment.fact_index for assessment in result.fact_assessments]
    expected = list(range(fact_count))
    if sorted(indexes) != expected or len(indexes) != len(set(indexes)):
        raise ValueError(
            f"Judge fact indexes must be exactly {expected}; received {indexes}."
        )
    return result


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+[\w'-]*\b", text))


def calculate_automated_quality_components(
    scenario: Scenario,
    generated_email: str,
) -> tuple[float, float, float]:
    """Score structure, placeholder discipline, and reference-relative concision."""
    lines = [line.strip() for line in generated_email.splitlines() if line.strip()]
    has_subject = any(line.lower().startswith("subject:") for line in lines[:2])
    has_greeting = any(
        re.match(
            r"^(dear|hello|hi|good (morning|afternoon|evening))\b",
            line,
            flags=re.IGNORECASE,
        )
        for line in lines
    )
    has_closing = any(
        re.match(
            r"^(best regards|regards|sincerely|kind regards|warm regards|"
            r"professional regards|thank you)[,.]?$",
            line,
            flags=re.IGNORECASE,
        )
        for line in lines
    )
    structure_score = mean([has_subject, has_greeting, has_closing])

    extra_placeholders = [
        placeholder
        for placeholder in re.findall(r"\[[^\]]+\]", generated_email)
        if placeholder.casefold() != "[your name]"
    ]
    placeholder_score = max(0.5, 1.0 - 0.1 * len(extra_placeholders))

    reference_words = max(1, _word_count(scenario.human_reference_email))
    word_ratio = _word_count(generated_email) / reference_words
    if word_ratio <= 1.25:
        concision_score = 1.0
    elif word_ratio <= 1.5:
        concision_score = 0.75
    elif word_ratio <= 1.75:
        concision_score = 0.5
    elif word_ratio <= 2.0:
        concision_score = 0.25
    else:
        concision_score = 0.0

    return structure_score, placeholder_score, concision_score


def calculate_automated_quality_score(
    scenario: Scenario,
    generated_email: str,
) -> float:
    """Return the mean of the deterministic professional-quality components."""
    return mean(calculate_automated_quality_components(scenario, generated_email))


def calculate_scores(
    judge_result: JudgeResult,
    scenario: Scenario,
    generated_email: str,
) -> EvaluationScores:
    """Calculate the three normalized metrics and per-email overall score."""
    if not judge_result.fact_assessments:
        raise ValueError("At least one fact assessment is required.")
    fact_score = mean(
        FACT_CREDITS[assessment.status] for assessment in judge_result.fact_assessments
    )
    tone_score = normalize_five_point_score(judge_result.tone_score)
    quality_judge_score = normalize_five_point_score(
        judge_result.professional_quality_score
    )
    (
        quality_structure_score,
        quality_placeholder_score,
        quality_concision_score,
    ) = calculate_automated_quality_components(
        scenario,
        generated_email,
    )
    quality_automated_score = mean(
        [
            quality_structure_score,
            quality_placeholder_score,
            quality_concision_score,
        ]
    )
    quality_score = mean([quality_judge_score, quality_automated_score])
    return EvaluationScores(
        fact_coverage_score=fact_score,
        tone_raw_score=judge_result.tone_score,
        tone_match_score=tone_score,
        professional_quality_raw_score=judge_result.professional_quality_score,
        professional_structure_score=quality_structure_score,
        professional_placeholder_score=quality_placeholder_score,
        professional_concision_score=quality_concision_score,
        professional_quality_automated_score=quality_automated_score,
        professional_quality_score=quality_score,
        overall_score=mean([fact_score, tone_score, quality_score]),
    )


def evaluate_email(
    client: LLMClient,
    settings: Settings,
    scenario: Scenario,
    generated_email: str,
) -> tuple[JudgeResult, EvaluationScores, CallMetadata]:
    """Judge one generated email and calculate its metrics."""
    prompt = build_judge_prompt(scenario, generated_email)
    response = client.generate_structured(
        prompt,
        model=settings.primary_judge_model,
        temperature=settings.judge_temperature,
        response_model=JudgeResult,
        validator=lambda result: validate_judge_result(result, len(scenario.key_facts)),
    )
    return (
        response.value,
        calculate_scores(response.value, scenario, generated_email),
        response.metadata,
    )


def _empty_record(
    scenario: Scenario,
    strategy: Strategy,
    settings: Settings,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario.id,
        "intent": scenario.intent,
        "tone": scenario.tone,
        "key_facts": scenario.key_facts,
        "strategy": strategy.value,
        "strategy_label": strategy.label,
        "generation_provider": None,
        "generation_model": settings.primary_generation_model,
        "generation_attempts": None,
        "generation_fallback_used": False,
        "judge_provider": None,
        "judge_model": settings.primary_judge_model,
        "judge_attempts": None,
        "judge_fallback_used": False,
        "generated_email": None,
        "generation_error": None,
        "judge_result": None,
        "scores": None,
        "scoring_version": SCORING_VERSION,
        "evaluation_error": None,
    }


def _apply_metadata(
    record: dict[str, Any],
    metadata: CallMetadata,
    *,
    judge: bool,
) -> None:
    prefix = "judge" if judge else "generation"
    record[f"{prefix}_provider"] = metadata.provider
    record[f"{prefix}_model"] = metadata.model
    record[f"{prefix}_attempts"] = metadata.attempts
    record[f"{prefix}_fallback_used"] = metadata.fallback_used


def _normalize_legacy_record(
    record: dict[str, Any],
    *,
    saved_generation_model: str | None,
    saved_judge_model: str | None,
) -> None:
    """Infer Gemini metadata for checkpoints written before provider routing."""
    if record.get("generated_email") and not record.get("generation_provider"):
        record["generation_provider"] = "gemini"
        record["generation_model"] = (
            record.get("generation_model") or saved_generation_model
        )
        record.setdefault("generation_attempts", None)
        record.setdefault("generation_fallback_used", False)
    if record.get("judge_result") and not record.get("judge_provider"):
        record["judge_provider"] = "gemini"
        record["judge_model"] = record.get("judge_model") or saved_judge_model
        record.setdefault("judge_attempts", None)
        record.setdefault("judge_fallback_used", False)


def _generation_is_reusable(record: dict[str, Any], settings: Settings) -> bool:
    provider = record.get("generation_provider")
    model = record.get("generation_model")
    return bool(
        record.get("generated_email")
        and provider
        and model
        and settings.is_configured_model(provider, model, judge=False)
    )


def _judge_is_reusable(record: dict[str, Any], settings: Settings) -> bool:
    provider = record.get("judge_provider")
    model = record.get("judge_model")
    return bool(
        record.get("scores") is not None
        and provider
        and model
        and settings.is_configured_model(provider, model, judge=True)
    )


def aggregate_results(
    records: Iterable[dict[str, Any]],
    *,
    expected_per_strategy: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate valid metric scores separately for each strategy."""
    records_list = list(records)
    summary: dict[str, dict[str, Any]] = {}
    for strategy in Strategy:
        matching = [
            record for record in records_list if record["strategy"] == strategy.value
        ]
        valid = [record for record in matching if record.get("scores") is not None]
        metric_averages: dict[str, float | None] = {
            "fact_coverage_score": None,
            "tone_match_score": None,
            "professional_quality_score": None,
        }
        if valid:
            for metric in metric_averages:
                metric_averages[metric] = mean(
                    record["scores"][metric] for record in valid
                )
        available = [value for value in metric_averages.values() if value is not None]
        summary[strategy.value] = {
            "strategy_label": strategy.label,
            "total_scenarios": (
                expected_per_strategy
                if expected_per_strategy is not None
                else len(matching)
            ),
            "recorded_count": len(matching),
            "evaluated_count": len(valid),
            "error_count": (
                expected_per_strategy - len(valid)
                if expected_per_strategy is not None
                else len(matching) - len(valid)
            ),
            "metric_averages": metric_averages,
            "overall_average_score": mean(available) if available else None,
        }
    return summary


def run_evaluation(
    client: LLMClient,
    settings: Settings,
    *,
    scenarios_path: Path | str = DEFAULT_SCENARIOS_PATH,
    output_dir: Path | str = DEFAULT_OUTPUTS_DIR,
    resume: bool = False,
) -> dict[str, Any]:
    """Generate, judge, aggregate, and persist the full comparison."""
    scenarios = load_scenarios(scenarios_path)
    scenarios_by_id = {scenario.id: scenario for scenario in scenarios}
    generator = EmailGenerator(client, settings)
    destination = Path(output_dir)
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    existing_records: dict[tuple[str, str], dict[str, Any]] = {}
    ordered_keys = [
        (scenario.id, strategy.value) for scenario in scenarios for strategy in Strategy
    ]

    if resume and (destination / "evaluation_results.json").exists():
        existing = json.loads(
            (destination / "evaluation_results.json").read_text(encoding="utf-8")
        )
        existing_records = {
            (record["scenario_id"], record["strategy"]): record
            for record in existing.get("records", [])
        }
        for record in existing_records.values():
            _normalize_legacy_record(
                record,
                saved_generation_model=existing.get("generation_model"),
                saved_judge_model=existing.get("judge_model"),
            )
            if record.get("generation_provider"):
                record["generation_fallback_used"] = bool(
                    record.get("generation_fallback_used")
                    or record["generation_provider"] != settings.primary_provider
                )
            if record.get("judge_provider"):
                record["judge_fallback_used"] = bool(
                    record.get("judge_fallback_used")
                    or record["judge_provider"] != settings.primary_provider
                )
            if not _generation_is_reusable(record, settings):
                record["generated_email"] = None
                record["generation_provider"] = None
                record["generation_model"] = settings.primary_generation_model
                record["generation_attempts"] = None
                record["generation_fallback_used"] = False
                record["generation_error"] = None
                record["judge_provider"] = None
                record["judge_model"] = settings.primary_judge_model
                record["judge_attempts"] = None
                record["judge_fallback_used"] = False
                record["judge_result"] = None
                record["scores"] = None
                record["evaluation_error"] = None
                continue
            if not _judge_is_reusable(record, settings):
                record["judge_provider"] = None
                record["judge_model"] = settings.primary_judge_model
                record["judge_attempts"] = None
                record["judge_fallback_used"] = False
                record["judge_result"] = None
                record["scores"] = None
                record["evaluation_error"] = None
            elif record.get("scoring_version") != SCORING_VERSION:
                judge_result = JudgeResult.model_validate(record["judge_result"])
                scenario = scenarios_by_id[record["scenario_id"]]
                scores = calculate_scores(
                    judge_result,
                    scenario,
                    record["generated_email"],
                )
                record["scores"] = scores.model_dump()
                record["scoring_version"] = SCORING_VERSION
        generated_at_utc = existing.get("generated_at_utc", generated_at_utc)
        LOGGER.info("Loaded %s saved records for resume.", len(existing_records))

    records_by_key = dict(existing_records)

    def current_result(
        *,
        run_status: str = "in_progress",
        stopped_early_reason: str | None = None,
    ) -> dict[str, Any]:
        ordered_records = [
            records_by_key[key] for key in ordered_keys if key in records_by_key
        ]
        fallback_used = any(
            record.get("generation_fallback_used")
            or record.get("judge_fallback_used")
            or (
                record.get("generation_provider")
                and record.get("generation_provider") != settings.primary_provider
            )
            or (
                record.get("judge_provider")
                and record.get("judge_provider") != settings.primary_provider
            )
            for record in ordered_records
        )
        effective_status = run_status
        if run_status == "completed" and fallback_used:
            effective_status = "completed_with_fallback"
        return {
            "project_title": "Email Generation Assistant",
            "scoring_version": SCORING_VERSION,
            "generated_at_utc": generated_at_utc,
            "run_status": effective_status,
            "stopped_early_reason": stopped_early_reason,
            "provider_order": list(settings.provider_order),
            "generation_provider": settings.primary_provider,
            "generation_model": settings.primary_generation_model,
            "judge_provider": settings.primary_provider,
            "judge_model": settings.primary_judge_model,
            "fallback_used": fallback_used,
            "comparison_warning": (
                "One or more calls used a fallback provider/model. The report "
                "contains mixed-provider results and is not a fully controlled "
                "single-model comparison."
                if fallback_used
                else None
            ),
            "generation_temperature": settings.generation_temperature,
            "judge_temperature": settings.judge_temperature,
            "metric_definitions": METRIC_DEFINITIONS,
            "records": ordered_records,
            "strategy_summary": aggregate_results(
                ordered_records,
                expected_per_strategy=len(scenarios),
            ),
        }

    for scenario_number, scenario in enumerate(scenarios, start=1):
        for strategy in Strategy:
            key = (scenario.id, strategy.value)
            existing_record = records_by_key.get(key)
            if existing_record and existing_record.get("scores") is not None:
                LOGGER.info(
                    "[%s/10] Reusing completed %s / Strategy %s",
                    scenario_number,
                    scenario.id,
                    strategy.value,
                )
                continue

            LOGGER.info(
                "[%s/10] Generating %s with %s",
                scenario_number,
                scenario.id,
                strategy.label,
            )
            record = _empty_record(scenario, strategy, settings)
            if existing_record and existing_record.get("generated_email"):
                email = existing_record["generated_email"]
                for field in (
                    "generated_email",
                    "generation_provider",
                    "generation_model",
                    "generation_attempts",
                    "generation_fallback_used",
                ):
                    record[field] = existing_record.get(field)
                LOGGER.info(
                    "Reusing generated email for %s / Strategy %s.",
                    scenario.id,
                    strategy.value,
                )
            else:
                try:
                    generation = generator.generate_scenario_with_metadata(
                        scenario, strategy
                    )
                    email = generation.value
                    record["generated_email"] = email
                    _apply_metadata(record, generation.metadata, judge=False)
                except DailyQuotaExceeded as exc:
                    record["generation_error"] = str(exc)
                    records_by_key[key] = record
                    result = current_result(
                        run_status="stopped_daily_quota",
                        stopped_early_reason=str(exc),
                    )
                    write_evaluation_outputs(result, destination)
                    LOGGER.error("%s", exc)
                    return result
                except Exception as exc:
                    record["generation_error"] = str(exc)
                    LOGGER.error(
                        "Generation failed for %s / %s: %s",
                        scenario.id,
                        strategy.value,
                        exc,
                    )
                    records_by_key[key] = record
                    write_evaluation_outputs(current_result(), destination)
                    continue

            try:
                judge_result, scores, judge_metadata = evaluate_email(
                    client, settings, scenario, email
                )
                record["judge_result"] = judge_result.model_dump()
                record["scores"] = scores.model_dump()
                record["scoring_version"] = SCORING_VERSION
                _apply_metadata(record, judge_metadata, judge=True)
            except DailyQuotaExceeded as exc:
                record["evaluation_error"] = str(exc)
                records_by_key[key] = record
                result = current_result(
                    run_status="stopped_daily_quota",
                    stopped_early_reason=str(exc),
                )
                write_evaluation_outputs(result, destination)
                LOGGER.error("%s", exc)
                return result
            except Exception as exc:
                record["evaluation_error"] = str(exc)
                LOGGER.error(
                    "Evaluation failed for %s / %s: %s",
                    scenario.id,
                    strategy.value,
                    exc,
                )
            records_by_key[key] = record
            write_evaluation_outputs(current_result(), destination)

    result = current_result(run_status="completed")
    write_evaluation_outputs(result, destination)
    return result


def write_evaluation_outputs(
    result: dict[str, Any],
    output_dir: Path | str = DEFAULT_OUTPUTS_DIR,
) -> None:
    """Write generated emails, complete JSON results, and flat CSV results."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    generated_payload = {
        "generated_at_utc": result["generated_at_utc"],
        "provider_order": result.get("provider_order", []),
        "generation_provider": result.get("generation_provider"),
        "generation_model": result["generation_model"],
        "fallback_used": result.get("fallback_used", False),
        "emails": [
            {
                "scenario_id": record["scenario_id"],
                "strategy": record["strategy"],
                "strategy_label": record["strategy_label"],
                "provider": record.get("generation_provider"),
                "model": record.get("generation_model"),
                "attempts": record.get("generation_attempts"),
                "fallback_used": record.get("generation_fallback_used", False),
                "email": record["generated_email"],
                "error": record["generation_error"],
            }
            for record in result["records"]
        ],
    }
    (destination / "generated_emails.json").write_text(
        json.dumps(generated_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (destination / "evaluation_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fieldnames = [
        "scenario_id",
        "strategy",
        "strategy_label",
        "intent",
        "tone",
        "generation_provider",
        "generation_model",
        "generation_attempts",
        "generation_fallback_used",
        "judge_provider",
        "judge_model",
        "judge_attempts",
        "judge_fallback_used",
        "fact_coverage_score",
        "tone_raw_score",
        "tone_match_score",
        "professional_quality_raw_score",
        "professional_structure_score",
        "professional_placeholder_score",
        "professional_concision_score",
        "professional_quality_automated_score",
        "professional_quality_score",
        "overall_score",
        "fact_assessments",
        "generation_error",
        "evaluation_error",
    ]
    with (destination / "evaluation_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in result["records"]:
            scores = record.get("scores") or {}
            judge_result = record.get("judge_result") or {}
            writer.writerow(
                {
                    "scenario_id": record["scenario_id"],
                    "strategy": record["strategy"],
                    "strategy_label": record["strategy_label"],
                    "intent": record["intent"],
                    "tone": record["tone"],
                    "generation_provider": record.get("generation_provider"),
                    "generation_model": record.get("generation_model"),
                    "generation_attempts": record.get("generation_attempts"),
                    "generation_fallback_used": record.get(
                        "generation_fallback_used", False
                    ),
                    "judge_provider": record.get("judge_provider"),
                    "judge_model": record.get("judge_model"),
                    "judge_attempts": record.get("judge_attempts"),
                    "judge_fallback_used": record.get("judge_fallback_used", False),
                    "fact_coverage_score": scores.get("fact_coverage_score"),
                    "tone_raw_score": scores.get("tone_raw_score"),
                    "tone_match_score": scores.get("tone_match_score"),
                    "professional_quality_raw_score": scores.get(
                        "professional_quality_raw_score"
                    ),
                    "professional_structure_score": scores.get(
                        "professional_structure_score"
                    ),
                    "professional_placeholder_score": scores.get(
                        "professional_placeholder_score"
                    ),
                    "professional_concision_score": scores.get(
                        "professional_concision_score"
                    ),
                    "professional_quality_automated_score": scores.get(
                        "professional_quality_automated_score"
                    ),
                    "professional_quality_score": scores.get(
                        "professional_quality_score"
                    ),
                    "overall_score": scores.get("overall_score"),
                    "fact_assessments": json.dumps(
                        judge_result.get("fact_assessments", []),
                        ensure_ascii=False,
                    ),
                    "generation_error": record.get("generation_error"),
                    "evaluation_error": record.get("evaluation_error"),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate both email prompting strategies with hosted LLMs."
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=DEFAULT_SCENARIOS_PATH,
        help="Path to the 10-scenario JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Directory for generated evaluation artifacts.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed records and generated emails from a previous run.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        settings = Settings.from_env()
        client = create_llm_client(settings)
        result = run_evaluation(
            client,
            settings,
            scenarios_path=args.scenarios,
            output_dir=args.output_dir,
            resume=args.resume,
        )
        for strategy, summary in result["strategy_summary"].items():
            LOGGER.info(
                "Strategy %s: %s/%s evaluated, overall=%s",
                strategy,
                summary["evaluated_count"],
                summary["total_scenarios"],
                (
                    f"{summary['overall_average_score']:.3f}"
                    if summary["overall_average_score"] is not None
                    else "n/a"
                ),
            )
        if result.get("run_status") == "stopped_daily_quota":
            LOGGER.error(
                "Evaluation checkpointed and stopped because a daily model quota "
                "was exhausted. Run `pixi run evaluate-resume` after quota reset."
            )
            return 2
        if result.get("run_status") == "completed_with_fallback":
            LOGGER.warning("%s", result.get("comparison_warning"))
        LOGGER.info("Evaluation outputs written to %s", args.output_dir)
        return 0
    except (ConfigurationError, LLMError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
