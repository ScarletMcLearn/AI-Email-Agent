from __future__ import annotations

import sys
from argparse import Namespace

import pytest
from src import cli, evaluate, report
from src.cli import collect_key_facts, run_interactive
from src.config import Settings
from src.generator import EmailGenerator
from src.report import (
    _analysis,
    _compact_error,
    _format_score,
    generate_reports,
    load_evaluation_results,
    validate_complete_evaluation,
)

from tests.fakes import FakeLLMClient
from tests.test_evaluate import FallbackClient


def test_cli_smoke() -> None:
    answers = iter(
        [
            "Follow up after a meeting",
            "The deadline is Friday.",
            "Alex owns the update.",
            "",
            "formal",
            "A",
        ]
    )
    output: list[str] = []
    generator = EmailGenerator(
        FakeLLMClient(),
        Settings(
            api_key=None,
            generation_model="fake",
            judge_model="fake",
            provider_order=("gemini",),
        ),
    )
    email = run_interactive(
        generator,
        input_fn=lambda prompt: next(answers),
        print_fn=output.append,
    )
    assert "ADVANCED-EMAIL" in email
    assert any("Generated email" in line for line in output)


def test_collect_key_facts_strips_each_line_and_documents_prompts() -> None:
    answers = iter([" First fact. ", "Second fact.", ""])
    output: list[str] = []
    facts = collect_key_facts(lambda _: next(answers), output.append)
    assert facts == ["First fact.", "Second fact."]
    assert output == [
        "Enter key facts one per line. Press Enter on a blank line when finished."
    ]


def test_cli_defaults_to_advanced_strategy() -> None:
    answers = iter(["Send update", "Launch Friday", "", "formal", ""])
    generator = EmailGenerator(
        FakeLLMClient(),
        Settings(api_key=None, generation_model="fake", provider_order=("gemini",)),
    )
    assert "ADVANCED-EMAIL" in run_interactive(
        generator, input_fn=lambda _: next(answers), print_fn=lambda _: None
    )


def test_cli_propagates_invalid_strategy() -> None:
    answers = iter(["Send update", "Launch Friday", "", "formal", "C"])
    generator = EmailGenerator(
        FakeLLMClient(),
        Settings(api_key=None, generation_model="fake", provider_order=("gemini",)),
    )
    with pytest.raises(ValueError, match="Strategy"):
        run_interactive(
            generator, input_fn=lambda _: next(answers), print_fn=lambda _: None
        )


def test_markdown_and_pdf_report_generation(workspace_tmp_path) -> None:
    from src.evaluate import run_evaluation

    results_dir = workspace_tmp_path / "results"
    reports_dir = workspace_tmp_path / "reports"
    settings = Settings(
        api_key=None,
        generation_model="fake",
        judge_model="fake",
        provider_order=("gemini",),
    )
    run_evaluation(FakeLLMClient(), settings, output_dir=results_dir)

    markdown_path, pdf_path = generate_reports(
        results_dir / "evaluation_results.json",
        reports_dir / "final_report.md",
        reports_dir / "final_report.pdf",
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Prompt Template - Strategy A" in markdown
    assert "Fact Coverage Score" in markdown
    assert "Comparative Analysis" in markdown
    assert "gemini / fake" in markdown
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert pdf_path.stat().st_size > 5_000


def test_report_does_not_claim_a_winner_on_tied_scores() -> None:
    data = {
        "strategy_summary": {
            "A": {"overall_average_score": 0.75, "metric_averages": {}},
            "B": {"overall_average_score": 0.75, "metric_averages": {}},
        }
    }
    analysis = _analysis(data)
    assert "tied" in analysis["winner"]
    assert "Do not make" in analysis["recommendation"]


def test_report_analysis_handles_missing_results_and_clear_winner() -> None:
    unavailable = _analysis(
        {
            "strategy_summary": {
                "A": {"overall_average_score": None},
                "B": {"overall_average_score": 0.5},
            }
        }
    )
    assert "unavailable" in unavailable["winner"]

    winner = _analysis(
        {
            "strategy_summary": {
                "A": {
                    "overall_average_score": 0.9,
                    "metric_averages": {
                        "fact_coverage_score": 0.8,
                        "tone_match_score": 0.9,
                        "professional_quality_score": 1.0,
                    },
                },
                "B": {
                    "overall_average_score": 0.5,
                    "metric_averages": {
                        "fact_coverage_score": 0.25,
                        "tone_match_score": 0.5,
                        "professional_quality_score": 0.75,
                    },
                },
            }
        }
    )
    assert "Strategy A performed better" in winner["winner"]
    assert "Fact Coverage" in winner["failure"]


def test_report_analysis_identifies_weakest_quality_component() -> None:
    analysis = _analysis(
        {
            "strategy_summary": {
                "A": {
                    "overall_average_score": 0.9,
                    "metric_averages": {
                        "fact_coverage_score": 1.0,
                        "tone_match_score": 1.0,
                        "professional_quality_score": 0.7,
                    },
                },
                "B": {
                    "overall_average_score": 0.8,
                    "metric_averages": {
                        "fact_coverage_score": 1.0,
                        "tone_match_score": 1.0,
                        "professional_quality_score": 0.4,
                    },
                },
            },
            "records": [
                {
                    "strategy": "B",
                    "scores": {
                        "professional_structure_score": 1.0,
                        "professional_placeholder_score": 0.8,
                        "professional_concision_score": 0.2,
                    },
                }
            ],
        }
    )
    assert "concision" in analysis["failure"]
    assert "0.200" in analysis["failure"]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (None, ""),
        ("429 RESOURCE_EXHAUSTED", "rate limit"),
        ("503 UNAVAILABLE", "temporarily unavailable"),
        (
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            "Daily free-tier",
        ),
    ],
)
def test_report_error_messages_are_readable(error, expected: str) -> None:
    assert expected in _compact_error(error)


def test_report_helpers_format_scores_and_truncate_long_errors() -> None:
    assert _format_score(None) == "N/A"
    assert _format_score(0.12345) == "0.123"
    assert len(_compact_error("x" * 300)) == 180


def test_load_evaluation_results_validates_input(workspace_tmp_path) -> None:
    missing = workspace_tmp_path / "missing.json"
    with pytest.raises(ValueError, match="Could not read"):
        load_evaluation_results(missing)
    invalid = workspace_tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        load_evaluation_results(invalid)


def test_final_report_requires_a_complete_evaluation() -> None:
    incomplete = {
        "run_status": "stopped_daily_quota",
        "records": [],
        "strategy_summary": {
            "A": {
                "total_scenarios": 10,
                "evaluated_count": 0,
                "error_count": 10,
            },
            "B": {
                "total_scenarios": 10,
                "evaluated_count": 0,
                "error_count": 10,
            },
        },
    }
    with pytest.raises(ValueError, match="evaluate-resume"):
        validate_complete_evaluation(incomplete)


def test_complete_evaluation_gate_checks_records_and_strategy_totals() -> None:
    records = [
        {
            "scores": {"overall_score": 1.0},
            "generation_error": None,
            "evaluation_error": None,
        }
        for _ in range(20)
    ]
    complete = {
        "run_status": "completed",
        "records": records,
        "strategy_summary": {
            strategy: {
                "total_scenarios": 10,
                "evaluated_count": 10,
                "error_count": 0,
            }
            for strategy in ("A", "B")
        },
    }
    validate_complete_evaluation(complete)
    complete["run_status"] = "completed_with_fallback"
    validate_complete_evaluation(complete)

    complete["records"] = records[:-1]
    with pytest.raises(ValueError, match="20 records"):
        validate_complete_evaluation(complete)


def test_argument_parsers_document_defaults_and_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--scenarios", "custom.json", "--output-dir", "out", "--resume"],
    )
    evaluation_args = evaluate.parse_args()
    assert evaluation_args.scenarios.name == "custom.json"
    assert evaluation_args.output_dir.name == "out"
    assert evaluation_args.resume is True

    monkeypatch.setattr(
        sys,
        "argv",
        ["report", "--input", "input.json", "--markdown", "r.md", "--pdf", "r.pdf"],
    )
    report_args = report.parse_args()
    assert report_args.input.name == "input.json"
    assert report_args.markdown.name == "r.md"
    assert report_args.pdf.name == "r.pdf"


def test_cli_main_returns_error_for_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli.Settings,
        "from_env",
        classmethod(lambda cls: Settings(api_key=None)),
    )
    assert cli.main() == 1


def test_evaluation_main_documents_success_quota_and_error_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmp_path,
) -> None:
    args = Namespace(
        scenarios=workspace_tmp_path / "scenarios.json",
        output_dir=workspace_tmp_path,
        resume=False,
    )
    monkeypatch.setattr(evaluate, "parse_args", lambda: args)
    monkeypatch.setattr(
        evaluate.Settings,
        "from_env",
        classmethod(lambda cls: Settings(api_key="key")),
    )
    monkeypatch.setattr(evaluate, "create_llm_client", lambda settings: object())

    summary = {
        "A": {
            "evaluated_count": 1,
            "total_scenarios": 1,
            "overall_average_score": 0.5,
        },
        "B": {
            "evaluated_count": 0,
            "total_scenarios": 1,
            "overall_average_score": None,
        },
    }
    monkeypatch.setattr(
        evaluate,
        "run_evaluation",
        lambda *args, **kwargs: {
            "strategy_summary": summary,
            "run_status": "completed",
        },
    )
    assert evaluate.main() == 0

    monkeypatch.setattr(
        evaluate,
        "run_evaluation",
        lambda *args, **kwargs: {
            "strategy_summary": summary,
            "run_status": "stopped_daily_quota",
        },
    )
    assert evaluate.main() == 2

    monkeypatch.setattr(
        evaluate,
        "run_evaluation",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad evaluation")),
    )
    assert evaluate.main() == 1


def test_report_main_documents_success_and_error_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmp_path,
) -> None:
    args = Namespace(
        input=workspace_tmp_path / "input.json",
        markdown=workspace_tmp_path / "report.md",
        pdf=workspace_tmp_path / "report.pdf",
    )
    monkeypatch.setattr(report, "parse_args", lambda: args)
    monkeypatch.setattr(
        report,
        "generate_reports",
        lambda *args: (args[1], args[2]),
    )
    assert report.main() == 0
    monkeypatch.setattr(
        report,
        "generate_reports",
        lambda *args: (_ for _ in ()).throw(ValueError("bad report")),
    )
    assert report.main() == 1


def test_mixed_provider_report_contains_prominent_warning(
    workspace_tmp_path,
) -> None:
    from src.evaluate import run_evaluation

    results_dir = workspace_tmp_path / "results"
    reports_dir = workspace_tmp_path / "reports"
    settings = Settings(
        api_key=None,
        generation_model="gemini-generation",
        judge_model="gemini-judge",
        groq_generation_model="groq-generation",
        groq_judge_model="groq-judge",
    )
    run_evaluation(FallbackClient(), settings, output_dir=results_dir)
    markdown_path, _ = generate_reports(
        results_dir / "evaluation_results.json",
        reports_dir / "final_report.md",
        reports_dir / "final_report.pdf",
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Mixed-provider warning" in markdown
    assert "fallback" in markdown.lower()
