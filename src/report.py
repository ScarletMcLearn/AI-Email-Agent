"""Generate the comparative Markdown and PDF reports."""

from __future__ import annotations

import argparse
import html
import json
import logging
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    LongTable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import DEFAULT_OUTPUTS_DIR
from src.prompts import STRATEGY_A_TEMPLATE, STRATEGY_B_TEMPLATE

LOGGER = logging.getLogger(__name__)
METRIC_KEYS = [
    ("fact_coverage_score", "Fact Coverage"),
    ("tone_match_score", "Tone Match"),
    ("professional_quality_score", "Professional Quality"),
]


def load_evaluation_results(path: Path | str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read evaluation results from {path}: {exc}"
        ) from exc
    required = {"metric_definitions", "records", "strategy_summary"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Evaluation results are missing keys: {sorted(missing)}")
    return data


def validate_complete_evaluation(data: dict[str, Any]) -> None:
    """Require the complete 10-scenario, two-strategy assessment result."""
    if data.get("run_status") not in {"completed", "completed_with_fallback"}:
        raise ValueError(
            "Evaluation is incomplete. Run `pixi run evaluate-resume` before "
            "generating the final report."
        )

    records = data["records"]
    if len(records) != 20:
        raise ValueError(f"Evaluation must contain 20 records; found {len(records)}.")

    summaries = data["strategy_summary"]
    for strategy in ("A", "B"):
        summary = summaries.get(strategy)
        if not summary:
            raise ValueError(f"Evaluation summary is missing Strategy {strategy}.")
        if (
            summary.get("total_scenarios") != 10
            or summary.get("evaluated_count") != 10
            or summary.get("error_count") != 0
        ):
            raise ValueError(
                f"Strategy {strategy} is incomplete: expected 10 evaluated "
                "scenarios and zero errors."
            )

    incomplete = [
        record
        for record in records
        if record.get("scores") is None
        or record.get("generation_error")
        or record.get("evaluation_error")
    ]
    if incomplete:
        raise ValueError(
            f"Evaluation contains {len(incomplete)} incomplete or failed record(s)."
        )


def _format_score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _compact_error(error: Any) -> str:
    """Keep report tables readable while raw files retain the complete error."""
    text = str(error or "").replace("\n", " ")
    if not text:
        return ""
    if "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in text:
        return "Daily free-tier request quota exhausted for this model."
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "Provider rate limit or quota exhausted."
    if "503" in text or "UNAVAILABLE" in text:
        return "Provider temporarily unavailable after retries."
    if len(text) > 180:
        return text[:177] + "..."
    return text


def _analysis(data: dict[str, Any]) -> dict[str, str]:
    summaries = data["strategy_summary"]
    valid = {
        strategy: summary
        for strategy, summary in summaries.items()
        if summary.get("overall_average_score") is not None
    }
    if len(valid) < 2:
        return {
            "winner": "A complete comparison is unavailable.",
            "failure": (
                "The lower-performing strategy cannot be identified because one or "
                "both strategies have no valid evaluated results."
            ),
            "recommendation": (
                "Run the evaluation again after resolving the recorded API or judge "
                "errors before making a production recommendation."
            ),
        }

    ranked = sorted(
        valid.items(),
        key=lambda item: item[1]["overall_average_score"],
        reverse=True,
    )
    winner_key, winner = ranked[0]
    loser_key, loser = ranked[1]
    if abs(winner["overall_average_score"] - loser["overall_average_score"]) < 1e-9:
        return {
            "winner": (
                f"The evaluated results are tied at "
                f"{winner['overall_average_score']:.3f}. They do not establish a "
                "better strategy."
            ),
            "failure": (
                "There is no lower-performing strategy in the available scores. "
                "Complete the failed cases before drawing a failure-mode conclusion."
            ),
            "recommendation": (
                "Do not make a data-based production selection from this tie. "
                "Complete the evaluation, then compare the full metric averages and "
                "judge rationales."
            ),
        }
    loser_metrics = loser["metric_averages"]
    lowest_key, lowest_label = min(
        METRIC_KEYS,
        key=lambda item: (
            loser_metrics[item[0]]
            if loser_metrics[item[0]] is not None
            else float("inf")
        ),
    )
    failure_detail = (
        f"{lowest_label}, its lowest average metric at {loser_metrics[lowest_key]:.3f}."
    )
    if lowest_key == "professional_quality_score":
        component_keys = [
            ("professional_structure_score", "email structure"),
            ("professional_placeholder_score", "placeholder discipline"),
            ("professional_concision_score", "concision"),
        ]
        loser_records = [
            record
            for record in data.get("records", [])
            if record.get("strategy") == loser_key and record.get("scores")
        ]
        component_averages = {
            key: sum(record["scores"][key] for record in loser_records)
            / len(loser_records)
            for key, _ in component_keys
            if loser_records
            and all(key in record["scores"] for record in loser_records)
        }
        if component_averages:
            component_key, component_label = min(
                component_keys,
                key=lambda item: component_averages.get(item[0], float("inf")),
            )
            failure_detail = (
                f"{lowest_label}, its lowest average metric at "
                f"{loser_metrics[lowest_key]:.3f}. Within that hybrid metric, "
                f"{component_label} was the weakest automated component at "
                f"{component_averages[component_key]:.3f}."
            )
    mixed_provider = bool(data.get("fallback_used"))
    recommendation_basis = (
        "The comparison includes fallback calls, so confirm the result with a "
        "single provider/model route before making the final production selection."
        if mixed_provider
        else (
            f"Recommend Strategy {winner_key} for production because it achieved "
            "the stronger aggregate result under the same scenarios, model, judge, "
            "and metric definitions. Retain monitoring for fact fidelity and review "
            "the recommendation whenever the prompt or model changes."
        )
    )
    return {
        "winner": (
            f"Strategy {winner_key} performed better with an overall average of "
            f"{winner['overall_average_score']:.3f}, compared with "
            f"{loser['overall_average_score']:.3f} for Strategy {loser_key}."
        ),
        "failure": (
            f"The largest measured weakness of Strategy {loser_key} was "
            f"{failure_detail}"
        ),
        "recommendation": (recommendation_basis),
    }


def build_markdown_report(data: dict[str, Any]) -> str:
    """Build a complete assessment report in Markdown."""
    summaries = data["strategy_summary"]
    incomplete_count = sum(
        summary.get("error_count", 0) for summary in summaries.values()
    )
    lines = [
        "# Email Generation Assistant - Evaluation Report",
        "",
        f"Generated: {data.get('generated_at_utc', 'unknown')}",
        "",
        f"Provider order: `{', '.join(data.get('provider_order', [])) or 'unknown'}`  ",
        f"Primary generation: `{data.get('generation_provider', 'unknown')}` / "
        f"`{data.get('generation_model', 'unknown')}`  ",
        f"Primary judge: `{data.get('judge_provider', 'unknown')}` / "
        f"`{data.get('judge_model', 'unknown')}`",
        "",
    ]
    if data.get("fallback_used"):
        lines.extend(
            [
                "> **Mixed-provider warning:** "
                + data.get(
                    "comparison_warning",
                    "Fallback calls were used; this is not a fully controlled "
                    "single-model comparison.",
                ),
                "",
            ]
        )
    if incomplete_count:
        lines.extend(
            [
                "> **Incomplete evaluation warning:** "
                f"{incomplete_count} scenario-strategy result(s) failed. "
                "Averages exclude failed records; consult the raw results for errors.",
                "",
            ]
        )

    lines.extend(
        [
            "## Prompt Template - Strategy A",
            "",
            "Advanced role-playing, few-shot, structured, fact-fidelity prompt:",
            "",
            "```text",
            STRATEGY_A_TEMPLATE,
            "```",
            "",
            "## Prompt Template - Strategy B",
            "",
            "Simple baseline prompt:",
            "",
            "```text",
            STRATEGY_B_TEMPLATE,
            "```",
            "",
            "## Custom Metrics",
            "",
        ]
    )
    for metric in data["metric_definitions"].values():
        lines.extend(
            [
                f"### {metric['name']}",
                "",
                f"**Definition:** {metric['definition']}",
                "",
                f"**Logic:** {metric['logic']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Strategy Averages",
            "",
            "| Strategy | Evaluated | Errors | Fact Coverage | Tone Match | Professional Quality | Overall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy in ["A", "B"]:
        summary = summaries[strategy]
        averages = summary["metric_averages"]
        lines.append(
            "| {strategy} | {evaluated}/{total} | {errors} | {fact} | {tone} | "
            "{quality} | {overall} |".format(
                strategy=strategy,
                evaluated=summary["evaluated_count"],
                total=summary["total_scenarios"],
                errors=summary["error_count"],
                fact=_format_score(averages["fact_coverage_score"]),
                tone=_format_score(averages["tone_match_score"]),
                quality=_format_score(averages["professional_quality_score"]),
                overall=_format_score(summary["overall_average_score"]),
            )
        )

    lines.extend(
        [
            "",
            "## Raw Evaluation Results",
            "",
            "| Scenario | Strategy | Generation | Judge | Fact Coverage | Tone (raw/norm) | Quality (raw/norm) | Overall | Error |",
            "|---|:---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for record in data["records"]:
        scores = record.get("scores")
        error = _compact_error(
            record.get("generation_error") or record.get("evaluation_error")
        ).replace("|", "\\|")
        if scores:
            fact = _format_score(scores["fact_coverage_score"])
            tone = (
                f"{scores['tone_raw_score']}/5 "
                f"({_format_score(scores['tone_match_score'])})"
            )
            quality = (
                f"{scores['professional_quality_raw_score']}/5 "
                f"({_format_score(scores['professional_quality_score'])})"
            )
            overall = _format_score(scores["overall_score"])
        else:
            fact = tone = quality = overall = "N/A"
        generation_route = (
            f"{record.get('generation_provider') or 'unknown'} / "
            f"{record.get('generation_model') or 'unknown'}"
        )
        judge_route = (
            f"{record.get('judge_provider') or 'unknown'} / "
            f"{record.get('judge_model') or 'unknown'}"
        )
        if record.get("generation_fallback_used"):
            generation_route += " (fallback)"
        if record.get("judge_fallback_used"):
            judge_route += " (fallback)"
        lines.append(
            f"| {record['scenario_id']} | {record['strategy']} | "
            f"{generation_route} | {judge_route} | {fact} | {tone} | "
            f"{quality} | {overall} | {error} |"
        )

    analysis = _analysis(data)
    lines.extend(
        [
            "",
            "## Comparative Analysis",
            "",
            "### Which strategy performed better?",
            "",
            analysis["winner"],
            "",
            "### Biggest failure mode of the lower-performing strategy",
            "",
            analysis["failure"],
            "",
            "### Production recommendation",
            "",
            analysis["recommendation"],
            "",
            "## Notes and Limitations",
            "",
            "- LLM-as-a-judge scores may vary despite a temperature of 0.0.",
            "- Free-tier availability and rate limits can change.",
            "- Mixed-provider runs are flagged and should not be treated as fully controlled comparisons.",
            "- Human review remains necessary for sensitive or high-impact emails.",
            "- This report can be regenerated as Markdown and PDF with `pixi run report`.",
            "",
        ]
    )
    return "\n".join(lines)


def _page_number(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#59636E"))
    canvas.drawRightString(
        landscape(letter)[0] - 0.55 * inch,
        0.35 * inch,
        f"Page {document.page}",
    )
    canvas.restoreState()


def _pdf_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=14,
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "ReportHeading1",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "ReportHeading2",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#24557A"),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#202830"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#202830"),
        ),
        "code": ParagraphStyle(
            "ReportCode",
            parent=sample["Code"],
            fontName="Courier",
            fontSize=6.5,
            leading=8.2,
            leftIndent=8,
            rightIndent=8,
            borderColor=colors.HexColor("#CBD5DF"),
            borderWidth=0.5,
            borderPadding=7,
            backColor=colors.HexColor("#F4F7F9"),
            spaceAfter=8,
        ),
        "warning": ParagraphStyle(
            "ReportWarning",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#7A3E00"),
            backColor=colors.HexColor("#FFF2D8"),
            borderPadding=7,
            spaceAfter=10,
        ),
    }


def _paragraph(text: Any, style: ParagraphStyle) -> Paragraph:
    safe = html.escape(str(text)).replace("\n", "<br/>")
    return Paragraph(safe, style)


def build_pdf_report(data: dict[str, Any], output_path: Path | str) -> None:
    """Build a polished landscape PDF from structured evaluation results."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = _pdf_styles()
    document = SimpleDocTemplate(
        str(path),
        pagesize=landscape(letter),
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Email Generation Assistant - Evaluation Report",
        author="AI Engineer Candidate",
    )
    story: list[Any] = [
        Paragraph("Email Generation Assistant", styles["title"]),
        Paragraph("Comparative Evaluation Report", styles["h1"]),
        _paragraph(
            f"Generated: {data.get('generated_at_utc', 'unknown')}\n"
            f"Provider order: {', '.join(data.get('provider_order', [])) or 'unknown'}\n"
            f"Primary generation: {data.get('generation_provider', 'unknown')} / "
            f"{data.get('generation_model', 'unknown')} | "
            f"Primary judge: {data.get('judge_provider', 'unknown')} / "
            f"{data.get('judge_model', 'unknown')}",
            styles["body"],
        ),
    ]

    summaries = data["strategy_summary"]
    incomplete_count = sum(
        summary.get("error_count", 0) for summary in summaries.values()
    )
    if incomplete_count:
        story.append(
            _paragraph(
                f"Incomplete evaluation: {incomplete_count} result(s) failed. "
                "Averages exclude failed records.",
                styles["warning"],
            )
        )
    if data.get("fallback_used"):
        story.append(
            _paragraph(
                data.get(
                    "comparison_warning",
                    "Fallback calls were used; this is not a fully controlled "
                    "single-model comparison.",
                ),
                styles["warning"],
            )
        )

    story.extend(
        [
            Paragraph("Prompt Template - Strategy A", styles["h1"]),
            _paragraph(
                "Advanced role-playing, few-shot, structured, fact-fidelity prompt.",
                styles["body"],
            ),
            Preformatted(STRATEGY_A_TEMPLATE, styles["code"], maxLineLength=125),
            Paragraph("Prompt Template - Strategy B", styles["h1"]),
            _paragraph("Simple baseline prompt.", styles["body"]),
            Preformatted(STRATEGY_B_TEMPLATE, styles["code"], maxLineLength=125),
            PageBreak(),
            Paragraph("Custom Metrics", styles["h1"]),
        ]
    )
    for metric in data["metric_definitions"].values():
        story.append(Paragraph(metric["name"], styles["h2"]))
        story.append(
            _paragraph(
                f"Definition: {metric['definition']}\nLogic: {metric['logic']}",
                styles["body"],
            )
        )

    story.append(Paragraph("Strategy Averages", styles["h1"]))
    summary_rows: list[list[Any]] = [
        [
            "Strategy",
            "Evaluated",
            "Errors",
            "Fact Coverage",
            "Tone Match",
            "Professional Quality",
            "Overall",
        ]
    ]
    for strategy in ["A", "B"]:
        summary = summaries[strategy]
        averages = summary["metric_averages"]
        summary_rows.append(
            [
                strategy,
                f"{summary['evaluated_count']}/{summary['total_scenarios']}",
                str(summary["error_count"]),
                _format_score(averages["fact_coverage_score"]),
                _format_score(averages["tone_match_score"]),
                _format_score(averages["professional_quality_score"]),
                _format_score(summary["overall_average_score"]),
            ]
        )
    summary_table = Table(
        summary_rows,
        colWidths=[
            0.7 * inch,
            0.8 * inch,
            0.55 * inch,
            1.05 * inch,
            0.9 * inch,
            1.25 * inch,
            0.75 * inch,
        ],
        repeatRows=1,
    )
    summary_table.setStyle(_table_style())
    story.append(summary_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Raw Evaluation Results", styles["h1"]))
    raw_rows: list[list[Any]] = [
        [
            "Scenario",
            "Strategy",
            "Generation",
            "Judge",
            "Fact",
            "Tone",
            "Quality",
            "Overall",
            "Error",
        ]
    ]
    for record in data["records"]:
        scores = record.get("scores") or {}
        error = _compact_error(
            record.get("generation_error") or record.get("evaluation_error")
        )
        raw_rows.append(
            [
                _paragraph(record["scenario_id"], styles["small"]),
                record["strategy"],
                _paragraph(
                    f"{record.get('generation_provider') or 'unknown'} / "
                    f"{record.get('generation_model') or 'unknown'}"
                    + (" (fallback)" if record.get("generation_fallback_used") else ""),
                    styles["small"],
                ),
                _paragraph(
                    f"{record.get('judge_provider') or 'unknown'} / "
                    f"{record.get('judge_model') or 'unknown'}"
                    + (" (fallback)" if record.get("judge_fallback_used") else ""),
                    styles["small"],
                ),
                _format_score(scores.get("fact_coverage_score")),
                (
                    f"{scores.get('tone_raw_score')}/5 "
                    f"({_format_score(scores.get('tone_match_score'))})"
                    if scores
                    else "N/A"
                ),
                (
                    f"{scores.get('professional_quality_raw_score')}/5 "
                    f"({_format_score(scores.get('professional_quality_score'))})"
                    if scores
                    else "N/A"
                ),
                _format_score(scores.get("overall_score")),
                _paragraph(error or "-", styles["small"]),
            ]
        )
    raw_table = LongTable(
        raw_rows,
        colWidths=[
            1.15 * inch,
            0.45 * inch,
            1.4 * inch,
            1.4 * inch,
            0.55 * inch,
            0.85 * inch,
            0.85 * inch,
            0.6 * inch,
            2.3 * inch,
        ],
        repeatRows=1,
    )
    raw_table.setStyle(_table_style())
    story.append(raw_table)

    analysis = _analysis(data)
    story.extend(
        [
            PageBreak(),
            Paragraph("Comparative Analysis", styles["h1"]),
            Paragraph("Which strategy performed better?", styles["h2"]),
            _paragraph(analysis["winner"], styles["body"]),
            Paragraph(
                "Biggest failure mode of the lower-performing strategy",
                styles["h2"],
            ),
            _paragraph(analysis["failure"], styles["body"]),
            Paragraph("Production recommendation", styles["h2"]),
            _paragraph(analysis["recommendation"], styles["body"]),
            Paragraph("Notes and Limitations", styles["h1"]),
            _paragraph(
                "LLM-as-a-judge scores may vary despite a temperature of 0.0.\n"
                "Free-tier availability and rate limits can change.\n"
                "Mixed-provider runs are not fully controlled comparisons.\n"
                "Human review remains necessary for sensitive or high-impact emails.",
                styles["body"],
            ),
        ]
    )
    document.build(story, onFirstPage=_page_number, onLaterPages=_page_number)


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("LEADING", (0, 0), (-1, -1), 9.5),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AEB9C4")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, colors.HexColor("#F4F7F9")],
            ),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )


def generate_reports(
    input_path: Path | str = DEFAULT_OUTPUTS_DIR / "evaluation_results.json",
    markdown_path: Path | str = DEFAULT_OUTPUTS_DIR / "final_report.md",
    pdf_path: Path | str = DEFAULT_OUTPUTS_DIR / "final_report.pdf",
) -> tuple[Path, Path]:
    """Read evaluation results and write both report formats."""
    data = load_evaluation_results(input_path)
    validate_complete_evaluation(data)
    markdown_destination = Path(markdown_path)
    pdf_destination = Path(pdf_path)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.write_text(
        build_markdown_report(data),
        encoding="utf-8",
    )
    build_pdf_report(data, pdf_destination)
    return markdown_destination, pdf_destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Markdown and PDF evaluation reports."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR / "evaluation_results.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR / "final_report.md",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR / "final_report.pdf",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        markdown_path, pdf_path = generate_reports(args.input, args.markdown, args.pdf)
        LOGGER.info("Markdown report written to %s", markdown_path)
        LOGGER.info("PDF report written to %s", pdf_path)
        return 0
    except (ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
