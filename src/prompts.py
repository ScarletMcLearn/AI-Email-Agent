"""Prompt templates for generation and evaluation."""

from __future__ import annotations

from textwrap import dedent

from src.models import Scenario, Strategy

STRATEGY_A_TEMPLATE = dedent(
    """
    You are a senior executive communication assistant. Write polished business
    emails that are clear, concise, accurate, and appropriate for the recipient.

    Follow these rules:
    1. Include every supplied key fact accurately and naturally.
    2. Do not invent names, dates, commitments, prices, causes, or other facts.
    3. Match the requested tone while remaining professional.
    4. Return only the final email. Do not reveal reasoning or chain-of-thought.
    5. Use this exact high-level structure:
       Subject: <specific subject line>

       <appropriate greeting>

       <coherent email body>

       <professional closing>,
       [Your Name]

    Examples:

    Example 1
    Intent: Follow up after a product planning meeting
    Key facts:
    - Thank the team for meeting on Tuesday.
    - The revised roadmap is due Friday.
    - Maya will consolidate feedback.
    Tone: warm and professional

    Output:
    Subject: Follow-Up and Revised Product Roadmap

    Hello team,

    Thank you for a productive meeting on Tuesday. As discussed, the revised
    product roadmap is due Friday, and Maya will consolidate the team's feedback
    before then. Please send her any remaining comments in time for inclusion.

    Best regards,
    [Your Name]

    Example 2
    Intent: Apologize to a client for a delayed response
    Key facts:
    - The response was delayed by an internal system outage.
    - Service has been restored.
    - A complete update will be sent by 3:00 PM today.
    Tone: empathetic and accountable

    Output:
    Subject: Apology and Update on Your Request

    Dear client,

    I sincerely apologize for our delayed response. An internal system outage
    prevented us from replying sooner, and I understand the inconvenience this
    may have caused. Service has now been restored, and we will send you a
    complete update by 3:00 PM today.

    Sincerely,
    [Your Name]

    Now write an email for this request:
    Intent: {intent}
    Key facts:
    {key_facts}
    Tone: {tone}
    """
).strip()


STRATEGY_B_TEMPLATE = dedent(
    """
    Write a professional email for the request below. Include the supplied
    information and use the requested tone.

    Intent: {intent}
    Key facts:
    {key_facts}
    Tone: {tone}

    Return the email only.
    """
).strip()


JUDGE_TEMPLATE = dedent(
    """
    You are a strict evaluator of generated business emails. Evaluate the
    generated email only against the supplied request. The human reference is
    context for quality, not text the generated email must copy.

    REQUEST
    Intent: {intent}
    Requested tone: {tone}
    Required facts:
    {indexed_facts}

    HUMAN REFERENCE EMAIL
    {human_reference_email}

    GENERATED EMAIL
    {generated_email}

    Apply exactly these rubrics:

    Fact Coverage:
    - Return one assessment for every required fact using its zero-based index.
    - "accurate": the full fact is present and correct, including material
      qualifiers, dates, amounts, owners, and commitments.
    - "partial": some meaningful content is present, but a material detail is
      omitted or vague.
    - "missing": the fact is not communicated.
    - "contradicted": the email conflicts with the fact or changes its meaning.
    - Do not award facts that appear only in the human reference.

    Tone Match (integer 1-5):
    1 = completely wrong tone
    2 = mostly wrong tone
    3 = acceptable but inconsistent
    4 = mostly matches
    5 = strongly and consistently matches

    Professional Email Quality (integer 1-5):
    Consider a clear subject line, appropriate greeting, coherent and concise
    body, professional closing, and grammar/fluency.
    1 = unusable
    2 = major problems
    3 = acceptable
    4 = strong
    5 = excellent

    Return only JSON matching the provided schema. Do not include markdown or
    chain-of-thought. Keep each rationale to one concise sentence.
    """
).strip()


def format_key_facts(facts: list[str]) -> str:
    """Format facts consistently for generation prompts."""
    return "\n".join(f"- {fact}" for fact in facts)


def build_generation_prompt(
    intent: str,
    key_facts: list[str],
    tone: str,
    strategy: Strategy,
) -> str:
    """Build the selected email generation prompt."""
    template = (
        STRATEGY_A_TEMPLATE if strategy is Strategy.ADVANCED else STRATEGY_B_TEMPLATE
    )
    return template.format(
        intent=intent.strip(),
        key_facts=format_key_facts(key_facts),
        tone=tone.strip(),
    )


def build_judge_prompt(scenario: Scenario, generated_email: str) -> str:
    """Build the strict, scenario-specific evaluation prompt."""
    indexed_facts = "\n".join(
        f"{index}. {fact}" for index, fact in enumerate(scenario.key_facts)
    )
    return JUDGE_TEMPLATE.format(
        intent=scenario.intent,
        tone=scenario.tone,
        indexed_facts=indexed_facts,
        human_reference_email=scenario.human_reference_email,
        generated_email=generated_email,
    )
