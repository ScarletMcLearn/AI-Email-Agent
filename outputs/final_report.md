# Email Generation Assistant - Evaluation Report

Generated: 2026-06-19T08:44:18.338574+00:00

Generation model: `gemini-2.5-flash`  
Judge model: `gemini-2.5-flash-lite`

> **Incomplete evaluation warning:** 17 scenario-strategy result(s) failed. Averages exclude failed records; consult the raw results for errors.

## Prompt Template - Strategy A

Advanced role-playing, few-shot, structured, fact-fidelity prompt:

```text
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
```

## Prompt Template - Strategy B

Simple baseline prompt:

```text
Write a professional email for the request below. Include the supplied
information and use the requested tone.

Intent: {intent}
Key facts:
{key_facts}
Tone: {tone}

Return the email only.
```

## Custom Metrics

### Fact Coverage Score

**Definition:** Measures whether every required key fact is included accurately.

**Logic:** The LLM judge labels each fact accurate, partial, missing, or contradicted. Python assigns credits of 1.0, 0.5, 0.0, and 0.0 respectively, then averages the credits across all required facts.

### Tone Match Score

**Definition:** Measures how consistently the email matches the requested tone.

**Logic:** The LLM judge applies a 1-5 rubric, where 1 is completely wrong and 5 is a strong match. Python normalizes the score with (raw - 1) / 4.

### Professional Email Quality Score

**Definition:** Measures professionalism, clarity, concision, formatting, and fluency.

**Logic:** The LLM judge scores subject line, greeting, coherent body, concision, closing, and grammar from 1-5. Python normalizes with (raw - 1) / 4.

## Strategy Averages

| Strategy | Evaluated | Errors | Fact Coverage | Tone Match | Professional Quality | Overall |
|---|---:|---:|---:|---:|---:|---:|
| A | 2/10 | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| B | 1/10 | 9 | 1.000 | 1.000 | 1.000 | 1.000 |

## Raw Evaluation Results

| Scenario | Strategy | Fact Coverage | Tone (raw/norm) | Quality (raw/norm) | Overall | Error |
|---|:---:|---:|---:|---:|---:|---|
| meeting-follow-up | A | 1.000 | 5/5 (1.000) | 5/5 (1.000) | 1.000 |  |
| meeting-follow-up | B | 1.000 | 5/5 (1.000) | 5/5 (1.000) | 1.000 |  |
| proposal-details-request | A | 1.000 | 5/5 (1.000) | 5/5 (1.000) | 1.000 |  |
| proposal-details-request | B | N/A | N/A | N/A | N/A | Daily free-tier request quota exhausted for this model. |
| delayed-response-apology | A | N/A | N/A | N/A | N/A | Daily free-tier request quota exhausted for this model. |
| delayed-response-apology | B | N/A | N/A | N/A | N/A | Daily free-tier request quota exhausted for this model. |
| urgent-deadline-reminder | A | N/A | N/A | N/A | N/A | Daily free-tier request quota exhausted for this model. |
| urgent-deadline-reminder | B | N/A | N/A | N/A | N/A | Daily free-tier request quota exhausted for this model. |
| empathetic-support-reply | A | N/A | N/A | N/A | N/A | Daily free-tier request quota exhausted for this model. |
| empathetic-support-reply | B | N/A | N/A | N/A | N/A | Daily free-tier request quota exhausted for this model. |
| sales-outreach | A | N/A | N/A | N/A | N/A |  |
| sales-outreach | B | N/A | N/A | N/A | N/A |  |
| project-status-update | A | N/A | N/A | N/A | N/A |  |

## Comparative Analysis

### Which strategy performed better?

The evaluated results are tied at 1.000. They do not establish a better strategy.

### Biggest failure mode of the lower-performing strategy

There is no lower-performing strategy in the available scores. Complete the failed cases before drawing a failure-mode conclusion.

### Production recommendation

Do not make a data-based production selection from this tie. Complete the evaluation, then compare the full metric averages and judge rationales.

## Notes and Limitations

- LLM-as-a-judge scores may vary despite a temperature of 0.0.
- Free-tier availability and rate limits can change.
- Human review remains necessary for sensitive or high-impact emails.
- This report can be regenerated as Markdown and PDF with `pixi run report`.
