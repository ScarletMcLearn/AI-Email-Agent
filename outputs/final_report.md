# Email Generation Assistant - Evaluation Report

Generated: 2026-06-19T08:44:18.338574+00:00

Provider order: `gemini`  
Primary generation: `gemini` / `gemini-3.1-flash-lite-preview`  
Primary judge: `gemini` / `gemini-3-flash-preview`

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

**Logic:** Python averages two components: the normalized 1-5 LLM quality rubric and an automated email-quality score. The automated score equally weights structural completeness (subject, greeting, closing), placeholder discipline, and concision relative to the human reference. Extra bracketed placeholders beyond [Your Name] reduce the placeholder component by 0.1 each, to a 0.5 floor. Concision receives 1.0, 0.75, 0.5, 0.25, or 0.0 when generated/reference word-count ratio is at most 1.25, 1.5, 1.75, 2.0, or above 2.0 respectively.

## Strategy Averages

| Strategy | Evaluated | Errors | Fact Coverage | Tone Match | Professional Quality | Overall |
|---|---:|---:|---:|---:|---:|---:|
| A | 10/10 | 0 | 1.000 | 1.000 | 0.912 | 0.971 |
| B | 10/10 | 0 | 1.000 | 1.000 | 0.803 | 0.934 |

## Raw Evaluation Results

| Scenario | Strategy | Generation | Judge | Fact Coverage | Tone (raw/norm) | Quality (raw/norm) | Overall | Error |
|---|:---:|---|---|---:|---:|---:|---:|---|
| meeting-follow-up | A | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.958) | 0.986 |  |
| meeting-follow-up | B | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.833) | 0.944 |  |
| proposal-details-request | A | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.942) | 0.981 |  |
| proposal-details-request | B | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.767) | 0.922 |  |
| delayed-response-apology | A | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.917) | 0.972 |  |
| delayed-response-apology | B | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.783) | 0.928 |  |
| urgent-deadline-reminder | A | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (1.000) | 1.000 |  |
| urgent-deadline-reminder | B | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.917) | 0.972 |  |
| empathetic-support-reply | A | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.875) | 0.958 |  |
| empathetic-support-reply | B | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.800) | 0.933 |  |
| sales-outreach | A | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.842) | 0.947 |  |
| sales-outreach | B | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.800) | 0.933 |  |
| project-status-update | A | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.875) | 0.958 |  |
| project-status-update | B | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.783) | 0.928 |  |
| meeting-reschedule | A | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.875) | 0.958 |  |
| meeting-reschedule | B | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.800) | 0.933 |  |
| invoice-follow-up | A | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.875) | 0.958 |  |
| invoice-follow-up | B | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.800) | 0.933 |  |
| interview-invitation | A | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.958) | 0.986 |  |
| interview-invitation | B | gemini / gemini-3.1-flash-lite-preview | gemini / gemini-3-flash-preview | 1.000 | 5/5 (1.000) | 5/5 (0.750) | 0.917 |  |

## Comparative Analysis

### Which strategy performed better?

Strategy A performed better with an overall average of 0.971, compared with 0.934 for Strategy B.

### Biggest failure mode of the lower-performing strategy

The largest measured weakness of Strategy B was Professional Quality, its lowest average metric at 0.803. Within that hybrid metric, concision was the weakest automated component at 0.050.

### Production recommendation

Recommend Strategy A for production because it achieved the stronger aggregate result under the same scenarios, model, judge, and metric definitions. Retain monitoring for fact fidelity and review the recommendation whenever the prompt or model changes.

## Notes and Limitations

- LLM-as-a-judge scores may vary despite a temperature of 0.0.
- Free-tier availability and rate limits can change.
- Mixed-provider runs are flagged and should not be treated as fully controlled comparisons.
- Human review remains necessary for sensitive or high-impact emails.
- This report can be regenerated as Markdown and PDF with `pixi run report`.
