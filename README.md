# Email Generation Assistant

A compact AI Engineer assessment project that generates professional emails from
an intent, required key facts, and a requested tone. It also compares an advanced
prompt against a baseline prompt across 10 fixed scenarios using three custom,
email-specific metrics.

Repository: [ScarletMcLearn/AI-Email-Agent](https://github.com/ScarletMcLearn/AI-Email-Agent)

The production route uses GroqCloud first and automatically falls back to Google
Gemini after retryable failures. Groq hosts open-weight models; GroqCloud itself
is a hosted commercial API, not an open-source service. No API keys or fabricated
evaluation results are committed.

## Features

- Interactive command-line email generation.
- Strategy A: senior executive assistant role, two few-shot examples, mandatory
  email structure, explicit fact fidelity, tone control, and final-answer-only
  output.
- Strategy B: a short baseline prompt without examples or strict structure.
- Controlled comparison using the same configured route and the same 10 scenarios.
- Groq strict JSON-schema judge responses validated again with Pydantic.
- Three-attempt exponential retry handling for generation and judging.
- Automatic Groq-to-Gemini fallback with actual provider/model metadata.
- Mixed-provider evaluation warnings in JSON, CSV, Markdown, and PDF artifacts.
- Failure isolation: one failed API call does not terminate the full evaluation.
- JSON, CSV, Markdown, and PDF assessment artifacts.
- Fully offline tests using deterministic provider clients.

## Project structure

```text
.
|-- data/scenarios.json
|-- outputs/
|-- src/
|   |-- cli.py
|   |-- config.py
|   |-- evaluate.py
|   |-- generator.py
|   |-- llm_client.py
|   |-- models.py
|   |-- prompts.py
|   `-- report.py
|-- tests/
|-- .env.example
|-- pixi.toml
|-- requirements.txt
`-- README.md
```

## Environment management: Pixi with uv

[Pixi](https://pixi.sh/) is the primary environment and task runner. The
`[pypi-dependencies]` section in `pixi.toml` is resolved and installed through
uv, while Pixi also pins a compatible Python runtime. The project tasks invoke
Python through `uv run` inside that Pixi environment.

Install Pixi, then create the environment:

```bash
pixi install
```

Useful commands:

```bash
pixi run app
pixi run evaluate
pixi run evaluate-resume
pixi run report
pixi run test
pixi run lint
pixi run format
pixi run format-check
pixi run verify
```

Ruff is the canonical Python linter and formatter. The lint configuration
enables all stable rules, excluding only rules that conflict with Ruff's
formatter and narrowly scoped test conventions. Run `pixi run format` before
committing; `pixi run verify` runs tests, linting, and the formatting check.

The explicit module commands also work through Pixi:

```bash
pixi run python -m src.cli
pixi run python -m src.evaluate
pixi run python -m src.report
pixi run pytest
```

`requirements.txt` is retained for assessment-platform compatibility. If Pixi is
unavailable, uv can create an equivalent local environment:

```bash
uv venv
uv pip install -r requirements.txt
```

Activate `.venv`, then use the direct `python -m ...` commands shown above
without the `pixi run` prefix. The fallback requirements include the same test,
coverage, lint, and formatting tools used by `pixi run verify`.

## Tests as executable documentation

The pytest suite is fully offline: it never calls Groq or Gemini and does not
require an API key. Deterministic clients model successful responses, retries,
fallback, malformed judge output, quota exhaustion, and partial evaluation runs.

Tests are grouped by application responsibility:

- `test_config.py` documents environment variables and validation.
- `test_models_and_prompts.py` documents accepted domain data and prompt shape.
- `test_generator.py` documents generation input and client delegation.
- `test_llm_client.py` documents retry, parsing, quota, and pacing behavior.
- `test_evaluate.py` documents scoring, persistence, failures, and resume rules.
- `test_cli_and_report.py` documents user interaction and report contracts.

Run the documented behavior and branch-coverage gate with:

```bash
pixi run test
uv run pytest --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=95
pixi run verify
```

Descriptive test names and realistic inputs are intentional: read an individual
test to see an example of the supported behavior and its expected result.

## Configure hosted API keys

1. Create a [GroqCloud API key](https://console.groq.com/keys).
2. Create a [Google AI Studio API key](https://aistudio.google.com/app/apikey).
3. Copy `.env.example` to `.env`.
4. Add both keys:

   ```dotenv
   GROQ_API_KEY=your_real_groq_key
   GEMINI_API_KEY=your_real_key_here
   LLM_PROVIDER_ORDER=groq,gemini
   ```

5. Run the project with `pixi run app`.

Both keys are required for the default automatic-fallback route. A single
provider can be selected with `LLM_PROVIDER_ORDER=groq` or
`LLM_PROVIDER_ORDER=gemini`, in which case only that provider's key is required.

Free-tier availability and limits can change. Check the current
[Groq rate limits](https://console.groq.com/docs/rate-limits),
[Groq models](https://console.groq.com/docs/models), and
[Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) and
[rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) before running
the 40 API calls normally required by a complete evaluation.

Never commit `.env` or share the key in source code, screenshots, or reports.

## Configuration

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | Yes when Groq is configured | None | GroqCloud API key |
| `GROQ_GENERATION_MODEL` | No | `llama-3.3-70b-versatile` | Primary email generation model |
| `GROQ_JUDGE_MODEL` | No | `openai/gpt-oss-20b` | Strict structured-output judge |
| `GROQ_MIN_REQUEST_INTERVAL_SECONDS` | No | `6` | Minimum delay between Groq calls |
| `GEMINI_API_KEY` | Yes when Gemini is configured | None | Google AI Studio API key |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Fallback generation model |
| `GEMINI_JUDGE_MODEL` | No | `gemini-2.5-flash-lite` | Fallback judge model |
| `GEMINI_MIN_REQUEST_INTERVAL_SECONDS` | No | `13` | Minimum delay between Gemini calls |
| `GEMINI_REQUEST_TIMEOUT_SECONDS` | No | `120` | Maximum duration of an individual API request |
| `LLM_PROVIDER_ORDER` | No | `groq,gemini` | Primary and fallback provider order |

Generation uses temperature `0.4`. Judging uses temperature `0.0` for greater
consistency. Each call is attempted up to three times with exponential backoff
and the provider's server-provided retry delay when available. Authentication,
malformed-request, permission, and unavailable-model errors fail immediately;
retryable quota, connection, 422, and server errors can trigger fallback.

## Run the email assistant

```bash
pixi run app
```

The CLI asks for:

1. Intent.
2. One or more key facts, entered one per line.
3. Tone.
4. Strategy A or B.

Strategy A is the default and recommended user-facing strategy.

## Prompt engineering approach

### Strategy A - advanced

The model acts as a senior executive communication assistant. The prompt:

- supplies two complete input-to-email examples;
- requires subject, greeting, coherent body, closing, and `[Your Name]`;
- requires every key fact and prohibits invented facts;
- explicitly controls tone;
- tells the model to output only the final email and not chain-of-thought.

### Strategy B - baseline

The baseline only asks for a professional email that includes the supplied facts
and tone. It intentionally omits few-shot examples and strong structural rules.

Both strategies use the same configured generation route. This makes prompt
strategy the controlled independent variable when no fallback occurs. The
primary judge is `openai/gpt-oss-20b`, which supports Groq strict structured
outputs. If fallback occurs, the run is completed but explicitly marked as a
mixed-provider, non-controlled comparison.

## Custom evaluation metrics

Exactly three metrics are implemented:

### 1. Fact Coverage Score

The judge returns one status per required fact:

- `accurate` = 1.0
- `partial` = 0.5
- `missing` = 0.0
- `contradicted` = 0.0

Python averages those credits. This preserves paraphrase recognition while
preventing the judge from silently omitting difficult facts.

### 2. Tone Match Score

The judge assigns 1-5:

- 1: completely wrong
- 2: mostly wrong
- 3: acceptable but inconsistent
- 4: mostly matches
- 5: strongly matches

The final normalized score is `(raw - 1) / 4`.

### 3. Professional Email Quality Score

This hybrid metric averages:

- the judge's normalized 1-5 assessment of subject, greeting, body coherence,
  concision, closing, and grammar/fluency; and
- an automated score that equally weights structural completeness, placeholder
  discipline, and concision relative to the human reference email.

The automated component requires a subject, greeting, and closing; reduces its
placeholder subscore by `0.1` for each bracketed placeholder beyond
`[Your Name]`, with a `0.5` floor; and uses fixed word-count ratio bands from
`1.0` at no more than `1.25x` the reference length to `0.0` above `2.0x`.
This makes unnecessary verbosity and unresolved template fields measurable
instead of relying only on a broad LLM quality rating.

The overall score for a valid result is the unweighted mean of the three
normalized metrics. Strategy averages exclude failed records and always report
evaluated and error counts.

## Run the evaluation

Configure the provider keys, then run:

```bash
pixi run evaluate
```

This performs 20 generations and 20 judge calls: both strategies for every
scenario. With the default Groq pacing, allow roughly 4 minutes. It writes:

- `outputs/generated_emails.json` - generated email or generation error for each
  scenario and strategy.
- `outputs/evaluation_results.csv` - flat raw scores and fact assessments.
- `outputs/evaluation_results.json` - metric definitions, all records, judge
  details, errors, and strategy averages.

If Groq exhausts retryable attempts, the same operation is attempted with
Gemini. If both providers fail, the combined error is recorded and the next case
runs.
The project never inserts a mock score into live evaluation files. Results are
checkpointed after every scenario-strategy pair. If a run is interrupted or
rate-limited, resume it without repeating completed calls:

```bash
pixi run evaluate-resume
```

Resume mode reuses a generated email or judge score only when its recorded
provider and model remain in the active configuration. Legacy checkpoints
without provider metadata are inferred as Gemini records. Changing a configured
model invalidates only the affected saved work.

## Generate the final report

After evaluation:

```bash
pixi run report
```

This reads `outputs/evaluation_results.json` and writes:

- `outputs/final_report.md`
- `outputs/final_report.pdf`

Both reports include the two prompt templates, metric definitions and logic,
raw results, strategy averages, comparative analysis, and the production
recommendation. Report generation is intentionally strict:
it refuses to create final artifacts unless all 20 scenario-strategy records are
scored, each strategy has 10 evaluated scenarios, and there are zero errors.
Completed fallback runs are allowed but receive prominent mixed-provider
warnings and a qualified recommendation.

## Run offline verification

```bash
pixi run test
```

Tests cover:

- scenario and input validation;
- advanced and baseline prompts;
- judge JSON parsing and semantic fact-index validation;
- score normalization and aggregation;
- retry exhaustion and malformed JSON recovery;
- Groq strict structured output and retry/error classification;
- automatic fallback and combined provider failures;
- provider metadata, mixed-run warnings, and legacy resume behavior;
- continuation after an individual API failure;
- all 20 paths in an offline end-to-end evaluation;
- CLI interaction;
- Markdown and PDF report generation.

The fake client is test-only and writes artifacts to pytest temporary
directories. It is never used to populate `outputs/`.

## Production recommendation

Before measured results exist, Strategy A is the safer default because it places
explicit controls around structure, tone, and factual fidelity. The generated
report replaces that design-based recommendation with a data-based comparison
after a successful live evaluation.

For production use, also add authentication, request limits, structured audit
logging without secrets, human approval for sensitive messages, prompt/version
tracking, and monitoring for model or rubric drift.

## Limitations

- LLM-as-a-judge evaluation can vary even at temperature `0.0`.
- Automatic fallback improves completion reliability but can weaken the
  controlled nature of an A/B evaluation; fallback runs are flagged.
- Human reference emails are context for quality; they are not semantic ground
  truth for every valid wording.
- The CLI uses placeholders because recipient and sender names are not among the
  required inputs.
- Groq and Gemini model availability, pricing, and free-tier quotas can change.

## Official provider references

- [Groq OpenAI compatibility](https://console.groq.com/docs/openai)
- [Groq structured outputs](https://console.groq.com/docs/structured-outputs)
- [Groq rate limits](https://console.groq.com/docs/rate-limits)
- [Python quickstart](https://ai.google.dev/gemini-api/docs/quickstart)
- [Structured outputs](https://ai.google.dev/gemini-api/docs/structured-output)
- [Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
