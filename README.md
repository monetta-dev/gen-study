# Entrance Exam Question Generation & Verification API

Python backend API for generating and validating Japanese university entrance exam questions with independent model agents.

Supported subjects:

- `mathematics`
- `physics`
- `chemistry`

`POST /generate` is text-only. It generates and validates Japanese
problem/solution text; visual generation is handled separately by
`POST /visuals/generate`, which generates TikZ, compiles it in the Docker
runtime, and returns rendered PNG assets. Requests to `/generate` containing
`allow_visuals` or `allow_diagram` are rejected.

## Requirements

- Python 3.11+
- `OPENROUTER_API_KEY` in `.env.local` or the shell environment
- Optional: `OPENROUTER_BASE_URL`, defaults to `https://openrouter.ai/api/v1`
- Optional visual API setting: `VISUAL_TIKZ_REPAIR_ATTEMPTS`, defaults to `2`
- Optional: `OPENROUTER_MODEL`, defaults to `qwen/qwen3.7-max`
- Optional: `OPENROUTER_TIMEOUT_SECONDS`, defaults to `300`
- Optional: `OPENROUTER_MAX_TOKENS`, defaults to `32768`
- Optional: `OPENROUTER_STREAM`, defaults to `true`
- Optional: `OPENROUTER_THINKING_ENABLED`, defaults to `true`
- Optional: `OPENROUTER_THINKING_BUDGET_TOKENS`, defaults to `16384`
- Optional: `OPENROUTER_RETRY_ATTEMPTS`, defaults to `3`
- Optional: `OPENROUTER_RETRY_DELAY_SECONDS`, defaults to `10`
- Optional narrow-agent thinking-budget overrides such as
  `COMPLIANCE_VALIDATOR_THINKING_BUDGET_TOKENS` and
  `VISUAL_TIKZ_VALIDATOR_THINKING_BUDGET_TOKENS`; blank means use the main
  Qwen thinking settings
- Optional per-agent model overrides such as `GENERATOR_MODEL`,
  `COMPLIANCE_VALIDATOR_MODEL`, and `VISUAL_TIKZ_MODEL`; blank means use
  `OPENROUTER_MODEL`
- Optional: `MAX_REPAIR_ATTEMPTS`; blank means no hard repair limit
- Optional: `GENERATION_TRACE`, defaults to `false`
- Optional: `API_HOST`, defaults to `127.0.0.1`
- Optional: `API_PORT`, defaults to `8000`
- Optional: `API_DEBUG`, defaults to `false`
- Optional: `API_MAX_BODY_BYTES`, defaults to `65536`
- Optional: `API_ALLOWED_ORIGIN`, defaults to `*`
- Optional: `GEN_STUDY_API_KEY`; when set, `POST /generate` requires this key

The implementation uses only the Python standard library.

## Run

Create `.env.local`:

```bash
cp .env.example .env.local
```

Then edit `.env.local` and set your actual API key:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key_here
VISUAL_TIKZ_REPAIR_ATTEMPTS=2
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=qwen/qwen3.7-max
OPENROUTER_STREAM=true
OPENROUTER_TIMEOUT_SECONDS=300
OPENROUTER_MAX_TOKENS=32768
OPENROUTER_THINKING_ENABLED=true
OPENROUTER_THINKING_BUDGET_TOKENS=16384
OPENROUTER_RETRY_ATTEMPTS=3
OPENROUTER_RETRY_DELAY_SECONDS=10
GENERATOR_MODEL=
REPAIR_MODEL=
CRITICAL_VALIDATOR_MODEL=
COMPLIANCE_VALIDATOR_MODEL=
VISUAL_INSTRUCTION_MODEL=
VISUAL_TIKZ_MODEL=
VISUAL_TIKZ_COMPILE_MODEL=
VISUAL_TIKZ_VALIDATOR_MODEL=
VISUAL_TIKZ_REPAIR_MODEL=
COMPOSITE_LAYOUT_MODEL=
GENERATOR_MAX_TOKENS=
GENERATOR_THINKING_BUDGET_TOKENS=
REPAIR_THINKING_BUDGET_TOKENS=
CRITICAL_VALIDATOR_THINKING_BUDGET_TOKENS=
COMPLIANCE_VALIDATOR_THINKING_BUDGET_TOKENS=
VISUAL_INSTRUCTION_THINKING_BUDGET_TOKENS=
COMPOSITE_LAYOUT_THINKING_BUDGET_TOKENS=
VISUAL_TIKZ_THINKING_BUDGET_TOKENS=
VISUAL_TIKZ_COMPILE_THINKING_BUDGET_TOKENS=
VISUAL_TIKZ_VALIDATOR_THINKING_BUDGET_TOKENS=
VISUAL_TIKZ_REPAIR_THINKING_BUDGET_TOKENS=
MAX_REPAIR_ATTEMPTS=
GENERATION_TRACE=false
VISUAL_TIKZ_REPAIR_ATTEMPTS=2
VISUAL_RENDER_ENABLED=true
VISUAL_COMPILE_TIMEOUT_SECONDS=30
VISUAL_COMPILE_MAX_ACTIONS=5
API_HOST=127.0.0.1
API_PORT=8000
API_DEBUG=false
API_MAX_BODY_BYTES=65536
API_ALLOWED_ORIGIN=*
GEN_STUDY_API_KEY=
```

Start the server:

```bash
python3 -m mathgen.api
```

The server listens on `http://127.0.0.1:8000` by default.

To see each Generator and Validator exchange in the server terminal, set:

```dotenv
GENERATION_TRACE=true
```

The trace shows the prompts sent to each agent, the JSON content returned by each agent, validator decisions, and repair exchanges. It does not expose private model thinking tokens.

For local debugging, enable detailed API errors:

```dotenv
API_DEBUG=true
```

Then restart:

```bash
python3 -m mathgen.api
```

Unexpected failures return a `request_id` in the browser response and log the
same ID in the server terminal. With `API_DEBUG=true`, the browser response also
includes the exception message. Keep `API_DEBUG=false` in production.

## Model Provider

The backend uses OpenRouter with Qwen 3.7. Max is the default; Plus can be
selected globally or per agent.

For Qwen 3.7 Max everywhere, configure Render or `.env.local` like this:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=qwen/qwen3.7-max
OPENROUTER_STREAM=true
OPENROUTER_THINKING_ENABLED=true
```

For Qwen 3.7 Plus everywhere:

```dotenv
OPENROUTER_MODEL=qwen/qwen3.7-plus
```

For a per-agent override, leave `OPENROUTER_MODEL` as the default and set the
specific role:

```dotenv
GENERATOR_MODEL=qwen3.7-max
COMPLIANCE_VALIDATOR_MODEL=qwen3.7-plus
VISUAL_TIKZ_VALIDATOR_MODEL=qwen3.7-plus
```

The provider model IDs are:

```text
qwen/qwen3.7-max
qwen/qwen3.7-plus
```

This calls:

```text
POST https://openrouter.ai/api/v1/chat/completions
```

The client uses the OpenAI-compatible Chat Completions endpoint. It uses internal
SSE streaming by default, then collects the streamed text and returns the same
normal JSON response from this API.

Thinking is sent as:

```json
{"reasoning":{"enabled":true,"max_tokens":16384,"exclude":true}}
```

When thinking is disabled, the client sends `{"reasoning":{"enabled":false}}`.
`OPENROUTER_THINKING_BUDGET_TOKENS` controls reasoning tokens, while
`OPENROUTER_MAX_TOKENS` independently controls final-answer tokens.

Role-specific thinking-budget overrides are supported for every agent. Leave
these blank to keep the main high-quality Qwen thinking budget everywhere.

```text
GENERATOR_THINKING_BUDGET_TOKENS
REPAIR_THINKING_BUDGET_TOKENS
CRITICAL_VALIDATOR_THINKING_BUDGET_TOKENS
COMPLIANCE_VALIDATOR_THINKING_BUDGET_TOKENS
VISUAL_INSTRUCTION_THINKING_BUDGET_TOKENS
VISUAL_TIKZ_THINKING_BUDGET_TOKENS
VISUAL_TIKZ_VALIDATOR_THINKING_BUDGET_TOKENS
VISUAL_TIKZ_REPAIR_THINKING_BUDGET_TOKENS
```

The generator, repair agent, critical validator, and TikZ generator should
generally stay on stronger settings unless you have tested quality carefully.

The same mechanism can tune the Generator if you test Qwen 3.7 Plus there:

```dotenv
GENERATOR_MODEL=qwen3.7-plus
GENERATOR_THINKING_BUDGET_TOKENS=8192
GENERATOR_MAX_TOKENS=24576
```

Leave these blank to use the global `OPENROUTER_THINKING_BUDGET_TOKENS` and
`OPENROUTER_MAX_TOKENS`. If a call repeatedly times out or yields an empty
stream, reduce the role-specific thinking budget after quality testing or
increase `OPENROUTER_TIMEOUT_SECONDS`.

Role-specific model overrides are also supported:

```text
GENERATOR_MODEL
REPAIR_MODEL
CRITICAL_VALIDATOR_MODEL
COMPLIANCE_VALIDATOR_MODEL
VISUAL_INSTRUCTION_MODEL
VISUAL_TIKZ_MODEL
VISUAL_TIKZ_COMPILE_MODEL
VISUAL_TIKZ_VALIDATOR_MODEL
VISUAL_TIKZ_REPAIR_MODEL
COMPOSITE_LAYOUT_MODEL
```

For visual generation, the API first asks a visual instruction agent whether a
diagram, graph, chemical structure, or explanatory figure is needed for the
completed problem and solution. It can request multiple separate visuals for one
problem. Visual assets are generated as TikZ source, compiled to PDF by an
agentic LaTeX compile loop, and returned as PNG for GUI display:

```dotenv
VISUAL_TIKZ_REPAIR_ATTEMPTS=2
VISUAL_RENDER_ENABLED=true
VISUAL_COMPILE_TIMEOUT_SECONDS=30
VISUAL_COMPILE_MAX_ACTIONS=5
PROBLEM_RENDER_TIMEOUT_SECONDS=180
PROBLEM_RENDER_DPI=180
COMPOSITE_RENDER_ENABLED=false
```

Each visual instruction includes a frozen `visual_contract` generated locally
from that problem and solution. The contract records the visual intent, diagram
type, required elements, interpretation choices, acceptance criteria, and
non-goals for that specific visual. TikZ generation, validation, repair, and
compilation all receive the same contract. Validators must check only blocking
contract violations and must not introduce new optional criteria or alternative
diagram conventions during repair validation.

Each TikZ source is checked by a source-level TikZ validator. If validation
requires repair, the TikZ repair agent revises the same source and validation
runs again until it passes or reaches `VISUAL_TIKZ_REPAIR_ATTEMPTS`. After that,
`TikZCompileAgent` chooses allowed LaTeX commands, observes stdout/stderr/logs
and generated files, and keeps iterating until a PDF is produced or it concludes
that the TikZ source itself needs repair. The compiled PDF is converted to PNG
with `pdftoppm`. The text-only `/generate` endpoint does not read these visual
settings.

`POST /generate` can optionally render the final problem and solution as PNG
pages by including `render_output: true`, `render_png: true`, or
`include_rendered_output: true` in the request. This uses LuaLaTeX and
`pdftoppm` in the same Docker runtime and returns `rendered_output.assets`.

The Docker image installs `texlive-full` and `poppler-utils`; `texlive-full`
covers the LaTeX package surface, while `poppler-utils` provides `pdftoppm` for
PNG conversion.

Keep `OPENROUTER_STREAM=true` for long thinking calls so the provider can send
SSE events while it works. Retryable provider errors such as 429 and common
5xx responses are retried according to `OPENROUTER_RETRY_ATTEMPTS` and
`OPENROUTER_RETRY_DELAY_SECONDS`.

## Production

For a VPS behind a reverse proxy, keep the API bound to localhost:

```dotenv
API_HOST=127.0.0.1
API_PORT=8000
API_DEBUG=false
GEN_STUDY_API_KEY=replace_with_a_long_random_secret
```

For a PaaS/container platform, bind to all interfaces and use the platform-provided port:

```bash
API_HOST=0.0.0.0 API_PORT="${PORT:-8000}" python3 -m mathgen.api
```

For Docker deployment with LaTeX rendering:

```bash
docker build -t gen-study .
docker run --env-file .env.production -p 8000:8000 gen-study
```

If `GEN_STUDY_API_KEY` is set, include either header:

```bash
X-API-Key: replace_with_a_long_random_secret
```

or:

```bash
Authorization: Bearer replace_with_a_long_random_secret
```

`GET /health` does not require authentication so load balancers can check the service.
The GUI is served from `GET /` and static assets are served from `/viewer/`.

## Validation Flow

Each generation attempt is checked by two focused validators:

1. `CriticalValidator` checks genuine correctness issues: calculation errors, invalid equations, unit errors, logical gaps, contradictions, missing conditions, and problem/solution inconsistencies.
2. `ComplianceValidator` checks system acceptance: Japanese curriculum scope, Japanese output, subject/topic alignment, requested level alignment, and text-only policy.

The two validators run in parallel because they are independent. When either
validator rejects a candidate, `RepairAgent` receives the same candidate plus
the validator issues and revises it directly. Validation then runs again. By
default there is no hard repair limit; set `MAX_REPAIR_ATTEMPTS` to a
non-negative integer if you need a production cost guardrail.

## Request

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"subject":"mathematics","level":"University of Tokyo","topic":"Probability Recurrence Relations"}'
```

Request rendered PNG pages for distribution:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"subject":"mathematics","level":"University of Tokyo","topic":"Probability Recurrence Relations","render_output":true}'
```

The response then includes:

```json
{
  "status": "success",
  "problem": "...",
  "solution": "...",
  "rendered_output": {
    "status": "compiled",
    "format": "png",
    "assets": [
      {
        "kind": "problem",
        "page": 1,
        "title": "問題",
        "mime_type": "image/png",
        "png_base64": "..."
      },
      {
        "kind": "solution",
        "page": 1,
        "title": "解答",
        "mime_type": "image/png",
        "png_base64": "..."
      }
    ]
  }
}
```

Generate a visual asset from a completed problem and solution:

```bash
curl -X POST http://127.0.0.1:8000/visuals/generate \
  -H 'Content-Type: application/json' \
  -d '{"subject":"physics","level":"University of Tokyo","topic":"Mechanics","problem":"問題文...","solution":"解答..."}'
```

If visuals are needed, the response includes one or more rendered PNG assets
with the original TikZ source retained:

```json
{
  "status": "success",
  "visual_backend": "tikz",
  "visual_needed": true,
  "visual_validation_status": "pass",
  "visual_assets": [
    {
      "mime_type": "image/png",
      "id": "visual_1",
      "placement": "problem",
      "purpose": "diagram",
      "png_base64": "...",
      "tikz_source": "\\begin{tikzpicture}...\\end{tikzpicture}",
      "compile_status": "compiled",
      "compile_engine": "lualatex"
    }
  ]
}
```

With API key authentication enabled:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: replace_with_a_long_random_secret' \
  -d '{"subject":"mathematics","level":"University of Tokyo","topic":"Probability Recurrence Relations"}'
```

Physics example:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"subject":"physics","level":"University of Tokyo","topic":"Mechanics"}'
```

Chemistry example:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"subject":"chemistry","level":"University of Tokyo","topic":"Acid-Base Titration"}'
```

Development full-pipeline example:

```bash
curl -sS -X POST http://127.0.0.1:8000/dev/generate-full \
  -H 'Content-Type: application/json' \
  -d '{"subject":"physics","level":"University of Tokyo","topic":"Mechanics"}' \
  -o full_response.json
```

This endpoint is available only when `COMPOSITE_RENDER_ENABLED=true`. It
generates the text, renders problem/solution PNGs, generates visual PNGs, and
creates composite problem/solution PNGs in one request.

## Successful Response

```json
{
  "status": "success",
  "problem": "Japanese question text including $LaTeX$",
  "solution": "Japanese solution and explanation including $LaTeX$"
}
```

If validation and repair attempts still fail, the API returns HTTP 422 with `status: "failed"` and the latest validator issues.

For backward compatibility, requests without `subject` are treated as `mathematics`.

## GUI

Start the API and open:

```text
http://127.0.0.1:8000/
```

The GUI lets users enter `subject`, `level`, and `topic`, call `POST /generate`,
and render the returned problem and solution with LaTeX formatting. If
`GEN_STUDY_API_KEY` is configured, enter the same key in the GUI connection
settings.

On Render, open the deployed service URL directly:

```text
https://your-service-name.onrender.com/
```

The viewer uses MathJax from a CDN, so LaTeX rendering needs browser access to
that script. The raw text still displays if MathJax is unavailable.

By default, the GUI is production-safe: it does not request process traces, hides
the process panel, and the plain-text panel shows only the final problem and
solution before MathJax rendering. Problem/solution PNG rendering and composite
PNG rendering remain available as API request options only; they are not exposed
in the GUI. Enable development-only GUI features with environment variables:

```dotenv
GUI_INCLUDE_TRACE=true
GUI_SHOW_PROCESS=true
GUI_SHOW_FULL_RAW_RESPONSE=true
```

Keep `COMPOSITE_RENDER_ENABLED=false` on 512 MB Render services.

## Test

```bash
python3 -m unittest discover -s tests
```
