# Entrance Exam Question Generation & Verification API

Python backend API for generating and validating Japanese university entrance exam questions with independent model agents.

Supported subjects:

- `mathematics`
- `physics`
- `chemistry`

`POST /generate` is text-only. It generates and validates Japanese
problem/solution text; visual generation is handled separately by
`POST /visuals/generate`, which returns TikZ source. Requests to `/generate`
containing `allow_visuals` or `allow_diagram` are rejected.

## Requirements

- Python 3.11+
- `OPENCODE_GO_API_KEY` in `.env.local` or the shell environment
- Optional visual API setting: `VISUAL_TIKZ_REPAIR_ATTEMPTS`, defaults to `2`
- Optional: `OPENCODE_GO_BASE_URL`, defaults to `https://opencode.ai/zen/go/v1`
- Optional: `OPENCODE_GO_MODEL`, defaults to `qwen3.7-max`; allowed values are
  `qwen3.7-max` and `qwen3.7-plus`
- Optional: `OPENCODE_GO_TIMEOUT_SECONDS`, defaults to `300`
- Optional: `OPENCODE_GO_MAX_TOKENS`, defaults to `32768`
- Optional: `OPENCODE_GO_STREAM`, defaults to `true`
- Optional: `OPENCODE_GO_THINKING_ENABLED`, defaults to `true`
- Optional: `OPENCODE_GO_THINKING_BUDGET_TOKENS`, defaults to `16384`
- Optional: `OPENCODE_GO_THINKING_DISPLAY`, defaults to `omitted`
- Optional: `OPENCODE_GO_RETRY_ATTEMPTS`, defaults to `3`
- Optional: `OPENCODE_GO_RETRY_DELAY_SECONDS`, defaults to `10`
- Optional narrow-agent thinking-budget overrides such as
  `COMPLIANCE_VALIDATOR_THINKING_BUDGET_TOKENS` and
  `VISUAL_TIKZ_VALIDATOR_THINKING_BUDGET_TOKENS`; blank means use the main
  Qwen thinking settings
- Optional per-agent model overrides such as `GENERATOR_MODEL`,
  `COMPLIANCE_VALIDATOR_MODEL`, and `VISUAL_TIKZ_MODEL`; blank means use
  `OPENCODE_GO_MODEL`
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
OPENCODE_GO_API_KEY=your_opencode_go_api_key_here
VISUAL_TIKZ_REPAIR_ATTEMPTS=2
OPENCODE_GO_BASE_URL=https://opencode.ai/zen/go/v1
OPENCODE_GO_MODEL=qwen3.7-max
OPENCODE_GO_STREAM=true
OPENCODE_GO_TIMEOUT_SECONDS=300
OPENCODE_GO_MAX_TOKENS=32768
OPENCODE_GO_THINKING_ENABLED=true
OPENCODE_GO_THINKING_BUDGET_TOKENS=16384
OPENCODE_GO_THINKING_DISPLAY=omitted
OPENCODE_GO_ANTHROPIC_VERSION=2023-06-01
OPENCODE_GO_USER_AGENT=gen-study/0.1 Qwen client
OPENCODE_GO_RETRY_ATTEMPTS=3
OPENCODE_GO_RETRY_DELAY_SECONDS=10
GENERATOR_MODEL=
REPAIR_MODEL=
CRITICAL_VALIDATOR_MODEL=
COMPLIANCE_VALIDATOR_MODEL=
VISUAL_INSTRUCTION_MODEL=
VISUAL_TIKZ_MODEL=
VISUAL_TIKZ_VALIDATOR_MODEL=
VISUAL_TIKZ_REPAIR_MODEL=
GENERATOR_MAX_TOKENS=
GENERATOR_THINKING_BUDGET_TOKENS=
COMPLIANCE_VALIDATOR_THINKING_BUDGET_TOKENS=
VISUAL_INSTRUCTION_THINKING_BUDGET_TOKENS=
VISUAL_TIKZ_VALIDATOR_THINKING_BUDGET_TOKENS=
MAX_REPAIR_ATTEMPTS=
GENERATION_TRACE=false
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

The backend is standardized on OpenCode Go with Qwen 3.7. Max is the default;
Plus can be selected globally or per agent.

For Qwen 3.7 Max everywhere, configure Render or `.env.local` like this:

```dotenv
OPENCODE_GO_API_KEY=your_opencode_go_api_key
OPENCODE_GO_MODEL=qwen3.7-max
OPENCODE_GO_STREAM=true
OPENCODE_GO_THINKING_ENABLED=true
OPENCODE_GO_THINKING_DISPLAY=omitted
```

For Qwen 3.7 Plus everywhere:

```dotenv
OPENCODE_GO_MODEL=qwen3.7-plus
```

For a per-agent override, leave `OPENCODE_GO_MODEL` as the default and set the
specific role:

```dotenv
GENERATOR_MODEL=qwen3.7-max
COMPLIANCE_VALIDATOR_MODEL=qwen3.7-plus
VISUAL_TIKZ_VALIDATOR_MODEL=qwen3.7-plus
```

Supported model values are only:

```text
qwen3.7-max
qwen3.7-plus
```

This calls:

```text
POST https://opencode.ai/zen/go/v1/messages
```

The client uses the Anthropic-compatible `/messages` endpoint. It uses internal
SSE streaming by default, then collects the streamed text and returns the same
normal JSON response from this API.

Thinking is sent as:

```json
{"thinking":{"type":"enabled","budget_tokens":16384,"display":"omitted"}}
```

`OPENCODE_GO_THINKING_BUDGET_TOKENS` must be less than
`OPENCODE_GO_MAX_TOKENS`. These are optional advanced overrides; the defaults
are intentionally high now that streaming avoids the Cloudflare read-timeout
problem.

Role-specific thinking-budget overrides are supported for narrow sub-agents,
for example `VISUAL_TIKZ_VALIDATOR_THINKING_BUDGET_TOKENS=4096`. Leave these
blank to keep the main high-quality Qwen thinking budget everywhere. The
generator, repair agent, and critical validator should generally stay on the
strongest settings unless you have tested quality carefully.

The same mechanism can tune the Generator if you test Qwen 3.7 Plus there:

```dotenv
GENERATOR_MODEL=qwen3.7-plus
GENERATOR_THINKING_BUDGET_TOKENS=8192
GENERATOR_MAX_TOKENS=24576
```

Leave these blank to use the global `OPENCODE_GO_THINKING_BUDGET_TOKENS` and
`OPENCODE_GO_MAX_TOKENS`. If `GENERATOR_MODEL=qwen3.7-plus` produces repeated
`HTTP 524` or empty-stream retries, the provider is not returning final text
within the current budget/timeout; either reduce the Generator budget after
quality testing or keep the Generator on `qwen3.7-max` and use Plus for narrower
agents.

Role-specific model overrides are also supported:

```text
GENERATOR_MODEL
REPAIR_MODEL
CRITICAL_VALIDATOR_MODEL
COMPLIANCE_VALIDATOR_MODEL
VISUAL_INSTRUCTION_MODEL
VISUAL_TIKZ_MODEL
VISUAL_TIKZ_VALIDATOR_MODEL
VISUAL_TIKZ_REPAIR_MODEL
```

For visual generation, the API first asks a visual instruction agent whether a
diagram, graph, chemical structure, or explanatory figure is needed for the
completed problem and solution. It can request multiple separate visuals for one
problem. Visual assets are generated as TikZ source:

```dotenv
VISUAL_TIKZ_REPAIR_ATTEMPTS=2
```

Each TikZ source is checked by a source-level TikZ validator. If validation
requires repair, the TikZ repair agent revises the same source and validation
runs again until it passes or reaches `VISUAL_TIKZ_REPAIR_ATTEMPTS`. The visual
API returns `tikz_source` only and does not compile LaTeX on the server. This
keeps the main API fast and avoids a heavy TeX runtime dependency. The text-only
`/generate` endpoint does not read this visual setting.

If OpenCode Go returns Cloudflare 1010 `browser_signature_banned`, the request
was blocked before reaching the LLM API. The client sends
`OPENCODE_GO_USER_AGENT` on all provider calls; try setting it to a stable
application identifier. If the same 1010 response persists, OpenCode must allow
server-side API clients or document the required client signature.

If OpenCode Go returns Cloudflare 524 `origin_response_timeout`, the model call
is taking too long for Cloudflare's edge timeout. Keep
`OPENCODE_GO_STREAM=true` so the provider can send SSE chunks while it works.
Retryable provider errors such as 524, 429, and common 5xx responses are retried
according to `OPENCODE_GO_RETRY_ATTEMPTS` and
`OPENCODE_GO_RETRY_DELAY_SECONDS`.

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

Generate a visual asset from a completed problem and solution:

```bash
curl -X POST http://127.0.0.1:8000/visuals/generate \
  -H 'Content-Type: application/json' \
  -d '{"subject":"physics","level":"University of Tokyo","topic":"Mechanics","problem":"問題文...","solution":"解答..."}'
```

If visuals are needed, the response includes one or more TikZ assets:

```json
{
  "status": "success",
  "visual_backend": "tikz",
  "visual_needed": true,
  "visual_validation_status": "pass",
  "visual_assets": [
    {
      "mime_type": "text/x-tex",
      "id": "visual_1",
      "placement": "problem",
      "purpose": "diagram",
      "tikz_source": "\\begin{tikzpicture}...\\end{tikzpicture}"
    }
  ]
}
```

The API does not render or compile the TikZ server-side. Use the GUI or your own
LaTeX toolchain to inspect the returned source.

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
solution before MathJax rendering. For local debugging, edit `viewer/config.js`:

```js
window.GEN_STUDY_CONFIG = {
  includeTrace: true,
  showProcess: true,
  showFullRawResponse: true
};
```

## Test

```bash
python3 -m unittest discover -s tests
```
