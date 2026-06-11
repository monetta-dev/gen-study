# Entrance Exam Question Generation & Verification API

Python backend API for generating and validating Japanese university entrance exam questions with independent model agents.

Supported subjects:

- `mathematics`
- `physics`
- `chemistry`

This API is text-only. It generates and validates Japanese problem/solution text; image and visual generation are handled by a separate API.
Requests containing `allow_visuals` or `allow_diagram` are rejected by this API.

## Requirements

- Python 3.11+
- `OPENCODE_GO_API_KEY` in `.env` or the shell environment
- Optional: `OPENCODE_GO_MODEL`, defaults to `qwen3.7-max`
- Optional future visual API setting: `VISUAL_VALIDATOR_MODEL`, defaults to `qwen3.7-plus`
- Optional visual API setting: `OPENAI_API_KEY`, required for `POST /visuals/generate`
- Optional visual API setting: `OPENAI_IMAGE_MODEL`, defaults to `gpt-image-2`
- Optional visual API setting: `OPENAI_IMAGE_SIZE`; blank means omitted
- Optional visual API setting: `OPENAI_IMAGE_QUALITY`; blank means omitted
- Optional visual API setting: `OPENAI_IMAGE_N`; blank means omitted
- Optional visual API setting: `OPENAI_IMAGE_RETRY_ATTEMPTS`, defaults to `1`
- Optional visual API setting: `OPENAI_IMAGE_RETRY_DELAY_SECONDS`, defaults to `60`
- Optional visual API setting: `VISUAL_REPAIR_ATTEMPTS`, defaults to `2`
- Optional: `OPENCODE_GO_BASE_URL`, defaults to `https://opencode.ai/zen/go/v1`
- Optional: `OPENCODE_GO_API_STYLE`; normally omitted and inferred from the model
- Optional: `OPENCODE_GO_TIMEOUT_SECONDS`, defaults to `300`
- Optional: `OPENCODE_GO_MAX_TOKENS`, defaults to `32768`
- Optional: `OPENCODE_GO_STREAM`, defaults to `true` for Anthropic-compatible `/messages`
- Optional: `OPENCODE_GO_THINKING_ENABLED`, defaults to `true`
- Optional: `OPENCODE_GO_THINKING_BUDGET_TOKENS`, defaults to `16384`
- Optional: `OPENCODE_GO_THINKING_DISPLAY`, defaults to `omitted`
- Optional: `OPENCODE_GO_RETRY_ATTEMPTS`, defaults to `1`
- Optional: `OPENCODE_GO_RETRY_DELAY_SECONDS`, defaults to `10`
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

Create `.env`:

```bash
cp .env.example .env
```

Then edit `.env` and set your actual API key:

```dotenv
OPENCODE_GO_API_KEY=your_opencode_go_api_key_here
OPENCODE_GO_MODEL=qwen3.7-max
VISUAL_VALIDATOR_MODEL=qwen3.7-plus
OPENCODE_GO_STREAM=true
OPENCODE_GO_TIMEOUT_SECONDS=300
OPENCODE_GO_THINKING_ENABLED=true
OPENCODE_GO_THINKING_DISPLAY=omitted
OPENCODE_GO_RETRY_ATTEMPTS=1
OPENCODE_GO_RETRY_DELAY_SECONDS=10
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_IMAGE_MODEL=gpt-image-2
OPENAI_IMAGE_SIZE=
OPENAI_IMAGE_QUALITY=
OPENAI_IMAGE_N=
OPENAI_IMAGE_RETRY_ATTEMPTS=1
OPENAI_IMAGE_RETRY_DELAY_SECONDS=60
VISUAL_REPAIR_ATTEMPTS=2
MAX_REPAIR_ATTEMPTS=
GENERATION_TRACE=false
API_HOST=127.0.0.1
API_PORT=8000
API_DEBUG=false
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

The trace shows the prompts sent to each agent, the JSON content returned by each agent, validator decisions, adjudicator decisions, and repair exchanges. It does not expose private model thinking tokens.

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

The backend is standardized on OpenCode Go. For Qwen3.7 Max, configure Render or
`.env` like this:

```dotenv
OPENCODE_GO_API_KEY=your_opencode_go_api_key
OPENCODE_GO_MODEL=qwen3.7-max
OPENCODE_GO_STREAM=true
OPENCODE_GO_THINKING_ENABLED=true
OPENCODE_GO_THINKING_DISPLAY=omitted
```

This calls:

```text
POST https://opencode.ai/zen/go/v1/messages
```

The client infers the Anthropic-compatible `/messages` endpoint for
`qwen3.7-max`. It uses internal SSE streaming by default, then collects the
streamed text and returns the same normal JSON response from this API.

Thinking is sent as:

```json
{"thinking":{"type":"enabled","budget_tokens":16384,"display":"omitted"}}
```

`OPENCODE_GO_THINKING_BUDGET_TOKENS` must be less than
`OPENCODE_GO_MAX_TOKENS`. These are optional advanced overrides; the defaults
are intentionally high now that streaming avoids the Cloudflare read-timeout
problem.

OpenCode Go models currently handled as Anthropic-compatible `/messages`:

- `qwen3.7-max`
- `qwen3.7-plus`
- `qwen3.6-plus`
- `minimax-m3`
- `minimax-m2.7`
- `minimax-m2.5`

For visual generation, the API first asks a visual instruction agent whether a
diagram, graph, chemical structure, or explanatory figure is needed for the
completed problem and solution. If needed, it sends that final natural-language
prompt to OpenAI's Image API using `OPENAI_IMAGE_MODEL`, then validates the PNG
with `qwen3.7-plus` through OpenCode Go.

Use `qwen3.7-plus` as the default image-recognition validator model:

```dotenv
VISUAL_VALIDATOR_MODEL=qwen3.7-plus
```

Configure the OpenAI image generation side separately:

```dotenv
OPENAI_API_KEY=your_openai_api_key
OPENAI_IMAGE_MODEL=gpt-image-2
OPENAI_IMAGE_SIZE=
OPENAI_IMAGE_QUALITY=
OPENAI_IMAGE_N=
OPENAI_IMAGE_RETRY_ATTEMPTS=1
OPENAI_IMAGE_RETRY_DELAY_SECONDS=60
VISUAL_REPAIR_ATTEMPTS=2
```

The text-only `/generate` endpoint does not read these visual settings. The
visual API uses a separate OpenCode Go client for image validation so validator
cost and quality can be tuned independently from the main text generation model.

OpenCode Go models currently handled as OpenAI-compatible `/chat/completions`:

- `glm-5.1`
- `glm-5`
- `kimi-k2.6`
- `kimi-k2.5`
- `mimo-v2.5`
- `mimo-v2.5-pro`

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

When either validator rejects a candidate, `Adjudicator` first determines whether the finding is valid:

- `repair_required`: send the candidate and validator issues to `RepairAgent`.
- `validator_false_positive`: ignore the validator finding and continue or accept.
- `clarify_definition_required`: send the candidate to `RepairAgent` with a clarification focus.

The repair agent revises the same candidate and validation runs again. By default there is no hard repair limit; set `MAX_REPAIR_ATTEMPTS` to a non-negative integer if you need a production cost guardrail.

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

If a visual is needed, the response includes one PNG asset as base64:

```json
{
  "status": "success",
  "visual_needed": true,
  "visual_validation_status": "pass",
  "visual_assets": [
    {
      "mime_type": "image/png",
      "base64": "...",
      "placement": "problem",
      "purpose": "diagram",
      "prompt": "..."
    }
  ]
}
```

If the vision validator is uncertain, the API preserves the generated image and
returns `visual_validation_status: "uncertain"` instead of automatically
rewriting a likely acceptable figure.

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

## Test

```bash
python3 -m unittest discover -s tests
```
