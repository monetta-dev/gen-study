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
- `LLM_API_KEY` or `DEEPSEEK_API_KEY` in `.env` or the shell environment
- Optional: `LLM_PROVIDER`, defaults to `deepseek`
- Optional: `LLM_API_STYLE`, defaults to `openai`; use `anthropic` for Anthropic-compatible `/messages`
- Optional: `LLM_BASE_URL`, falls back to `DEEPSEEK_BASE_URL`, then `https://api.deepseek.com`
- Optional: `LLM_MODEL`, falls back to `DEEPSEEK_MODEL`, then `deepseek-v4-pro`
- Optional: `LLM_TIMEOUT_SECONDS`, defaults to `300`
- Optional: `LLM_MAX_TOKENS`, defaults to `8192` for Anthropic-compatible `/messages`
- Optional: `LLM_STREAM`, defaults to `true` for OpenCode Go Anthropic-compatible `/messages`
- Optional: `LLM_USER_AGENT`, defaults to `gen-study/0.1 Python LLM client`
- Optional: `LLM_THINKING_MODE`, defaults to `auto`; use `enabled` to require a thinking parameter
- Optional: `LLM_THINKING_BUDGET_TOKENS`, defaults to `4096` for Anthropic-compatible thinking
- Optional: `LLM_THINKING_DISPLAY`, defaults to `omitted`
- Optional: `LLM_THINKING_FIELD`; normally leave empty so provider/model presets choose it
- Optional: `LLM_RETRY_ATTEMPTS`, defaults to `1` for retryable provider HTTP errors
- Optional: `LLM_RETRY_DELAY_SECONDS`, defaults to `10`
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
LLM_PROVIDER=opencode-go
LLM_API_STYLE=anthropic
LLM_API_KEY=your_opencode_go_api_key_here
LLM_BASE_URL=https://opencode.ai/zen/go/v1
LLM_MODEL=qwen3.7-max
LLM_TIMEOUT_SECONDS=300
LLM_MAX_TOKENS=4096
LLM_STREAM=true
LLM_USER_AGENT=gen-study/0.1 Python LLM client
LLM_THINKING_MODE=enabled
LLM_THINKING_BUDGET_TOKENS=1024
LLM_THINKING_DISPLAY=omitted
LLM_RETRY_ATTEMPTS=1
LLM_RETRY_DELAY_SECONDS=10
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

The backend supports two API styles:

- `LLM_API_STYLE=openai`: OpenAI-compatible `POST /chat/completions`
- `LLM_API_STYLE=anthropic`: Anthropic-compatible `POST /messages`

DeepSeek remains the default for backward compatibility, but you can switch
providers with `LLM_*` environment variables. OpenCode Go model IDs and endpoint
families are based on the OpenCode Go docs.

For OpenCode Go Qwen3.7 Max, configure Render or `.env` like this:

```dotenv
LLM_PROVIDER=opencode-go
LLM_API_KEY=your_opencode_go_api_key
LLM_MODEL=qwen3.7-max
LLM_TIMEOUT_SECONDS=300
LLM_MAX_TOKENS=4096
LLM_STREAM=true
LLM_USER_AGENT=gen-study/0.1 Python LLM client
LLM_THINKING_MODE=enabled
LLM_THINKING_BUDGET_TOKENS=1024
LLM_THINKING_DISPLAY=omitted
LLM_RETRY_ATTEMPTS=1
LLM_RETRY_DELAY_SECONDS=10
```

This calls:

```text
POST https://opencode.ai/zen/go/v1/messages
```

The client infers `LLM_API_STYLE=anthropic` and
`LLM_BASE_URL=https://opencode.ai/zen/go/v1` for `qwen3.7-max`. It also enables
internal SSE streaming by default for OpenCode Go `/messages`, then collects the
streamed text and returns the same normal JSON response from this API.

Thinking is sent as:

```json
{"thinking":{"type":"enabled","budget_tokens":1024,"display":"omitted"}}
```

`LLM_THINKING_BUDGET_TOKENS` must be less than `LLM_MAX_TOKENS`.
The `.env.example` uses a smaller `1024` thinking budget because OpenCode Go is
behind Cloudflare, which can return HTTP 524 if the origin does not complete a
response within about 120 seconds.

OpenCode Go models currently handled as Anthropic-compatible `/messages`:

- `qwen3.7-max`
- `qwen3.7-plus`
- `qwen3.6-plus`
- `minimax-m3`
- `minimax-m2.7`
- `minimax-m2.5`

OpenCode Go models currently handled as OpenAI-compatible `/chat/completions`:

- `deepseek-v4-pro`
- `deepseek-v4-flash`
- `glm-5.1`
- `glm-5`
- `kimi-k2.6`
- `kimi-k2.5`
- `mimo-v2.5`
- `mimo-v2.5-pro`

For OpenCode Go OpenAI-compatible models, use:

```dotenv
LLM_PROVIDER=opencode-go
LLM_API_KEY=your_opencode_go_api_key
LLM_MODEL=deepseek-v4-pro
LLM_THINKING_MODE=enabled
```

This calls:

```text
POST https://opencode.ai/zen/go/v1/chat/completions
```

For direct DeepSeek, `LLM_THINKING_MODE=enabled` sends:

```json
{"thinking":{"type":"enabled"}}
```

For Qwen through Alibaba DashScope OpenAI-compatible mode, configure Render or
`.env` like this:

```dotenv
LLM_PROVIDER=qwen
LLM_API_STYLE=openai
LLM_API_KEY=your_dashscope_api_key
LLM_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-max
LLM_THINKING_MODE=enabled
LLM_THINKING_FIELD=enable_thinking
```

If your Alibaba Cloud account/region uses the China endpoint, use:

```dotenv
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Use the exact Qwen model identifier shown in your DashScope console. The older
`DEEPSEEK_*` variables still work as a fallback.

If OpenCode Go returns Cloudflare 1010 `browser_signature_banned`, the request
was blocked before reaching the LLM API. The client sends `LLM_USER_AGENT` on
all provider calls; try setting it to a stable application identifier. If the
same 1010 response persists, OpenCode must allow server-side API clients or
document the required client signature.

If OpenCode Go returns Cloudflare 524 `origin_response_timeout`, the model call
is taking too long for Cloudflare's edge timeout. Keep `LLM_STREAM=true` so the
provider can send SSE chunks while it works. If 524 still occurs, keep thinking
enabled but lower the budget first:

```dotenv
LLM_MAX_TOKENS=4096
LLM_THINKING_BUDGET_TOKENS=1024
```

If 524 still occurs, try:

```dotenv
LLM_MAX_TOKENS=2048
LLM_THINKING_BUDGET_TOKENS=512
```

Retryable provider errors such as 524, 429, and common 5xx responses are retried
according to `LLM_RETRY_ATTEMPTS` and `LLM_RETRY_DELAY_SECONDS`.

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

The repair agent revises the same candidate and validation runs again. By default the API performs one initial generation and up to three repairs.

## Request

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"subject":"mathematics","level":"University of Tokyo","topic":"Probability Recurrence Relations"}'
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
