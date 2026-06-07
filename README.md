# Entrance Exam Question Generation & Verification API

Python backend API for generating and validating Japanese university entrance exam questions with independent DeepSeek V4 Pro agents.

Supported subjects:

- `mathematics`
- `physics`
- `chemistry`

This API is text-only. It generates and validates Japanese problem/solution text; image and visual generation are handled by a separate API.
Requests containing `allow_visuals` or `allow_diagram` are rejected by this API.

## Requirements

- Python 3.11+
- `DEEPSEEK_API_KEY` in `.env` or the shell environment
- Optional: `DEEPSEEK_BASE_URL`, defaults to `https://api.deepseek.com`
- Optional: `DEEPSEEK_MODEL`, defaults to `deepseek-v4-pro`
- Optional: `DEEPSEEK_TIMEOUT_SECONDS`, defaults to `300`
- Optional: `GENERATION_TRACE`, defaults to `false`
- Optional: `API_HOST`, defaults to `127.0.0.1`
- Optional: `API_PORT`, defaults to `8000`
- Optional: `API_DEBUG`, defaults to `false`
- Optional: `API_MAX_BODY_BYTES`, defaults to `65536`
- Optional: `GEN_STUDY_API_KEY`; when set, `POST /generate` requires this key

The implementation uses only the Python standard library.

## Run

Create `.env`:

```bash
cp .env.example .env
```

Then edit `.env` and set your actual API key:

```dotenv
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_TIMEOUT_SECONDS=300
GENERATION_TRACE=false
API_HOST=127.0.0.1
API_PORT=8000
API_DEBUG=false
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

The trace shows the prompts sent to each agent, the JSON content returned by each agent, validator decisions, adjudicator decisions, and repair exchanges. It does not expose private model thinking tokens; DeepSeek uses thinking internally and returns the final message content.

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

## Viewer

A static viewer is available at `viewer/index.html`. It renders generated JSON responses with LaTeX formatting.

Open it directly in a browser:

```bash
xdg-open viewer/index.html
```

Or start a local static file server:

```bash
python3 -m http.server 8080
```

Then open `http://127.0.0.1:8080/viewer/`.

Save an API response and load it in the viewer:

```bash
curl --max-time 600 -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"subject":"physics","level":"University of Tokyo","topic":"Mechanics"}' \
  -o generated-response.json
```

The viewer uses MathJax from a CDN, so LaTeX rendering needs browser access to that script. The raw text still displays if MathJax is unavailable.

## Test

```bash
python3 -m unittest discover -s tests
```
