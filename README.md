# Entrance Exam Question Generation & Verification API

Python backend API for generating and validating Japanese university entrance exam questions with independent DeepSeek V4 Pro agents.

Supported subjects:

- `mathematics`
- `physics`
- `chemistry`

Physics generation is text-only by default. When `allow_diagram` is enabled, the generator may return a structured `diagram_spec`; the server renders that spec to SVG.

Chemistry generation is also text-only. The generator is instructed not to create diagrams, structural drawings, image references, or questions that require a visual diagram to solve.

## Requirements

- Python 3.11+
- `DEEPSEEK_API_KEY` in `.env` or the shell environment
- Optional: `DEEPSEEK_BASE_URL`, defaults to `https://api.deepseek.com`
- Optional: `DEEPSEEK_MODEL`, defaults to `deepseek-v4-pro`
- Optional: `DEEPSEEK_TIMEOUT_SECONDS`, defaults to `300`
- Optional: `GENERATION_TRACE`, defaults to `false`

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
```

Start the server:

```bash
python3 -m mathgen.api
```

The server listens on `http://127.0.0.1:8000`.

To see each Generator and Validator exchange in the server terminal, set:

```dotenv
GENERATION_TRACE=true
```

The trace shows the prompts sent to each agent, the JSON content returned by each agent, and both validator decisions. It does not expose private model thinking tokens; DeepSeek uses thinking internally and returns the final message content.

## Validation Flow

Each generation attempt is checked by two focused validators:

1. `CriticalValidator` checks genuine correctness issues: calculation errors, invalid equations, unit errors, logical gaps, contradictions, missing conditions, and problem/solution/diagram inconsistencies.
2. `ComplianceValidator` checks system acceptance: Japanese curriculum scope, Japanese output, subject/topic alignment, and diagram policy.

The API retries generation when either validator rejects the candidate. The external success and failure response shapes remain unchanged.

## Request

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"subject":"mathematics","level":"University of Tokyo","topic":"Probability Recurrence Relations"}'
```

Physics example:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"subject":"physics","level":"University of Tokyo","topic":"Mechanics"}'
```

Physics diagram example:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"subject":"physics","level":"University of Tokyo","topic":"Mechanics","allow_diagram":true}'
```

`allow_diagram` defaults to `false` and is currently supported only for physics. When enabled, the Generator may return a structured `diagram_spec`; the server renders it deterministically to SVG. The LLM is not allowed to output raw SVG directly.

Supported `diagram_spec.kind` values:

- `inclined_plane`
- `force_diagram`
- `circuit`

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
  "solution": "Japanese solution and explanation including $LaTeX$",
  "diagram": {
    "type": "svg",
    "description": "Optional diagram description",
    "spec": {"kind": "inclined_plane"},
    "svg": "<svg ...>...</svg>"
  }
}
```

The `diagram` field is omitted when no diagram is generated.

If the validator rejects all generation attempts, the API returns HTTP 422 with `status: "failed"` and the validator checklist result.

For backward compatibility, requests without `subject` are treated as `mathematics`.

## Test

```bash
python3 -m unittest discover -s tests
```
