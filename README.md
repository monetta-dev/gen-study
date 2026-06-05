# Math Question Generation & Verification API

Python backend API for generating and validating Japanese university entrance exam mathematics questions with two independent DeepSeek V4 Pro agents.

## Requirements

- Python 3.11+
- `DEEPSEEK_API_KEY` in `.env` or the shell environment
- Optional: `DEEPSEEK_BASE_URL`, defaults to `https://api.deepseek.com`
- Optional: `DEEPSEEK_MODEL`, defaults to `deepseek-v4-pro`
- Optional: `DEEPSEEK_TIMEOUT_SECONDS`, defaults to `300`

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
```

Start the server:

```bash
python3 -m mathgen.api
```

The server listens on `http://127.0.0.1:8000`.

## Request

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"level":"University of Tokyo","topic":"Probability Recurrence Relations"}'
```

## Successful Response

```json
{
  "status": "success",
  "problem": "Question text including $LaTeX$",
  "solution": "Solution and explanation including $LaTeX$"
}
```

If the validator rejects all generation attempts, the API returns HTTP 422 with `status: "failed"` and the validator checklist result.

## Test

```bash
python3 -m unittest discover -s tests
```
