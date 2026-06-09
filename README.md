# Prompt Strategy Benchmark

An end-to-end benchmark application for comparing prompt strategies against Gemini responses, objective metrics, persisted benchmark history, and exportable results.

## Tech Stack

- Python
- FastAPI
- Streamlit
- SQLite
- Pydantic
- python-dotenv
- pandas
- requests
- pytest
- google-genai

## Project Structure

```text
backend/      FastAPI app, Gemini integration, prompt strategies, benchmark engine, metrics, persistence, exports
frontend/     Streamlit application for running benchmarks and downloading exports
tests/        Automated tests for strategies, metrics, benchmark execution, persistence, history, retrieval, and exports
```

## Setup

1. Create a virtual environment.

```bash
python -m venv .venv
```

2. Activate the virtual environment.

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Create a local environment file.

```bash
copy .env.example .env
```

5. Add your Gemini API key to `.env`.

```text
GEMINI_API_KEY=YOUR_API_KEY_HERE
```

## Configuration

Supported environment variables:

```text
APP_NAME=Prompt Strategy Benchmark
APP_ENV=development
GEMINI_API_KEY=YOUR_API_KEY_HERE
GEMINI_MODEL=gemini-2.5-flash
GEMINI_API_VERSION=v1
GEMINI_TIMEOUT_SECONDS=30
DATABASE_URL=sqlite:///benchmark.db
```

The Streamlit frontend reads `PROMPT_BENCHMARK_API_URL` and defaults to `http://127.0.0.1:8000`.

## Run the Backend

```bash
python run_backend.py
```

The API is available at `http://127.0.0.1:8000`.

## Run the Frontend

```bash
streamlit run frontend/app.py
```

## API Endpoints

```text
GET  /
GET  /health
POST /generate-strategies
POST /benchmark
POST /test-gemini
GET  /benchmark/history
GET  /benchmark/{run_id}
GET  /benchmark/{run_id}/export/json
GET  /benchmark/{run_id}/export/csv
```

## Benchmark Workflow

1. Submit `user_input` to `POST /benchmark`.
2. The backend generates five prompt variants:
   - Zero-Shot
   - Role Prompting
   - Structured Prompting
   - Expert Prompting
   - Reasoning-Oriented Prompting
3. Each prompt is sent to Gemini sequentially.
4. The backend records response text, status, latency, response length, word count, and token usage when Gemini returns it.
5. The run and result rows are saved to SQLite.
6. Saved runs can be retrieved from history or exported as JSON/CSV.

## Persistence

SQLite persistence uses two tables:

```text
benchmark_runs
- run_id
- created_at
- user_input

benchmark_results
- id
- run_id
- strategy_name
- prompt
- response
- status
- latency_ms
- response_length
- word_count
- input_tokens
- output_tokens
- total_tokens
```

## Testing

Run all tests:

```bash
pytest
```

The test suite uses temporary SQLite databases and fake Gemini clients where needed, so it does not require a live Gemini API call.
