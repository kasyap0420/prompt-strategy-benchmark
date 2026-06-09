# Prompt Strategy Benchmark

A starter application for benchmarking prompt strategies with a Python backend, FastAPI API foundation, Streamlit frontend, SQLite setup, and a clean path toward Gemini integration.

## Tech Stack

- Python
- FastAPI
- Streamlit
- SQLite
- Pydantic
- python-dotenv
- pytest

## Project Structure

```text
backend/      FastAPI app, configuration, database setup, Gemini client wrapper, strategy and benchmark foundations
frontend/     Streamlit application foundation
tests/        Test package placeholder
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

5. Add local values to `.env` when ready.

## Run the Backend

```bash
python run_backend.py
```

The API will be available at `http://127.0.0.1:8000`.

## Run the Frontend

```bash
streamlit run frontend/app.py
```

## Current Phase

This repository is Phase 1 scaffolding only. Live Gemini calls, benchmark orchestration, scoring, persistence schema, and deployment are intentionally left for later phases.
