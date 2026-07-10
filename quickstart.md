# CareMind Quickstart

## Prerequisites
- Python 3.11+
- uv
- Node.js 20+
- Redis
- Pinecone account if using cloud vector search
- NVIDIA API key

## Setup
1. Copy `.env.example` to `.env`.
2. Install backend dependencies with `uv sync`.
3. Start Redis if you want cache support.
4. Start the backend with `uv run python backend/run_server.py`.
5. Open the browser UI.
6. Load a sample document.
7. Ask a question.
8. Check `/metrics`.
9. Run `uv run python backend/evaluate.py`.

## Local Run
```bash
uv sync
uv run python backend/run_server.py
```

Open `http://127.0.0.1:8000`.

## Docker Run
```bash
docker compose up --build
```

## Expected Demo Flow
- Upload synthetic PDF.
- Ask a question.
- See cited answer.
- Ask the same question again.
- See cache hit.
- Run eval and inspect baseline diff.

## Troubleshooting
- If Redis fails, cache falls back to SQLite.
- If Pinecone is unavailable, use local vector fallback.
- If NVIDIA keys are missing, use local mock or hashing fallback for demos.
