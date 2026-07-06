# CareMind Quickstart

## Prerequisites
- Python 3.11+
- Node.js 20+
- Redis
- Pinecone account if using cloud vector search
- NVIDIA API key

## Setup
1. Copy `.env.example` to `.env`.
2. Install backend dependencies.
3. Install frontend dependencies.
4. Start Redis.
5. Start the backend.
6. Start the web app.
7. Load a sample document.
8. Ask a question.
9. Check `/metrics`.
10. Run `python evaluate.py`.

## Local Run
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