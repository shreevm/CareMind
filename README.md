# CareMind MVP

CareMind is a LangGraph-powered agentic RAG assistant for medical and research documents. It supports PDF/text upload, document search, cited answers, report comparison, session memory, and a shared backend for the browser UI and VS Code sidebar.

The app is built to use NVIDIA NIM endpoints and Pinecone when keys are configured. For local demos, it falls back to deterministic local embeddings, SQLite vector search, and SQLite-backed chat history.

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:8000`.

Optional environment variables:

```env
NVIDIA_API_KEY=your_nvidia_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_CHAT_MODEL=meta/llama-3.1-8b-instruct
NVIDIA_EMBEDDING_MODEL=nvolveqa_40k
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=caremind-index
CAREMIND_VECTOR_BACKEND=pinecone
REDIS_URL=redis://localhost:6379/0
CAREMIND_USERNAME=demo
CAREMIND_PASSWORD=demo
```

Set `CAREMIND_VECTOR_BACKEND=local` for the keyless demo. Set it to `pinecone` after your Pinecone key is ready.

## API

- `POST /upload` uploads and indexes a PDF or text file.
- `POST /chat` routes a question through retrieval, comparison, clarification, and safety checks.
- `POST /compare` compares two indexed reports.
- `GET /search` returns retrieved chunks.
- `GET /documents` lists workspace documents.
- `POST /demo/seed` creates two synthetic reports for a quick demo.
- `GET /metrics` returns request, route, tool, cache, and latency metrics.

## Agent Routes

The backend chooses among:

- `direct`: product/help questions such as `is this CareMind?`
- `retrieve`: uploaded-document RAG with citations.
- `compare`: report comparison tool.
- `medical_education`: general medical education RAG over a small trusted built-in corpus.
- `clarify`: asks for more context when the request is too vague.

The router is implemented as a compiled LangGraph `StateGraph`:

```text
route -> check_cache -> direct | clarify | compare | medical_education | retrieve -> finalize
```

## Redis Cache

Redis is used when `REDIS_URL` or Redis host settings are reachable. It stores:

- conversation session state,
- exact response cache for retrieval, comparison, and medical education routes,
- temporary agent state with `CAREMIND_SESSION_TTL`.

If Redis is unavailable, CareMind falls back to SQLite chat history and disables response cache.

## Evaluation

Run:

```bash
python evaluate.py
```

The evaluator seeds demo reports and reports:

- route accuracy,
- citation pass rate,
- average latency.

## Deployment

Local Docker:

```bash
docker compose up --build
```

Cloud deployment shape:

- FastAPI container on Render, Fly.io, Azure Container Apps, or a VM.
- Redis managed service or the included compose Redis for demos.
- Pinecone hosted index with `CAREMIND_VECTOR_BACKEND=pinecone`.
- NVIDIA NIM endpoint via `NVIDIA_API_KEY`.
- HTTPS and basic auth enabled with `CAREMIND_USERNAME` and `CAREMIND_PASSWORD`.

## VS Code Sidebar

The extension scaffold lives in `vscode-extension`.

```bash
cd vscode-extension
npm install
npm run compile
```

Run the extension from VS Code and set `caremind.apiBaseUrl` if your backend is not on `http://127.0.0.1:8000`.

## Demo Flow

1. Start the backend.
2. Click `Seed demo reports` or upload synthetic medical PDFs/text.
3. Ask `What are the key findings?`
4. Ask `What changed between reports?`
5. Open the VS Code sidebar and continue the same session against the shared backend.

CareMind is for medical education and document interpretation only. It is not a diagnosis or treatment plan.
