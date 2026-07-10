# CareMind MVP

CareMind is a LangGraph-powered multi-agent RAG assistant for medical and research documents. It supports PDF/text upload, document search, cited answers, report comparison, session memory, and a shared backend for the browser UI and VS Code sidebar.

The app is built to use NVIDIA NIM endpoints and Pinecone when keys are configured. For local demos, it falls back to deterministic local embeddings, SQLite vector search, and SQLite-backed chat history.

## Architecture

```text
Browser / VS Code
  -> FastAPI API
  -> LangGraph CareMindAgent orchestrator
  -> SupervisorAgent
  -> check_cache
  -> specialist agent
       - DirectResponseAgent: product/help answers
       - DocumentRAGAgent: uploaded-document retrieval
       - MedicalEducationAgent: trusted education corpus retrieval
       - ReportComparisonAgent: two-report comparison
       - ClarificationAgent: missing-context questions
  -> LLM generation
       - NVIDIA chat model when NVIDIA_API_KEY is set
       - optional medical LLM endpoint for MedicalEducationAgent
       - local grounded fallback when no key is set
  -> safety/disclaimer finalization
  -> cited response
```

Retrieval path:

```text
Upload PDF/text
  -> extract text
  -> chunk text
  -> embed chunks
  -> upsert to vector store
       - Pinecone when CAREMIND_VECTOR_BACKEND=pinecone
       - SQLite/local cosine fallback otherwise
  -> user question embedding
  -> top-k chunk retrieval
  -> lexical/clinical rerank
  -> citations
```

Evaluation path:

```text
backend/ragas_questions.jsonl
  -> run CareMind retrieval
  -> run CareMind generation
  -> save RAGAS-shaped rows:
       user_input
       retrieved_contexts
       response
       reference
  -> run RAGAS metrics if ragas is installed
  -> always write fallback retrieval/generation diagnostics
```

## Run Locally

```bash
uv sync
uv run python backend/run_server.py
```

Open `http://127.0.0.1:8000`.

Optional environment variables:

```env
NVIDIA_API_KEY=your_nvidia_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_CHAT_MODEL=meta/llama-3.1-8b-instruct
NVIDIA_EMBEDDING_MODEL=nvolveqa_40k
MEDICAL_LLM_BASE_URL=http://127.0.0.1:8001/v1
MEDICAL_LLM_MODEL=google/medgemma-4b-it
MEDICAL_LLM_API_KEY=
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=caremind-index
CAREMIND_VECTOR_BACKEND=pinecone
REDIS_URL=redis://localhost:6379/0
CAREMIND_USERNAME=demo
CAREMIND_PASSWORD=demo
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=caremind-dev
LANGCHAIN_ENDPOINT=
```

Set `CAREMIND_VECTOR_BACKEND=local` for the keyless demo. Set it to `pinecone` after your Pinecone key is ready.

`MEDICAL_LLM_BASE_URL` is optional. Use it when you are serving a Hugging Face medical model through an OpenAI-compatible server such as vLLM. Keep it on a different port than CareMind, for example `8001`, because CareMind uses `8000`.

Set `LANGCHAIN_TRACING_V2=true` and provide `LANGCHAIN_API_KEY` to send LangGraph runs to LangSmith. Traces include graph inputs, outputs, tool results, and retrieved document snippets, so only enable this where the document/chat data is allowed to be sent to LangSmith.

## API

- `POST /upload` uploads and indexes a PDF or text file.
- `POST /chat` routes a question through retrieval, comparison, clarification, and safety checks.
- `POST /compare` compares two indexed reports.
- `GET /search` returns retrieved chunks.
- `GET /documents` lists workspace documents.
- `DELETE /cache?workspace_id=default` clears cached chat responses for a workspace.
- `POST /demo/seed` creates two synthetic reports for a quick demo.
- `GET /metrics` returns request, route, tool, cache, and latency metrics.

## Multi-Agent Routes

The backend uses a controlled supervisor-specialist pattern. `CareMindAgent` owns the LangGraph state machine, while specialist agents own narrow responsibilities:

- `SupervisorAgent`: classifies each request into a route.
- `DirectResponseAgent`: answers product/help questions such as `what do you do?`.
- `DocumentRAGAgent`: runs uploaded-document RAG with citations.
- `MedicalEducationAgent`: runs general medical education RAG over a small trusted built-in corpus.
- `ReportComparisonAgent`: compares two indexed reports.
- `ClarificationAgent`: asks for more context when the request is too vague.

The agents are orchestrated as a compiled LangGraph `StateGraph`:

```text
SupervisorAgent -> check_cache -> specialist agent -> safety/citation finalization
```

## Redis Cache

Redis is used when `REDIS_URL` or Redis host settings are reachable. It stores:

- conversation session state,
- exact response cache for retrieval, comparison, and medical education routes,
- temporary agent state with `CAREMIND_SESSION_TTL`.

Response cache keys include a revision hash of the workspace's indexed documents and chunks. If document rows or chunk text change in SQLite, the next chat request uses a fresh cache key and stores a new response. Uploads and demo seeding also clear old cached responses for that workspace.

If Redis is unavailable, CareMind falls back to SQLite chat history and disables response cache.

## Evaluation

Run:

```bash
uv run python backend/evaluate.py
```

The evaluator seeds demo reports and reports:

- route accuracy,
- citation pass rate,
- average latency.

### RAGAS Evaluation

Run:

```bash
uv run python backend/evaluate_ragas.py
```

This evaluates retrieval first and generation second. It writes:

- `backend/eval_reports/ragas/<timestamp>.json`
- `backend/eval_reports/ragas/<timestamp>.ragas_dataset.jsonl`
- `backend/eval_reports/ragas/<timestamp>.md`

The `.ragas_dataset.jsonl` file uses the standard RAGAS shape:

```json
{
  "user_input": "What are the key findings?",
  "retrieved_contexts": ["..."],
  "response": "...",
  "reference": "..."
}
```

If `ragas` is not installed, the script still runs local diagnostics:

- `retrieval_reference_term_recall`: whether retrieved chunks contain expected gold terms.
- `generation_reference_term_recall`: whether the final answer contains expected gold terms.
- `faithfulness_proxy_supported_sentence_rate`: rough answer support from retrieved context.
- `route_accuracy`
- `citation_pass_rate`

For full RAGAS metrics, configure an evaluator LLM and run:

```bash
uv run python backend/evaluate_ragas.py
```

Full RAGAS mode attempts:

- `LLMContextRecall`
- `Faithfulness`
- `FactualCorrectness`

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
