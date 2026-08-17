# CareMind Quickstart

## Prerequisites

- Python 3.11+
- `uv`
- Node.js 20+
- Redis
- Supabase project with:
  - Postgres
  - pgvector extension enabled
  - storage bucket for documents/images
- NVIDIA API key (for text LLM + embeddings)
- (Optional) `ELEVENLABS_API_KEY` for Scribe v2 batch speech-to-text voice input
- (Optional) document vision endpoint (for example Groq/OpenAI-compatible) and optional MedGemma-style radiology endpoint

---

## Setup

1. **Environment**

   - Copy `.env.example` to `.env`.
   - Set the following environment variables:

     - `SUPABASE_URL`
     - `SUPABASE_PUBLISHABLE_KEY`
     - `SUPABASE_SECRET_KEY`
     - `SUPABASE_DB_URL` (optional direct Postgres connection)
     - `SUPABASE_VECTOR_TABLE=caremind_document_chunks_qwen3_1024`
     - `SUPABASE_MATCH_FUNCTION=match_caremind_document_chunks_qwen3_1024`

     - `NVIDIA_API_KEY`
     - `NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1`
     - `NVIDIA_CHAT_MODEL`
     - `CAREMIND_EMBEDDING_PROVIDER=qwen`
     - `CAREMIND_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B`
     - `CAREMIND_EMBEDDING_INDEX_VERSION=qwen3-0.6b-v1-1024`

     - `REDIS_URL` (e.g., `redis://localhost:6379/0`)

     - `CAREMIND_GUARDRAILS_ENABLED=true`
     - `CAREMIND_RESPONSE_CACHE_ENABLED=true`
     - `CAREMIND_SEMANTIC_CACHE_ENABLED=true`

     - (Optional) `CAREMIND_EXTERNAL_SEARCH_ENABLED` and NCBI E-utilities vars for PubMed.
     - (Optional) `CAREMIND_MEDLINEPLUS_ENABLED` and MedlinePlus vars for richer health-topic education.
     - (Optional) `CAREMIND_DOCUMENT_VLM_BASE_URL`, `CAREMIND_DOCUMENT_VLM_MODEL`, and `CAREMIND_DOCUMENT_VLM_API_KEY` for upload-time document vision.
     - (Optional) `ELEVENLABS_API_KEY` for browser voice input through backend `POST /speech/transcribe`.
     - `CAREMIND_ATTACHMENT_STORAGE_BUCKET` and `CAREMIND_TEMP_ATTACHMENT_TTL_SECONDS` if changing attachment storage/lifecycle defaults.

2. **Install backend dependencies**

   ```bash
   uv sync
   ```

3. **Run Supabase migrations**

   - Run the migrations that define:
     - `profiles`, `workspaces`, `documents`, `caremind_document_chunks`,
     - `sessions`, `messages`, `citations`,
     - and any other tables from `data_model.md`.

   Example:

   ```bash
   # Example placeholder â€“ use your actual migration files
   supabase db push
   # or, if you keep raw SQL:
   psql "$SUPABASE_DB_URL" -f supabase/migrations/20260713000000_create_caremind_vectors.sql
   ```

4. **Start Redis**

   ```bash
   redis-server
   ```

5. **Start the backend**

   ```bash
   uv run python backend/run_server.py
   ```

6. **Start the frontend (web app)**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

7. **Open the browser UI**

   - Visit the frontend URL (e.g., `http://localhost:3000`).
   - Authenticate (Supabase Auth) if enabled.

8. **Load a sample document**

   - Upload a synthetic PDF or deâ€‘identified report into a workspace.
   - Wait for ingestion (extract â†’ chunk â†’ embed â†’ pgvector).

9. **Ask a question**

   - Example: â€œWhat are the key findings?â€

10. **Check metrics**

    - Backend should expose Prometheus metrics at `/metrics`.
    - Optional JSON view (`/metrics.json`) if implemented.

11. **Run evaluation**

    ```bash
    uv run python backend/evaluate.py
    ```

    - This runs internal tests and MIRAGE/PubMedQA (once wired), writes `backend/eval_runs/<timestamp>.json`, and updates `baseline.json` on PASS.

---

## Local Run

Minimal local run (no frontend, hitting the FastAPI docs/Swagger):

```bash
uv sync
uv run python backend/run_server.py
```

Open `http://127.0.0.1:8000` for API docs and basic health checks.

---

## Docker Run

Assuming a `docker-compose.yml` that includes backend, frontend, Redis, and Postgres/Supabase proxy:

```bash
docker compose up --build
```

Then open the frontend URL (e.g., `http://localhost:3000`).

---

## Expected Demo Flow

1. Upload a synthetic PDF into the default workspace.
2. Ask a document question:
   - See a cited answer from `clinical_document_qa`.
3. Ask the same question again:
   - See a cache hit (exact or semantic) reflected in logs/metrics.
4. Ask a general medical education question (e.g., â€œWhat is hypertension?â€):
   - Route to `medical_knowledge_qa`, retrieve from MedCorp, and return citations.
5. Ask a nursing question (once implemented):
   - Route to `nursing_care_qa`.
6. Run eval:
   - `uv run python backend/evaluate.py` and inspect the baseline diff / PASSâ€‘WARNâ€‘FAIL.
7. Try a dosing/emergencyâ€‘style question:
   - Confirm `emergency_redirect` fires (no LLM call, refusal + guidance to seek care).
8. Optional:
   - Use voice input to ask a question. The browser records audio, the backend calls ElevenLabs Scribe v2 with automatic language detection, and `/chat` receives the verbatim transcript plus `detected_language`, `language_confidence`, timestamps, confidence, and modality.
   - Attach an image/report in the chat composer and ask "What does this say?" Confirm the request sends `attachment_ids` and the answer uses that exact attachment.
   - Upload a radiology image with linked report evidence and call `imaging_qa`; without a configured radiology model, CareMind should not claim independent pixel interpretation.

---

## Troubleshooting

- **Redis unavailable**
  - Symptom: cache misses only; potential errors on cache access.
  - Behavior: CareMind falls back to Postgres/SQLite for chat history and disables response caching. Ensure `REDIS_URL` is correct and Redis is running.
  - Inspect: `GET /cache/debug?workspace_id=default&session_id=<session_id>` shows `redis_enabled`, session/exact/semantic cache counts, and the key patterns CareMind uses. Exact responses live under `caremind:{workspace_id}:cache:*`, semantic responses under `caremind:{workspace_id}:semcache:*`, and recent session messages under `caremind:{workspace_id}:session:{session_id}`.

- **Supabase / pgvector unavailable**
  - Symptom: document upload or search fails.
  - Behavior: document upload/search returns a clear serviceâ€‘unavailable error instead of partial indexing.
  - Check:
    - Supabase project status,
    - that pgvector extension is enabled,
    - and that `SUPABASE_VECTOR_TABLE` matches the actual table.

- **NVIDIA keys missing or invalid**
  - Symptom: LLM/embedding calls fail.
  - Behavior: backend should return a clear configuration error.
  - For local development, you can implement a mock/path that returns deterministic dummy embeddings or canned responses to exercise the pipeline without real model calls.

- **Voice transcription unavailable**
  - Symptom: mic recording works, but transcription fails.
  - Behavior: `/speech/transcribe` returns a configuration/service error and the chat message is not sent.
  - Check:
    - `ELEVENLABS_API_KEY` is set on the backend.
    - `elevenlabs` is installed via `uv sync`.
    - The browser granted microphone permission.
    - The uploaded audio blob is non-empty.

- **PDF text extraction fails**
  - Symptom: no text or garbled text on retrieval.
  - Behavior: upload should be rejected with an error message; do not index error messages or binary data.
  - Check:
    - PDF parser logs,
    - file type and encoding.

- **Eval harness issues**
  - Symptom: `evaluate.py` crashes or produces empty metrics.
  - Behavior: script should fail loudly with clear error messages.
  - Check:
    - path to eval datasets (internal, MIRAGE, PubMedQA),
    - environment variables for eval configs,
    - baseline file path (`CAREMIND_EVAL_BASELINE_PATH`).

- **Metrics or tracing missing**
  - Ensure:
    - Prometheus endpoint `/metrics` is enabled.
    - LangSmith API key and project are configured for tracing.
    - Grafana is pointed to the correct Prometheus data source.

