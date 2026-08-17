# CareMind

CareMind is a LangGraph-powered multi-agent RAG assistant for medical and research documents. It supports PDF/text/image upload, first-class chat attachments, configurable Groq/OpenAI-compatible document vision, OCR fallback for scans/report photos, structured evidence retrieval, cited answers, report comparison, browser chat history, session memory, observability, evaluation, and Redis-backed response caching.

The production architecture uses FastAPI, LangGraph, Supabase Postgres with pgvector, object storage, Redis, background ingestion workers, and explicit evidence records. SQLite remains the local development store for document metadata, indexed chunk records, structured extraction records, and chat history.

CareMind is for medical education and document interpretation only. It is not a diagnosis, treatment plan, emergency triage system, or medication-dosing tool.

## 1. Architecture

Runtime request path:

```text
Browser / Next.js
  -> /speech/transcribe for voice input
       - browser records audio with MediaRecorder
       - backend calls ElevenLabs Scribe v2 batch STT
       - automatic language detection; no client language setting
       - returns verbatim transcript, confidence, word timestamps, detected_language, language_confidence
  -> FastAPI API
  -> input guardrails
       - emergency redirect
       - prompt-injection detection
       - low-confidence voice clarification
  -> LangGraph CareMindAgent orchestrator
  -> ConversationResolverAgent
       - resolves follow-up pronouns/entities
       - rewrites retrieval-ready standalone queries
       - rewrites the latest turn into a RECAP-style current intent
       - emits route hints, constraints, and confidence for planning
       - current-message attachments have highest priority
       - carries active patient/report/image/attachment state
  -> SupervisorAgent / intent router
  -> check_cache
       - exact cache over resolved query + workspace revision + active entities
       - semantic cache over resolved query, not raw follow-up text
  -> specialist agent
       - DirectResponseAgent: product/help answers
       - DocumentRAGAgent: uploaded-document retrieval
       - MedicalEducationAgent: trusted education corpus retrieval
       - ReportComparisonAgent: two-report comparison
       - ImagingAgent: image/report-grounded answers and future radiology model plug-in
       - ClarificationAgent: missing-context questions
  -> EvidenceManager / retrieval policy
  -> LLM generation
       - NVIDIA chat model when NVIDIA_API_KEY is set
       - optional medical LLM endpoint for MedicalEducationAgent
       - local grounded fallback when no key is set
  -> citation coverage check
  -> safety/disclaimer finalization
  -> cited response
```

Upload and indexing path:

```text
Upload PDF/text/image
  -> create ingestion job
  -> store raw asset (local private upload path in this build; Supabase Storage-ready metadata)
  -> create message_attachments row for the current conversation
  -> classify file/document modality
  -> extract text when available
  -> DocumentUnderstandingAgent once per uploaded document version
       - configured Groq/OpenAI-compatible vision model when CAREMIND_DOCUMENT_VLM_BASE_URL is set
       - outputs structured JSON, patient entities, labs, diagnoses, medications, measurements, type, summary, confidence
       - OCR is fallback/support for report photos and scans when the VLM/text parser cannot confidently parse
  -> store raw structured JSON in document_understanding / extraction_runs
  -> normalize into evidence_items
  -> chunk and embed structured evidence, not repeated VLM calls
  -> upsert embeddings and chunk metadata to Supabase pgvector
  -> store document metadata, evidence metadata, and chunk records
  -> update Document Memory
  -> update Conversation Memory when upload includes session_id
  -> chat binds attachment_ids to the exact user message when sent
```

Document and radiology vision paths stay independent:

```text
Document pipeline
  -> DocumentUnderstandingAgent
  -> configured Groq/OpenAI-compatible document vision model
  -> OCR/local fallback when unavailable or low-confidence
  -> structured JSON + evidence items
  -> DocumentRAGAgent

Radiology pipeline
  -> ImagingAgent
  -> MedGemma-style medical vision model only when configured
  -> linked report/evidence fallback or explicit limitation
  -> cited imaging answer
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
  -> compare against baseline
  -> mark PASS / WARN / FAIL
```

## 2. Core Components

- `FastAPI`: API, streaming chat, uploads, metrics, cache debug, and static frontend serving.
- `CareMindAgent`: LangGraph state machine for chat-time orchestration.
- `DocumentUnderstandingAgent`: upload-time document parser. It does not answer user questions.
- `EvidenceManager`: normalizes structured JSON, chunks, citations, and retrieval candidates into auditable evidence.
- `DocumentRAGAgent`: answers uploaded-document questions from stored evidence and citations.
- `MedicalEducationAgent`: answers general medical education questions from trusted education evidence, MedlinePlus health-topic enrichment, optional sanitized PubMed enrichment, and controlled model educational context when evidence is thin.
- `ImagingAgent`: answers image-scoped questions from linked report/caption/evidence today and owns future CXR/CT/MRI medical-vision integration.
- `ConversationResolverAgent`: resolves current-message attachments, active patient/report/image context, and follow-ups before routing and cache lookup.
- `SupervisorAgent`: state-aware route classifier with hard safety checks before cache or generation.
- `SafetyLayer`: emergency redirect, prompt-injection refusal, unsupported-answer handling, and disclaimer finalization.
- `ConversationMemory`: Redis-backed session/cache interface with SQLite/Postgres persistence.
- `SQLiteStore`: local development metadata store.
- `Supabase Postgres + pgvector`: production durable metadata, evidence, and vector backend.

## 3. Run Locally

```bash
uv sync
uv run python backend/run_server.py
```

Open `http://127.0.0.1:8000`.

For the Next.js frontend during development, run the API and frontend in separate terminals:

```bash
uv run python backend/run_server.py
npm run dev
```

Open `http://127.0.0.1:3000` or the port Next chooses if 3000 is busy. The Next app calls the FastAPI backend at `http://127.0.0.1:8002` during local dev, matching `backend/run_server.py`. Set `NEXT_PUBLIC_API_BASE_URL` if your API runs somewhere else.

For a static production export:

```bash
npm run build
```

The export is written to `frontend/out`, and FastAPI serves it from `/` when that directory exists.

Local Docker:

```bash
docker compose up --build
```

## 4. Environment Variables

Optional environment variables:

```env
# Text LLM and embeddings
NVIDIA_API_KEY=your_nvidia_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_CHAT_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
NVIDIA_EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
NVIDIA_REASONING_ENABLED=true
NVIDIA_REASONING_BUDGET=2048
NVIDIA_ENABLE_THINKING=
CAREMIND_EMBEDDING_PROVIDER=qwen
CAREMIND_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
CAREMIND_EMBEDDING_DIM=1024
CAREMIND_EMBEDDING_INDEX_VERSION=qwen3-0.6b-v1-1024
CAREMIND_EMBEDDING_BATCH_SIZE=4
CAREMIND_EMBEDDING_DEVICE=auto
CAREMIND_EMBEDDING_NORMALIZE=true
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBEDDING_MODEL=

# Medical education model
MEDICAL_LLM_BASE_URL=http://127.0.0.1:8001/v1
MEDICAL_LLM_MODEL=google/medgemma-4b-it
MEDICAL_LLM_API_KEY=
MEDICAL_LLM_PROVIDER=openai-compatible
MEDICAL_HF_MODEL=EpistemeAI/Reasoning-Medical0.1-27B

# Document understanding
CAREMIND_DOCUMENT_UNDERSTANDING_ENABLED=true
# Example for Groq: CAREMIND_DOCUMENT_VLM_BASE_URL=https://api.groq.com/openai/v1
CAREMIND_DOCUMENT_VLM_BASE_URL=
CAREMIND_DOCUMENT_VLM_MODEL=llama-4-scout-17b-16e-instruct
CAREMIND_DOCUMENT_VLM_API_KEY=
CAREMIND_DOCUMENT_VLM_MIN_CONFIDENCE=0.55
CAREMIND_DOCUMENT_VLM_TIMEOUT_SECONDS=60

# Future radiology vision
CAREMIND_RADIOLOGY_VISION_BASE_URL=
CAREMIND_RADIOLOGY_VISION_MODEL=google/medgemma-4b-it

# Speech-to-text
ELEVENLABS_API_KEY=
CAREMIND_TRANSCRIPT_MIN_CONFIDENCE=0.65

# Storage and vector search
CAREMIND_VECTOR_BACKEND=supabase
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_PUBLISHABLE_KEY=
SUPABASE_SECRET_KEY=
SUPABASE_JWKS_URL=https://YOUR_PROJECT_REF.supabase.co/auth/v1/.well-known/jwks.json
NEXT_PUBLIC_SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=
SUPABASE_DB_URL=
SUPABASE_VECTOR_TABLE=caremind_document_chunks_qwen3_1024
SUPABASE_MATCH_FUNCTION=match_caremind_document_chunks_qwen3_1024
PINECONE_API_KEY=
PINECONE_INDEX_NAME=caremind-index

# Redis and cache
REDIS_URL=redis://localhost:6379/0
CAREMIND_SESSION_TTL=3600
CAREMIND_RESPONSE_CACHE_ENABLED=true
CAREMIND_CACHE_TTL_SECONDS=86400
CAREMIND_SEMANTIC_CACHE_ENABLED=true
CAREMIND_SEMANTIC_CACHE_THRESHOLD=0.92

# Uploads and OCR fallback
CAREMIND_MAX_UPLOAD_BYTES=20971520
CAREMIND_ALLOWED_UPLOAD_CONTENT_TYPES=application/pdf,text/plain,text/markdown,image/png,image/jpeg,image/webp
CAREMIND_ATTACHMENT_STORAGE_BUCKET=local-uploads
CAREMIND_TEMP_ATTACHMENT_TTL_SECONDS=604800
CAREMIND_IMAGE_OCR_ENABLED=true
CAREMIND_IMAGE_OCR_ENGINE=auto
CAREMIND_IMAGE_OCR_MIN_CHARS=30
CAREMIND_IMAGE_OCR_STRUCTURING_ENABLED=true
NVIDIA_OCR_ENDPOINT=https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2
NVIDIA_OCR_MODEL=nvidia/nemotron-ocr-v2
NVIDIA_OCR_TIMEOUT_SECONDS=60

# External literature search
CAREMIND_EXTERNAL_SEARCH_ENABLED=true
NCBI_EUTILS_BASE_URL=https://eutils.ncbi.nlm.nih.gov/entrez/eutils
NCBI_TOOL=caremind
NCBI_EMAIL=
NCBI_API_KEY=
CAREMIND_EXTERNAL_SEARCH_MAX_RESULTS=4
CAREMIND_EXTERNAL_SEARCH_TIMEOUT_SECONDS=12
CAREMIND_MEDLINEPLUS_ENABLED=true
MEDLINEPLUS_BASE_URL=https://wsearch.nlm.nih.gov/ws/query
MEDLINEPLUS_TOOL=caremind
MEDLINEPLUS_EMAIL=
CAREMIND_MEDLINEPLUS_MAX_RESULTS=3
CAREMIND_MEDLINEPLUS_TIMEOUT_SECONDS=8
CAREMIND_MEDICAL_EDUCATION_MODEL_CONTEXT_ENABLED=true
CAREMIND_MEDICAL_EDUCATION_MODEL_CONTEXT_MIN_CHARS=900

# Safety, auth, and tracing
CAREMIND_GUARDRAILS_ENABLED=true
CAREMIND_TRANSCRIPT_MIN_CONFIDENCE=0.65
CAREMIND_USERNAME=demo
CAREMIND_PASSWORD=demo
CAREMIND_LOG_PATH=data/caremind-debug.log
CAREMIND_LOG_LEVEL=INFO
CAREMIND_CAPTURE_PRINTS=true
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=caremind-dev
LANGCHAIN_ENDPOINT=
```

Supabase pgvector is the default hosted vector backend. Run the Supabase migrations, including `supabase/migrations/20260816000000_create_qwen3_embedding_index.sql`, then set either `SUPABASE_URL` + `SUPABASE_SECRET_KEY` or `SUPABASE_DB_URL`. The secret key path uses Supabase REST/RPC for server-side vector writes; the DB URL path connects directly to Postgres.

CareMind defaults to local `Qwen/Qwen3-Embedding-0.6B` embeddings through `sentence-transformers`. Query text uses `encode_query`; uploaded document chunks, OCR-derived evidence, structured document-understanding evidence, and other corpus text use `encode_document`. Embeddings are normalized and tagged with `CAREMIND_EMBEDDING_INDEX_VERSION`. Do not compare or merge Qwen vectors with existing NVIDIA `nvidia/nv-embedqa-e5-v5` vectors, even though both are 1024-dimensional; keep them in separate tables/index versions and backfill before cutover.

If Supabase is missing, unreachable, or the table/function has not been created, upload, search, and document-RAG chat requests return a clear `503` instead of silently falling back to SQLite vector search in production mode.

To use local Ollama embeddings, set `CAREMIND_EMBEDDING_PROVIDER=ollama`, `OLLAMA_EMBEDDING_MODEL=nomic-embed-text:v1.5`, and `CAREMIND_EMBEDDING_DIM=768`. pgvector columns have fixed dimensions, so use a separate table, function, and `CAREMIND_EMBEDDING_INDEX_VERSION` for that model.

`MEDICAL_LLM_BASE_URL` is optional. Use it when you are serving a Hugging Face medical model through an OpenAI-compatible server such as vLLM. Keep it on a different port than CareMind, for example `8001`, because CareMind uses `8000`.

Set `MEDICAL_LLM_PROVIDER=transformers` to load `MEDICAL_HF_MODEL` directly with Transformers for medical-education reasoning. This is opt-in because `EpistemeAI/Reasoning-Medical0.1-27B` is large and should run on suitable local/hosted GPU capacity.

Set `LANGCHAIN_TRACING_V2=true` and provide `LANGCHAIN_API_KEY` to send LangGraph runs to LangSmith. Traces should include sanitized workflow metadata, tool status, retrieval counts, generation configuration, guardrail outcomes, and latency. Do not send raw reasoning content, patient identifiers, raw retrieved medical chunks, embedding vectors, credentials, local file paths, or sensitive prompt/document content to LangSmith.

## 5. Functional Requirements

- Users can upload medical PDFs, text files, clinical document images, scanned reports, prescriptions, discharge summaries, referral letters, lab reports, blood reports, and ECG reports.
- Users can upload image assets. Report photos and scans are parsed through document understanding; future CXR/CT/MRI radiology images are handled by the independent ImagingAgent radiology path.
- Uploaded files also become chat attachments scoped to the current `session_id`. The chat request sends `attachment_ids`, and the backend validates the IDs before binding them to the exact user message.
- Users can type or speak questions. Voice is transcribed before routing and uses the same safety, routing, retrieval, cache, and citation stack.
- Voice uses backend `POST /speech/transcribe` with ElevenLabs Scribe v2 batch STT. The frontend does not set a language; Scribe detects it automatically.
- Voice transcript metadata includes the original transcript text, confidence, word timestamps, modality, `detected_language`, and `language_confidence`. The original transcript is passed to routing/generation unchanged.
- Generated answers use a session language override when one is set in conversation state; otherwise voice answers default to the detected language.
- DocumentUnderstandingAgent runs once per uploaded document version, stores structured JSON, computes confidence, updates document memory, and generates structured evidence embeddings.
- Subsequent user questions use stored structured evidence and retrieval; they do not repeatedly invoke the document vision model.
- Document-grounded answers must cite uploaded evidence.
- General education answers use trusted education evidence, MedlinePlus health-topic enrichment, and optional clearly marked model educational context when retrieved evidence is thin.
- Explicit literature/similar-case requests use sanitized PubMed enrichment.
- Report comparison loads and compares indexed evidence from two selected documents.
- Current-message attachments are deterministic context and outrank active documents, active images, prior retrieval, and vector similarity. Contextual follow-ups reuse the active attachment evidence until the user clearly switches topic.
- Emergency symptoms, dosing decisions, and prompt-injection attempts are blocked before retrieval or generation where appropriate.
- Response cache keys include workspace revision, route, active patient/report/image/attachment state, current `attachment_ids`, active document IDs, vector backend, embedding model, generation model, retrieval settings, and resolved query.

## 6. API

- `POST /upload` uploads and indexes a PDF, text file, image, scan, or report photo. The response includes `attachment_id` and safe attachment metadata for chat use.
- `POST /speech/transcribe` accepts recorded audio as multipart form data, calls ElevenLabs Scribe v2 with automatic language detection, and returns `TranscriptMetadata`.
- `POST /chat` routes a question through retrieval, comparison, clarification, education, imaging, and safety checks. Include `attachment_ids` to bind uploaded files to the current message, for example `{"message":"What does this say?","attachment_ids":["attachment_..."]}`.
- `POST /chat/stream` streams route/status/delta/final events for chat.
- `POST /chat/inspect` returns a non-streaming inspected chat response for debugging.
- `POST /compare` compares two indexed reports.
- `GET /search` returns retrieved chunks.
- `GET /documents` lists workspace documents.
- `GET /images` lists uploaded image assets.
- `DELETE /documents/{document_id}?workspace_id=default` deletes a document and associated chunks/vectors where supported.
- `DELETE /images/{image_id}?workspace_id=default` deletes an image asset and its linked OCR/document-understanding document.
- `DELETE /chat/sessions/{session_id}?workspace_id=default` deletes a chat from durable message/context storage and clears its Redis session key.
- `GET /cache/debug?workspace_id=default&session_id=<id>` reports Redis status, key patterns, and cache counts without returning cached medical text.
- `DELETE /cache?workspace_id=default` clears cached chat responses for a workspace.
- `DELETE /cache/{workspace_id}` clears cached chat responses for a workspace using the path form.
- `POST /demo/seed` creates two synthetic reports for a quick demo.
- `GET /metrics` returns Prometheus exposition text with `caremind_*` metrics.
- `GET /metrics.json` returns the same in-memory metrics as JSON for debugging.
- `GET /health` returns application and vector-backend health.

## 7. Agent Routing

The backend uses a controlled supervisor-specialist pattern. `CareMindAgent` owns the LangGraph state machine, while specialist agents own narrow responsibilities:

- `DocumentUnderstandingAgent`: upload-time only. Parses uploaded clinical documents into structured JSON/evidence and never answers user questions.
- `ConversationResolverAgent`: resolves current-message attachments first, then pronouns, omitted patient/report/image references, active document IDs, follow-up query rewrites, and RECAP-style `intent_rewrite`.
- `SupervisorAgent`: classifies each request into a route, using the rewritten current intent when it is confident and falling back to rule routing otherwise.
- `DirectResponseAgent`: answers product/help questions such as `what do you do?`.
- `DocumentRAGAgent`: runs uploaded-document RAG with citations. For report-guidance questions such as "what should the patient do?", it first scans the active document chunks for explicit recommendations, follow-up plans, instructions, advice, conclusions, impressions, or treatment guidance. If none are found, it may add a separate cited "General next steps" section from trusted education evidence only.
- `MedicalEducationAgent`: runs general medical education RAG over trusted built-in corpus and optional external literature.
- `ImagingAgent`: runs image/report-grounded answers today and future MedGemma-style radiology interpretation behind a separate model interface.
- `ReportComparisonAgent`: compares two indexed reports.
- `ClarificationAgent`: asks for more context when the request is too vague or evidence targets are ambiguous.

Route taxonomy:

| Route | Purpose | Backing Source |
|---|---|---|
| `direct` | Product/help answers | Static CareMind capability text |
| `retrieve` | Uploaded document QA | Structured evidence, chunks, pgvector |
| `medical_education` | General medical education | Trusted education corpus + MedlinePlus + optional PubMed/model context |
| `imaging` | Uploaded image/report QA | Image metadata, linked reports, future radiology model |
| `compare` | Compare two reports | Indexed chunks/evidence from selected documents |
| `clarify` | Missing or ambiguous context | Conversation state |
| `emergency_redirect` | Emergency/dosing/safety redirect | Hard-coded no-generation path |

Routing keeps general medical education separate from uploaded-document QA. A standalone medical topic or concept question such as `What is pneumonia?`, `What foods help lower cholesterol?`, or `What does a CBC test check for?` routes to `medical_education` unless the user explicitly scopes it to `my report`, `the uploaded document`, `the patient`, an image, or a named report. Document-scoped questions such as `Does my report mention pneumonia?` stay on `retrieve`. Ambiguous follow-ups such as `What does that mean?` use active report/image context only after a prior cited document/image turn established that context. General education turns keep their own topic focus, so a follow-up like `What causes it?` after `What is pneumonia?` remains on `medical_education` and resolves against pneumonia rather than an uploaded report.

Chat graph:

```text
Input guardrail
  -> ConversationResolverAgent
  -> SupervisorAgent
  -> cache lookup
  -> route-specific specialist agent
  -> evidence/citation validation
  -> safety finalization
  -> memory/cache write
```

## 8. Multimodal Ingestion

Document understanding is the primary multimodal path for clinical documents:

- A configured Groq/OpenAI-compatible document vision endpoint is the primary target for PDFs, lab reports, blood reports, ECG reports, discharge summaries, prescriptions, referral letters, phone photos of reports, and scanned clinical documents.
- The model returns structured JSON, not free-form answers.
- OCR is fallback/support when the vision endpoint is unavailable, fails, or returns low confidence. Set `CAREMIND_IMAGE_OCR_ENGINE=nvidia` to force NVIDIA Nemotron OCR v2, or keep `auto` to use NVIDIA when `NVIDIA_API_KEY` is configured and then fall back to local OCR engines. If no readable text is found, the original image asset remains available and the upload does not fail solely because OCR/indexing failed.
- Every extraction has model name, extraction source, confidence, warnings, raw JSON, and index text.
- Structured evidence is embedded once and reused for later questions.

Future radiology support is isolated:

- CXR, CT, MRI, and other radiology image interpretation plug into `ImagingAgent`.
- Radiology does not share the document-ingestion model path.
- Ordinary reports never pay radiology-model latency or cost.
- Radiology output must still become evidence with provenance before final answering.

## 9. Evidence And Retrieval

CareMind is evidence-first. Final answers should be generated from selected evidence, not raw model memory.

Evidence records contain:

- `evidence_id`
- `workspace_id`
- `source_id`
- `source_type`
- `document_id` / `image_id`
- page, span, or coordinates where available
- normalized clinical concept
- value, unit, date, reference range, and flag where available
- confidence
- extraction method
- model name/version
- source quote or evidence text

Retrieval policy:

- Reuse previous retrieved chunks/evidence when the follow-up is scoped to the same active patient/report/image and prior evidence covers the question.
- Retrieve again when a new entity, document, lab, medication, time constraint, comparison target, or explicit evidence request appears.
- Use metadata filters, dense retrieval, local lexical/clinical reranking, parent-child neighbor expansion, and citation assembly.
- Return no-evidence responses when retrieved evidence is missing or below threshold instead of filling gaps from the model.

## 10. Memory

CareMind uses layered memory:

- Workspace memory: documents, images, summaries, structured JSON, evidence items, and chunks.
- Document memory: document type, summary, structured extraction confidence, extracted patient/entities, and recent cited chunks.
- Conversation memory: active patient, active report/document, active image, active document IDs, last route/tool, last answer preview, last retrieval, and conversation entities.
- Retrieval memory: last retrieved chunks, last citations, reuse confidence, and retrieval diagnostics.
- Cache memory: exact and semantic response cache in Redis.

Upload memory behavior:

- `POST /upload` with `session_id` marks the uploaded document or OCR-derived image document as active.
- Patient, report, lab-test, medication, and disease entities extracted during upload are merged into conversation entities.
- Upload without `session_id` still updates durable document memory; later chat turns can discover the document from workspace state.

Session state also tracks the active patient, report/document, image, last route/tool, last retrieved chunks, citations, answer preview, evidence need, and an entity memory map. `DocumentRAGAgent` reuses active retrieved chunks only when the new resolved query has the same evidence need and the reused chunks still pass the configured similarity threshold. If the evidence need changes, such as from key findings to report instructions, it performs a new targeted search.

Retrieval scores use one internal convention: higher is better. Chunks below `CAREMIND_MIN_RETRIEVAL_SIMILARITY` are filtered before generation; if every candidate is below threshold, the evidence set is empty. Returned retrieval diagnostics enforce `kept_count + filtered_count == returned_count` and track duplicate and below-threshold filters separately.

## 11. Storage

Production storage separates raw assets, extraction outputs, evidence, vectors, and conversation state:

```text
Raw files
  -> Supabase storage / S3-compatible object store

Structured metadata
  -> Supabase Postgres

Vector search
  -> Supabase pgvector caremind_document_chunks

Session/cache
  -> Redis

Local development
  -> SQLite metadata/chunks/messages + optional local vectors
```

Core tables/concepts:

- `documents`: uploaded files and OCR/document-understanding derived documents.
- `image_assets`: uploaded images and report photos.
- `document_understanding`: raw structured JSON, model, confidence, summary, and index text.
- `ingestion_jobs`: production job state, retries, partial failures, and processing status.
- `extraction_runs`: versioned parser/OCR/VLM runs.
- `evidence_items`: normalized claim-supporting evidence atoms.
- `citation_links`: answer/citation to evidence/source mapping.
- `chunks`: local development chunk metadata and embeddings.
- `caremind_document_chunks`: Supabase pgvector table.
- `messages`: chat messages.
- `conversation_context`: durable session state.
- `workspace_revisions`: cache invalidation namespace.

SQLite is a local development/runtime convenience. Supabase Postgres is the production source of truth. Derived vectors are rebuildable from documents, extraction runs, and evidence items.

## 12. Redis Cache

Redis is used when `REDIS_URL` or Redis host settings are reachable. It stores:

- conversation session state,
- exact response cache for retrieval, comparison, imaging, and medical education routes,
- semantic response cache for resolved queries,
- temporary agent state with `CAREMIND_SESSION_TTL`.

Response cache keys include a workspace revision counter, not a per-request hash over every document. If document rows, image rows, evidence rows, or chunk records change, the next chat request uses a fresh cache namespace.

Redis key shapes:

- `caremind:{workspace_id}:session:{session_id}` stores the recent chat-session message list.
- `caremind:{workspace_id}:cache:{cache_key}` stores exact cached responses.
- `caremind:{workspace_id}:semcache:{cache_namespace}:{cache_key}` stores semantic-cache embeddings plus the cached response.

If Redis is unavailable, CareMind falls back to SQLite/Postgres chat history and disables response cache.

## 13. Safety And Guardrails

- Emergency symptoms and medication-dosing requests route to `emergency_redirect`.
- Prompt-injection attempts in user input or retrieved/uploaded content are treated as untrusted data.
- Retrieved chunks, structured JSON, OCR text, and external literature are evidence, not instructions.
- The model must not diagnose, prescribe, or invent unsupported clinical explanations.
- Unsupported patient-specific facts must return "not provided in the retrieved evidence" style answers.
- External literature queries remove obvious patient identifiers before NCBI/PubMed requests.
- Raw document text, embeddings, secrets, and sensitive content are not written to normal operational logs.
- MIMIC/credentialed data mode requires controlled access, audit logging, and separate deployment from public synthetic demos.

## 14. Observability

Operational logs are written to `CAREMIND_LOG_PATH` and mirrored to the console. Logs include upload stages, ingestion job stages, extraction/chunking counts, embedding dimensions, vector-store calls, Supabase HTTP status/error bodies, request latency, and failure class. Raw document text, embeddings, and secrets are intentionally not logged.

`docker-compose.yml` includes Redis, Prometheus, and Grafana:

```bash
docker compose up --build
```

- API: `http://127.0.0.1:8000`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3005` (`admin` / `GRAFANA_ADMIN_PASSWORD`, default `admin`)

Prometheus scrapes `api:8000/metrics` every 15 seconds using `ops/prometheus/prometheus.yml`. Grafana provisions the Prometheus datasource and a `CareMind Overview` dashboard from `ops/grafana/`.

Grafana Cloud remote write is optional. Keep real cloud credentials in GitHub/server secrets, not in `ops/prometheus/prometheus.yml`; use `ops/prometheus/prometheus.cloud.yml.example` as a template if you enable remote write.

Key exported metrics include:

- `caremind_http_requests_total`
- `caremind_agent_routes_total`
- `caremind_tool_calls_total`
- `caremind_cache_events_total`
- `caremind_cache_lookup_total`
- `caremind_retrieval_reuse_total`
- `caremind_guardrail_events_total`
- `caremind_latency_ms`

Production observability should also track:

- ingestion job duration and status,
- OCR/VLM latency and failure rate,
- structured extraction confidence distribution,
- vector upsert/search latency,
- citation coverage rate,
- no-evidence/refusal rate,
- cache stale/bypass events,
- active entity changes,
- tenant/workspace error rate,
- database migration version.

## 15. Tracing

LangSmith tracing can be enabled with `LANGCHAIN_TRACING_V2=true`.

Traces include:

- resolved query,
- entity-resolution metadata,
- active patient/report/image state,
- router decision and route confidence,
- cache namespace/key and hit/miss reason,
- retrieval reuse decision,
- retrieval diagnostics,
- selected evidence and citation coverage,
- MCP/tool calls,
- sanitized generation configuration and final-answer metadata,
- guardrail outcomes,
- TTFT and total stream latency.

Traces should log decisions, evidence IDs, counts, hashes, and structured metadata. They should not log raw reasoning content, hidden chain-of-thought, patient identifiers, raw retrieved chunks, embedding vectors, credentials, local file paths, secrets, or sensitive prompt/document content.

## 16. Evaluation

Run:

```bash
uv run python backend/evaluate.py
```

The evaluator seeds demo reports and reports:

- route accuracy,
- citation pass rate,
- guardrail pass rate,
- retrieval relevance,
- average latency.

Each run writes `backend/eval_runs/<timestamp>.json` and `.md`, compares against `backend/eval_runs/baseline.json`, and marks the run `PASS`, `WARN`, or `FAIL`.

By default, `backend/evaluate.py` uses `CAREMIND_VECTOR_BACKEND=sqlite` so local eval can run without hosted vector network access. Set `CAREMIND_EVAL_USE_CONFIGURED_VECTOR=true` to evaluate against the configured Supabase or Pinecone backend.

### 16.1 RAGAS Evaluation

Run:

```bash
uv run python backend/evaluate_ragas.py
```

This evaluates retrieval first and generation second. It writes:

- `backend/eval_runs/ragas/<timestamp>.json`
- `backend/eval_runs/ragas/<timestamp>.ragas_dataset.jsonl`
- `backend/eval_runs/ragas/<timestamp>.md`

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

Full RAGAS mode attempts:

- `LLMContextRecall`
- `Faithfulness`
- `FactualCorrectness`

### 16.2 Production Eval Gates

Production release gates should include:

- route accuracy regression check,
- document-understanding extraction regression set,
- OCR fallback regression set,
- retrieval/citation coverage check,
- no-evidence behavior check,
- prompt-injection guardrail set,
- emergency/dosing guardrail set,
- cache invalidation and workspace revision check,
- PubMed/external-literature sanitization check.

## 17. Deployment

Local Docker:

```bash
docker compose up --build
```

CI/CD:

- `.github/workflows/ci.yml` runs Python compile checks, backend tests, frontend build, and Docker image build on push/PR.
- `.github/workflows/deploy.yml` builds and pushes `ghcr.io/<owner>/<repo>:latest`, then deploys over SSH when server secrets are configured.
- For server deploys, set `CAREMIND_API_IMAGE=ghcr.io/<owner>/<repo>:latest` or let the deploy workflow export it before `docker compose pull api`.
- Required deploy secrets: `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`. Optional: `DEPLOY_PORT`, `DEPLOY_PATH`, `GHCR_USERNAME`, `GHCR_TOKEN`.

Production deployment shape:

- FastAPI API container on Render, Fly.io, Azure Container Apps, Kubernetes, or a VM.
- Background ingestion worker for large PDFs, OCR, VLM extraction, embeddings, retries, and re-indexing.
- Redis managed service for session state and exact/semantic cache.
- Supabase Postgres with pgvector and `CAREMIND_VECTOR_BACKEND=supabase`.
- Supabase storage or S3-compatible object storage for raw uploads and derived artifacts.
- NVIDIA NIM endpoint via `NVIDIA_API_KEY`.
- Optional Groq/OpenAI-compatible document vision endpoint behind `CAREMIND_DOCUMENT_VLM_BASE_URL`.
- Optional MedGemma-style radiology endpoint behind `CAREMIND_RADIOLOGY_VISION_BASE_URL`.
- HTTPS and basic auth or Supabase Auth enabled.
- Separate synthetic demo, credentialed MIMIC, and production user-data environments.

Scaling milestones:

- 10 users: single API container, Redis, Supabase, synchronous small-file ingestion is acceptable.
- 100 users: add background ingestion workers, DB pooling, upload job states, worker concurrency limits, and VLM/OCR rate limits.
- 10k users / millions of documents: partition by tenant/workspace, autoscale workers, separate online API from offline indexing, add tenant quotas, data retention/deletion workflows, and rebuildable vector indexes.

## 18. Demo Flow

1. Start the backend.
2. Click `Seed demo reports` or upload synthetic medical PDFs/text/images.
3. Ask `What are the key findings?`
4. Ask `What changed between reports?`
5. Ask a follow-up such as `Why fatigue?` to verify active patient/report memory.
6. Ask a general education question such as `What are symptoms of tuberculosis?`
7. Upload a report photo or scan and confirm it is parsed through document understanding/OCR fallback.
8. Check the trace panel for route, resolved query, evidence, cache state, retrieval reuse, and TTFT.
9. Open Grafana and confirm request, cache, latency, and guardrail metrics.
10. Run `uv run python backend/evaluate.py` and confirm PASS/WARN/FAIL output.

CareMind is for medical education and document interpretation only. It is not a diagnosis or treatment plan.
