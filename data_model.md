# CareMind Data Model

This document describes the core entities and relationships for CareMindâ€™s backend data model (Supabase Postgres + pgvector + Redis). It supports:

- Users and workspaces.
- Documents, images, chunks, messages, and message attachments.
- Voice transcripts with Scribe v2 detected-language metadata.
- Citations and caching.
- Evaluation runs and LangSmith traces.
- Routing, variants, and MIRAGE/PubMedQA evaluation.

---

## 1. User & Identity

### User / Profile

In Supabase, authentication is handled by `auth.users`. CareMind stores additional profile data in a `profiles` table.

**Profile**

- `id` (uuid, PK)
- `auth_user_id` (uuid, unique, FK to `auth.users.id`)
- `email` (text, nullable)
- `created_at` (timestamptz)
- `last_seen_at` (timestamptz)
- `default_workspace_id` (uuid, FK to `workspaces.id`, nullable)
- `feature_flags` (jsonb, default `{}`)

Behavior:

- On first authenticated request:
  - Upsert `profiles` row keyed by `auth_user_id` and set `created_at`/`last_seen_at`.
- On subsequent requests:
  - Update `last_seen_at` and optionally `feature_flags`.

---

## 2. Workspaces & Sessions

### Workspace

A workspace groups documents, sessions, and chat history, scoped to a user (or multiple users in the future).

- `id` (uuid, PK)
- `owner_id` (uuid, FK to `profiles.id`)
- `name` (text)
- `data_mode` (text enum: `synthetic` | `mimic`)
- `created_at` (timestamptz)

Notes:

- Each user gets a default workspace on first use.
- `data_mode` separates synthetic vs MIMICâ€‘backed workspaces.

### Session (Conversation)

Represents a conversational session within a workspace.

- `id` (uuid, PK)
- `workspace_id` (uuid, FK to `workspaces.id`)
- `user_id` (uuid, FK to `profiles.id`)
- `route_default` (text, nullable; optional default route hint)
- `eval_config_id` (text, nullable; config id for eval runs, if any)
- `variant` (text, nullable; e.g., `A` or `B` for A/B testing)
- `started_at` (timestamptz)
- `last_active_at` (timestamptz)

### ConversationContext / Session State

CareMind keeps a durable per-session context payload in SQLite/Postgres, with Redis as the fast session cache when available.

- `session_id` (text / uuid)
- `workspace_id` (text / uuid)
- `active_patient` / `current_patient` (text)
- `active_document_id` (text / uuid)
- `active_document_name` / `current_document` (text)
- `active_report` / `current_report` (text)
- `active_image_id` / `current_image` (text / uuid, nullable)
- `active_attachment_id` / `active_attachment_type` / `active_attachment_name` (text, nullable)
- `active_document_ids` (jsonb/text array)
- `last_attachment_ids` (jsonb/text array)
- `last_structured_evidence_id` (text / uuid, nullable)
- `current_topic` / `conversation_focus` (text, nullable)
- `last_route` (text)
- `last_tool` (text)
- `last_retrieved_chunks` (jsonb)
- `last_citations` (jsonb)
- `last_summary` / `last_answer_preview` (text)
- `conversation_entities` (jsonb)
- `document_memory` (jsonb)
- `updated_at` (timestamptz)

`conversation_entities` groups detected entities by kind: `patient`, `report`, `image`, `disease`, `medication`, `doctor`, `hospital`, `lab_test`, and `timeline_event`. This state is used by `ConversationResolverAgent` before routing and by cache namespacing.

---

## 3. Documents, Images, Chunks

### Document

Represents an uploaded file (PDF, text, etc.), an OCR-derived image document, or a logical document.

- `document_id` / `id` (uuid/text, PK)
- `workspace_id` (uuid, FK to `workspaces.id`)
- `filename` (text)
- `content_type` / `file_type` (text; e.g., `application/pdf`, `text/plain`, `application/x-ocr-json`)
- `source_type` (text; e.g., `pdf_deid`, `synthetic_au`, `mimic_iv_note`, `medcorp`)
- `file_path` / `storage_path` (text; local path, Supabase storage path, or S3 key)
- `source_image_id` (uuid/text, nullable; links OCR-derived documents back to `image_assets.image_id`)
- `summary` (text; cached upload/OCR summary)
- `document_type` (text, nullable; e.g., `lab_report`, `prescription`, `ecg_report`, `discharge_summary`)
- `understanding_confidence` (float, nullable; overall parse confidence from DocumentUnderstandingAgent)
- `created_at` (timestamptz)
- `metadata` (jsonb, optional; page counts, OCR flags, etc.)

### ImageAsset

Represents an uploaded clinical image (e.g., chest Xâ€‘ray).

- `image_id` / `id` (uuid/text, PK)
- `workspace_id` (uuid, FK to `workspaces.id`)
- `ocr_document_id` (uuid/text, FK to `documents.id`, nullable; link to OCR-derived text document)
- `filename` (text)
- `content_type` (text; e.g., `image/png`, `image/jpeg`, `image/webp`)
- `modality` (text; e.g., `CXR`, `CT`, `MRI`, `US`, `photo`)
- `file_path` / `storage_path` (text; local path, Supabase storage path, or S3 key)
- `report_text_summary` (text; summary of paired report/caption/OCR if available)
- `created_at` (timestamptz)
- `metadata` (jsonb; e.g., DICOM tags, if available, or dataset info)

Document understanding and OCR flow:

- The raw image stays in `image_assets`.
- DocumentUnderstandingAgent runs once per uploaded document version and writes structured JSON before retrieval.
- A configured Groq/OpenAI-compatible document vision endpoint is the primary document-understanding target when `CAREMIND_DOCUMENT_VLM_BASE_URL` is configured.
- OCR is fallback/support for screenshots, scans, and phone photos when the vision endpoint is unavailable, fails, or returns low confidence.
- The structured output is saved as a linked `documents` row using `source_image_id` when the source is an image, then chunked and embedded as structured evidence.
- Deleting the image also deletes its linked OCR document, local chunks, and vector rows.

### DocumentUnderstanding

Stores the raw structured JSON produced at upload time. This is document memory, not chat memory.

- `document_id` (uuid/text, PK, FK to `documents.document_id`)
- `workspace_id` (uuid/text)
- `filename` (text)
- `content_type` (text)
- `document_type` (text)
- `summary` (text)
- `confidence` (float)
- `model_name` (text; configured document vision model, OCR post-processing, or local fallback)
- `extraction_source` (text; `document_vlm`, `ocr_postprocess`, or `local_fallback`)
- `raw_json` (json/text; patient, dates, diagnoses, medications, lab values, measurements, sections, warnings)
- `index_text` (text; structured evidence representation used for embeddings)
- `created_at` / `updated_at` (timestamptz)

Behavior:

- The agent does not answer user questions.
- It executes once per uploaded document version.
- Later questions use `document_understanding.index_text`, chunk rows, and vector search instead of repeatedly invoking the vision model.
- If upload includes a `session_id`, the backend creates a `message_attachments` row for that conversation and marks the uploaded document or OCR-derived image document as active in conversation context.

### Chunk

Represents a text chunk associated with a document, with an embedding stored in pgvector.

- `id` (uuid, PK)
- `workspace_id` (uuid, FK to `workspaces.id`)
- `document_id` (uuid, FK to `documents.id`)
- `chunk_index` (integer; sequence number within document)
- `text` (text; chunk content)
- `embedding` (vector; pgvector column)
- `metadata` (jsonb; e.g., page number, section, source dataset)
- `created_at` (timestamptz)

Notes:

- The `embedding` column lives in the Supabase `caremind_document_chunks` table.
- Local SQLite also keeps `chunks(chunk_id, document_id, workspace_id, document_name, page, position, text, embedding_json, metadata_json)` as the metadata/source-of-truth side of the index.
- Supabase stores `caremind_document_chunks(chunk_id, workspace_id, document_id, document_name, text, metadata jsonb, embedding vector)`.
- Parent-child retrieval uses child chunk rows for vector search, then expands selected hits with neighboring chunk rows from the same document when local metadata is available.
- Structured document chunks use metadata such as `{"source": "document_understanding", "document_type": "...", "understanding_model": "...", "understanding_confidence": 0.82}`.
- OCR-derived image chunks preserve `{"source": "ocr", "image_id": "...", "ocr_engine": "..."}` and add document-understanding metadata alongside it.
- CareMind uses one shared vector table; each document is separated by `workspace_id` and `document_id`, not by a different index.
- Indexes:
  - pgvector index on `embedding`.
  - B-tree index on `(workspace_id, document_id)` for metadata filters.

### DocumentMemory

Document-level memory is separate from chat history and conversation state.

- `document_id` (uuid/text)
- `document_name` (text)
- `summary` (text; cached at upload from extracted text)
- `last_chunks` (jsonb; recent chunk traces used for answers)
- `last_citations` (jsonb)
- `last_answer_preview` (text)
- `updated_at` (timestamptz)

---

## 4. Messages & Citations

### Message

Stores chat messages (user and assistant) within a session.

- `id` (uuid, PK)
- `session_id` (uuid, FK to `sessions.id`)
- `workspace_id` (uuid, FK to `workspaces.id`)
- `user_id` (uuid, FK to `profiles.id`, nullable for system messages)
- `role` (text; `user` | `assistant` | `system`)
- `content` (text; message text)
- `modality` (text; `text` | `voice` | `image` | `system`)
- `route` (text; route used for the assistant message, e.g., `clinical_document_qa`)
- `tool_calls` (jsonb; MCP/tools metadata, nullable)
- `citations` (jsonb; denormalized citation info, optional)
- `created_at` (timestamptz)

### MessageAttachment

Represents the conversational relationship between a user message and one uploaded file. It does not duplicate the underlying document/image metadata; it points to `documents` or `image_assets`.

- `id` (text/uuid, PK; returned to the frontend as `attachment_id`)
- `message_id` (integer/uuid, nullable until the chat message is sent; FK to `messages.id`)
- `conversation_id` / `session_id` (text/uuid)
- `workspace_id` (text/uuid)
- `user_id` (text/uuid, nullable in local/basic-auth mode)
- `attachment_type` (text; `document` or `image`)
- `filename` (text)
- `mime_type` (text)
- `file_size` (integer)
- `storage_bucket` (text; local default is `local-uploads`, Supabase default can be `caremind-attachments`)
- `storage_path` (private local path or object-storage key; never raw base64)
- `image_asset_id` (text/uuid, nullable)
- `document_id` (text/uuid, nullable; for PDFs/text docs or OCR/document-understanding evidence derived from an image)
- `processing_status` (text; e.g., `completed`, `failed`)
- `processing_error` (text, nullable)
- `created_at` (timestamptz)
- `expires_at` (timestamptz, nullable)
- `persistence_mode` (text; `temporary`, `saved`, or `saved_compat`)

Behavior:

- `POST /upload` creates the content asset and a conversation-scoped attachment row.
- `POST /chat` accepts `attachment_ids`, validates they belong to the same workspace and conversation, then binds them to the inserted user message.
- Current-message attachments are deterministic context. `ConversationResolverAgent` considers them before named entities, active documents/images, previous focus, or vector search.
- `DocumentRAGAgent` and `ReportComparisonAgent` use exact attachment-linked document chunks before semantic search.
- Follow-up turns such as "Explain it simply" reuse the active attachment's stored structured evidence instead of calling the vision model again.
- Temporary attachments can expire and delete private binaries when no saved document/image is referenced. Saved documents and images remain until explicitly deleted, even if a chat referencing them is removed.

### Citation

Normalized citation records for assistant answers. Each citation points from an answer to a specific source.

- `id` (uuid, PK)
- `answer_message_id` (uuid, FK to `messages.id`)
- `source_type` (text; `document`, `image`, `education_corpus`, `external_pubmed`)
- `source_id` (uuid or text; ID of `documents.id`, `image_assets.id`, or external source identifier)
- `chunk_id` (uuid, FK to `chunks.id`, nullable; when citation points to a specific chunk)
- `quote` (text; excerpt used in the answer)
- `offset_start` (integer; character offset within chunk text, nullable)
- `offset_end` (integer; character offset within chunk text, nullable)
- `metadata` (jsonb; e.g., page number, URL, PubMed ID)
- `created_at` (timestamptz)

Notes:

- Denormalized `citations` on `messages` can store the same info for fast lookup; `Citation` table supports detailed auditing and eval.
- Chat deletion removes browser-local history, `messages` rows for the `session_id`/workspace, `message_attachments` relationship rows for that conversation, durable conversation context, and the Redis session key. It does not delete saved uploaded documents, images, vectors, or workspace-level response cache entries.

---

## 5. Cache & Semantic Responses

Redis is the primary cache for sessions and responses; we also persist some cache metadata in Postgres (optional). The `CacheEntry` entity below is conceptual; it may live in Redis only or be mirrored to Postgres.

### CacheEntry (Conceptual / Optional Postgres Table)

Represents a cached answer for a query under a specific configuration.

- `id` (uuid, PK)
- `workspace_id` (uuid, FK to `workspaces.id`)
- `doc_set_hash` (text; hash of the document set/workspace revision)
- `route` (text; route this cache entry applies to)
- `modality` (text; `text` | `image` | `voice`)
- `vector_backend` (text; e.g., `supabase`)
- `embedding_model` (text)
- `generation_model` (text)
- `top_k` (integer)
- `min_retrieval_similarity` (double precision)
- `query_text` (text)
- `resolved_query_text` (text)
- `active_document_ids` (jsonb/text array)
- `entity_fingerprint` (text)
- `query_embedding` (vector; optional, for semantic cache if persisted)
- `answer` (text)
- `citations` (jsonb)
- `tool_calls` (jsonb, nullable)
- `cache_type` (text; `exact` | `semantic`)
- `ttl_expires_at` (timestamptz)
- `created_at` (timestamptz)

Notes:

- In practice, the primary storage for cached answers is Redis; this table can be used for debugging, analytics, or offline inspection.
- Exact and semantic cache entries are scoped to workspace revision, current `attachment_ids`, active attachment IDs, and active entity/document context so stale or cross-patient follow-up answers are not reused.
- Redis key shapes are:
  - `caremind:{workspace_id}:session:{session_id}` for recent session messages.
  - `caremind:{workspace_id}:cache:{cache_key}` for exact cached responses.
  - `caremind:{workspace_id}:semcache:{cache_namespace}:{cache_key}` for semantic-cache embeddings plus cached responses.
- `GET /cache/debug` reports Redis availability, key patterns, and counts without returning cached answer text.

---

## 6. Evaluation & Traces

### EvalRun

Stores the results of an evaluation run (internal test set, MIRAGE, PubMedQA, etc.).

- `id` (uuid, PK)
- `created_at` (timestamptz)
- `dataset_name` (text; e.g., `internal_doc_qa`, `MIRAGE`, `PubMedQA`)
- `eval_config_id` (text; identifies the model/retriever config)
- `git_sha` (text; code commit hash, optional)
- `metrics_json` (jsonb; aggregate metrics: accuracy, citation_pass_rate, etc.)
- `status` (text; `PASS` | `WARN` | `FAIL`)
- `baseline_flag` (boolean; true if this run is the current baseline)
- `notes` (text, nullable; freeâ€‘form description)

### EvalExample (Optional, if storing perâ€‘item results)

If perâ€‘question/perâ€‘example evals are persisted:

- `id` (uuid, PK)
- `eval_run_id` (uuid, FK to `eval_runs.id`)
- `dataset_name` (text)
- `example_id` (text; datasetâ€‘specific identifier)
- `route` (text)
- `input` (jsonb; question, context, etc.)
- `output` (jsonb; model answer, citations)
- `label` (jsonb; ground truth)
- `metrics_json` (jsonb; perâ€‘example metrics, e.g., correct/incorrect)
- `langsmith_trace_url` (text, nullable)
- `created_at` (timestamptz)

### TraceRef

Minimal join between eval runs and LangSmith traces for quick navigation.

- `id` (uuid, PK)
- `eval_run_id` (uuid, FK to `eval_runs.id`)
- `route` (text)
- `variant` (text, nullable; eval config / A/B variant)
- `langsmith_trace_url` (text)
- `created_at` (timestamptz)

---

## 7. External Literature & Tools

### ExternalSearchQuery

Optional table capturing sanitized external search queries (e.g., PubMed) for auditing and debugging.

- `id` (uuid, PK)
- `workspace_id` (uuid, FK to `workspaces.id`)
- `session_id` (uuid, FK to `sessions.id`, nullable)
- `user_id` (uuid, FK to `profiles.id`)
- `original_query` (text; redacted/sanitized)
- `sanitized_query` (text)
- `tool_name` (text; e.g., `pubmed_literature_search`)
- `result_count` (integer)
- `latency_ms` (integer)
- `created_at` (timestamptz)

---

## 8. Voice Metadata

### VoiceTranscript

Stores browser voice input transcribed by ElevenLabs Scribe v2 batch STT. The transcript text is preserved exactly as returned by STT; language metadata is stored alongside it and is not used to overwrite, translate, or normalize the transcript.

- `id` (uuid, PK)
- `message_id` (uuid/integer, FK to `messages.id`, nullable until attached)
- `session_id` (uuid/text)
- `workspace_id` (uuid, FK to `workspaces.id`)
- `transcript` (text; original Scribe transcript text)
- `confidence` (double precision)
- `language` (text; compatibility field, usually same as `detected_language`)
- `detected_language` (text; Scribe language code)
- `language_confidence` (double precision; Scribe language probability)
- `duration_seconds` (double precision, nullable)
- `audio_storage_path` (text, nullable; original audio file, if stored)
- `timestamps` (jsonb; word/segment timings and speaker/audio-event fields returned by Scribe)
- `modality` / `input_modality` (text; `voice`)
- `started_at` / `ended_at` (timestamptz, nullable; browser recording bounds)
- `created_at` (timestamptz)

Behavior:

- The frontend records audio and calls `POST /speech/transcribe`; it does not set a language manually.
- `/chat` receives the verbatim transcript as `message` and the full `transcript` metadata object.
- The router, generation path, traces, and persistence receive the same transcript metadata unchanged.
- Answer generation uses `conversation_context.response_language_override`/`selected_response_language` when set; otherwise voice answers default to `detected_language`.

---

## 9. A/B Testing & Routing Metadata

### VariantAssignment

Variant assignment may be deterministic and computed at runtime (hashing `user_id`/`workspace_id`), but a table can be used if persistent control is needed.

- `id` (uuid, PK)
- `user_id` (uuid, FK to `profiles.id`)
- `workspace_id` (uuid, FK to `workspaces.id`)
- `route` (text; e.g., `medical_knowledge_qa`)
- `variant` (text; e.g., `A`, `B`)
- `created_at` (timestamptz)

Notes:

- Alternatively, variant can be stored only in `sessions.variant` and inferred.

---

## 10. Relationships (Summary)

- **User / Profile**
  - `profiles.auth_user_id` â†’ Supabase `auth.users.id`.
  - One profile â†’ many workspaces, sessions, messages.

- **Workspace**
  - One workspace â†’ many documents, image assets, sessions, chunks, messages.
  - `workspaces.owner_id` â†’ `profiles.id`.

- **Document / Image / Chunk**
  - `documents.workspace_id` â†’ `workspaces.id`.
  - `image_assets.workspace_id` â†’ `workspaces.id`.
  - `image_assets.ocr_document_id` â†’ `documents.id` (optional OCR-derived document).
  - `chunks.workspace_id` â†’ `workspaces.id`.
  - `chunks.document_id` â†’ `documents.id`.

- **Sessions & Messages**
  - `sessions.workspace_id` â†’ `workspaces.id`.
  - `sessions.user_id` â†’ `profiles.id`.
  - `messages.session_id` â†’ `sessions.id`.
  - `messages.workspace_id` â†’ `workspaces.id`.
  - `messages.user_id` â†’ `profiles.id` (optional for system messages).
  - `message_attachments.message_id` â†’ `messages.id`.
  - `message_attachments.document_id` â†’ `documents.id` when indexed document evidence exists.
  - `message_attachments.image_asset_id` â†’ `image_assets.id` when the attachment is an image.

- **Citations**
  - `citations.answer_message_id` â†’ `messages.id`.
  - `citations.chunk_id` â†’ `chunks.id` (optional).
  - `citations.source_id` â†’ `documents.id` / `image_assets.id` / external ID, depending on `source_type`.

- **Eval & Traces**
  - `eval_runs` stand alone.
  - `trace_refs.eval_run_id` â†’ `eval_runs.id`.
  - Optional `eval_examples.eval_run_id` â†’ `eval_runs.id`.

This model is intended to be implemented in Supabase Postgres with pgvector for embeddings plus Redis for session/caching, matching the CareMind specs and plan.
