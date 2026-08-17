# CareMind Plan

## 1. Purpose

This document defines the technical implementation plan for CareMind. It translates the product requirements in `specs.md` into an executable architecture, build sequence, and system design.

CareMind is implemented as a webâ€‘only AI medical chatbot, backed by Supabase (Postgres + pgvector + Auth), Redis, NVIDIAâ€‘hosted models, and a MIRAGE/PubMedQA evaluation loop.[web:19][web:24]

---

## 2. Technical Strategy

CareMind will be built as a modular agentic RAG platform with:

- FastAPI backend.
- LangGraph workflow orchestration.
- NVIDIAâ€‘hosted text inference and embeddings.
- Supabase pgvector for vector search (Pinecone only later if scale demands it).[web:19][web:24]
- Supabase Auth, storage, and Postgres for users, workspaces, documents, chunks, and chat history.[web:19][web:22]
- Redis session and cache state.
- LangSmith tracing.
- Prometheus metrics and Grafana dashboards.
- Web frontend (Next.js) as the only user interface.
- STTâ€‘first voice input.
- Optional vision model for imaging mode.
- Offline MIRAGE and PubMedQA evaluation harness for medical_knowledge_qa.[web:3][web:17]

The system will favor retrievalâ€‘grounded generation over fineâ€‘tuning the generation model for factual correctness. For early stages and <1â€“2M chunks, Supabase pgvector is preferred over an external vector DB; Pinecone becomes an option only if vector count or latency requirements exceed what Postgres can comfortably handle.[web:19][web:24]

---

## 3. Architecture Overview

### 3.1 Core Request Path

User input enters the web frontend, is normalized, routed through a FastAPI backend, checked by input guardrails, passed to the router, optionally served from cache, then sent through retrieval, tool use, and generation, and finally screened again by output guardrails before response.

The implemented request path now inserts conversation grounding before routing:

```text
User input
  -> input guardrails
  -> ConversationResolverAgent
       -> resolve pronouns, omitted entities, and active patient/report/image state
       -> rewrite follow-ups into standalone resolved queries
  -> SupervisorAgent route decision using resolved query + context
  -> response cache keyed by workspace revision + active entities + resolved query
  -> retrieval reuse check
  -> specialist agent
  -> generation
  -> safety finalization
```

### 3.2 Core Services

- **API service**: chat, upload, compare, metrics, eval, health.
- **Agent service**: `CareMindAgent` LangGraph orchestrator plus specialist agents under `backend/caremind/agents/`.
- **Multiâ€‘agent orchestration layer**: supervisor agent plus specialist agents for:
  - document RAG,
  - medical and nursing education,
  - report comparison,
  - imaging,
  - safety,
  - memory,
  - citation validation.
- **Conversation grounding service**: `ConversationResolverAgent` resolves follow-up references, active entities, and standalone retrieval queries before the supervisor routes.
- **Retrieval service**: document chunk search via Supabase pgvector; imageâ€‘report matching over MIMICâ€‘CXR/ROCOv2 where available.
- **MCP tool server**: document search, comparison, timeline, education tools (imaging tools later).
- **Cache/session layer**: Redis with SQLite/Postgres fallback for history if Redis unavailable.
- **Eval service**: seeded tests, MIRAGE/PubMedQA harness, baseline diffing, and report generation.[web:3][web:17]
- **Observability layer**: LangSmith + Prometheus + Grafana.

---

## 4. Data Flow

### 4.1 Text Workflow

1. User uploads PDF/text via web app.
2. Backend extracts text, normalizes, chunks, embeds via NVIDIA, and writes chunks + embeddings to Supabase pgvector.
3. User asks a question:
   - ConversationResolverAgent resolves patient/report/image references and rewrites follow-ups.
   - Router selects clinical_document_qa or medical_knowledge_qa using the resolved query and active state.
   - Exact/semantic cache is checked against the resolved query, workspace revision, and active entity context.
   - If previous retrieved chunks cover the follow-up, they are reused.
   - Otherwise retrieval queries Supabase pgvector, filtered by workspace and document metadata.
   - Child vector hits are expanded with adjacent parent context from the source document.
   - LLM generates a grounded answer using retrieved context, with citations.
4. Answer, citations, and metadata are stored in Postgres (messages table) and optionally cached in Redis.

### 4.2 Imaging Workflow

1. User uploads image (e.g., chest Xâ€‘ray).
2. Image file stored in Supabase storage; metadata (modality, linked report ID) stored in Postgres.
3. For imaging_qa:
   - Retrieve nearest reports/captions from MIMICâ€‘CXR or ROCOv2 indexes.
   - Run visionâ€‘language model (MedGemma or fallback) with question + retrieved text.
   - Generate grounded description with citations to matched reports/captions.

### 4.3 Voice Workflow

1. Web app records audio with `MediaRecorder`.
2. Backend `POST /speech/transcribe` sends the audio to ElevenLabs Scribe v2 batch STT.
3. Scribe v2 uses automatic language detection; the client/backend do not set a manual language.
4. The original transcript text is preserved exactly and sent to `/chat` as standard text input with `input_modality=voice`.
5. Transcript metadata includes confidence, timestamps, modality, `detected_language`, and `language_confidence`.
6. Router, retrieval, generation, guardrails, and citations behave identically to typed input.
7. Generation answers use a session language override when present, otherwise the detected voice language.
8. Optional TTS service converts final answer to audio for playback.

---

## 5. Routing Design

The router classifies requests into:

- `clinical_document_qa`
- `medical_knowledge_qa`
- `imaging_qa`
- `report_comparison`
- `nursing_care_qa`
- `emergency_redirect`
- `clarify`

The router acts as the **Supervisor Agent**, selecting the correct specialist agent based on:

- user intent,
- presence of uploaded documents/images,
- resolved query,
- active patient/report/image state,
- safety constraints (e.g., emergency/dosage),
- explicit hints (â€œin generalâ€, â€œnot about this patientâ€, â€œPubMedâ€, â€œsimilar caseâ€),
- cache state (but routing happens before cache lookup).

The router should be cheap, fast, and independently measurable. A small classifier or LoRAâ€‘tuned router model is preferred over using the main generation LLM for routing decisions.

---

## 5.1 Multiâ€‘Agent Design

CareMind uses a supervisorâ€‘specialist pattern:

- **Supervisor Agent**: classifies intent and chooses the workflow/route.
- **Conversation Resolver Agent**: resolves pronouns, omitted entities, active patient/report/image references, active document IDs, and follow-up query rewrites before routing.
- **Document RAG Agent**: answers from uploaded documents with citations.
- **Medical Education Agent**: answers from MedCorp (PubMed + StatPearls + textbooks + Wikipedia), aligned with MIRAGE/MedRAG.[web:3]
- **Nursing Education Agent**: answers from a nursingâ€‘focused subset of MedCorp.
- **Report Comparison Agent**: compares two uploaded reports and returns cited differences.
- **Imaging Agent**: handles imaging_qa (image â†’ similar reports â†’ vision model â†’ cited description).
- **Safety Agent**: checks input and output for unsafe medical behavior and enforces emergency_redirect.
- **Memory Agent**: manages Redis session memory and cache interactions with Postgres conversation history.
- **Citation Agent**: verifies answers are supported by retrieved evidence and enforces citation policy.
- **Direct Response Agent**: handles product/help questions without retrieval.
- **Clarification Agent**: asks for missing context when a request is underspecified.

Agents are orchestrated with LangGraph. Each agent owns a narrow responsibility and logs trace steps for observability.

Planned implementation files:

- `backend/caremind/agent.py`: LangGraph orchestration, cache checks, final response assembly.
- `backend/caremind/agents/conversation_resolver.py`: conversation grounding, entity memory, and resolved query generation.
- `backend/caremind/agents/supervisor.py`: route selection.
- `backend/caremind/agents/document_rag.py`: uploadedâ€‘document RAG.
- `backend/caremind/agents/medical_education.py`: MedCorpâ€‘backed RAG + external literature enrichment.
- `backend/caremind/agents/nursing_education.py`: nursingâ€‘scope QA.
- `backend/caremind/agents/comparison.py`: report comparison.
- `backend/caremind/agents/imaging.py`: imaging QA.
- `backend/caremind/agents/direct.py`: product/help responses.
- `backend/caremind/agents/clarification.py`: missingâ€‘context responses.
- `backend/caremind/agents/safety.py`: safety guardrails.

---

## 6. Retrieval Design

### 6.1 Text Retrieval

- Use NVIDIA embeddings and Supabase pgvector for document and knowledge retrieval.[web:19][web:24]
- Supabase Postgres stores:
  - chunk text,
  - metadata (document ID, workspace ID, source dataset),
  - embedding vector (pgvector).
- A single vector table (`caremind_document_chunks`) with workspace/document metadata and pgvector index is the primary vector store.
- SQLite is used only for lightweight dev metadata if needed, not as a vector fallback.
- Use parent-child retrieval for long PDFs: vector search selects child chunks, then generation receives neighboring parent context from the same document where available.
- Reuse active retrieved chunks for follow-up questions when lexical/entity confidence is high enough; otherwise perform a fresh vector search.

Design considerations:

- For MVP (<1M chunks), Supabase pgvector is sufficient and simpler than managing Pinecone.[web:19][web:24]
- If future vector scale or latency demands justify, Pinecone can be added as an external vector backend while Supabase remains the source of truth.

### 6.2 Imaging Retrieval

- Store image metadata (modality, file path, associated report ID) in Postgres.
- For imaging_qa:
  - Use either:
    - precomputed image embeddings (if a vision embedding index is available), or
    - nearest neighbor on associated report text chunks via pgvector.
  - Retrieve topâ€‘k relevant reports/captions.
- Pass question + retrieved report text + image to the vision model to generate a grounded answer, with citations.

### 6.3 Citation Policy

Every factual answer must include citations to:

- retrieved document passages,
- imageâ€‘associated source reports/captions,
- or trusted education corpus (MedCorp) passages.

Answers that cannot be supported with citations must either:

- return a clarification request, or
- be clearly labeled as uncertain with a conservative, educationalâ€‘only response.

---

## 7. Cache Design

Use Redis for:

- Session state (recent conversation context, active patient/report context).
- Entity memory for patients, reports, images, diseases, medications, doctors, hospitals, lab tests, and timeline events.
- Document-level memory for cached summaries, last chunks, citations, and answer preview per document.
- Exactâ€‘match response cache (question â†’ answer + citations).
- Semantic response cache (query embedding similarity above threshold).
- Temporary agent state (intermediate results during long workflows).

Cache behavior:

- Cache keys scoped by:
  - `workspace_id`, `workspace_revision`, `route`, `modality`, `top_k`, `vector_backend`, `embedding_model`, `generation_model`, `min_retrieval_similarity`.
- Conversation-aware cache keys additionally include active document IDs, active patient/report/image context, and an entity fingerprint.
- Exact cache keys additionally include normalized query string.
- Semantic cache uses cosine similarity with threshold (default 0.92).
- Exact and semantic cache lookups use the resolved query rather than raw follow-up text.
- Debug/bypass flags disable cache reads/writes.
- If Redis is unavailable:
  - fallback to Postgres/SQLite for chat history only,
  - disable semantic and exact response caching.

---

## 8. Safety Design

### 8.1 Input Guardrails

- Screen uploads and queries for:
  - prompt injection and systemâ€‘prompt stealing,
  - scope creep (offâ€‘topic, nonâ€‘medical or inappropriate uses),
  - emergency/dosage requests.
- If emergencyâ€‘like or dosageâ€‘related, trigger `emergency_redirect` before routing.

### 8.2 Output Guardrails

- Screen generated responses for:
  - diagnosticâ€‘sounding claims,
  - treatment/dosage recommendations,
  - prompt/system content leakage.
- Add a clear nonâ€‘diagnostic, educationalâ€‘only disclaimer for medically relevant outputs.
- If guardrails flag high risk or low confidence, either:
  - answer at a high level with strong disclaimers, or
  - refuse and recommend seeking professional care.

### 8.3 Emergency Behavior

- Emergencyâ€‘like queries must bypass generation and go directly to emergency_redirect:
  - Refuse to answer,
  - Instruct user to seek immediate medical attention,
  - Avoid storing unnecessary details about potential emergencies.

---

## 9. Observability Design

### 9.1 Tracing

Use LangSmith to trace:

- router decisions and chosen route,
- resolved query and entity resolution metadata,
- retrieval queries and topâ€‘k results,
- retrieval reuse decisions and confidence,
- MCP tool calls and outputs,
- sanitized generation configuration and selected model,
- guardrail checks and outcomes,
- cache hits/misses,
- perâ€‘node and total latency.

Do not send raw reasoning content, private chain-of-thought, patient identifiers, raw retrieved medical chunks, embedding vectors, credentials, local file paths, or sensitive prompt/document content to LangSmith.

Traces are tagged with:

- `workspace_id`,
- `route`,
- `cache_hit` type (none/exact/semantic),
- `eval_config_id` (for offline eval runs),
- `variant` (for online A/B tests).

### 9.2 Metrics

Expose Prometheus metrics for:

- request counts by route and status,
- route distribution,
- tool call counts (including `pubmed_literature_search`),
- cache hits/misses by type,
- cache lookup outcomes and reasons,
- retrieval reuse hit/miss counts,
- error counts by stage (router, retrieval, LLM, guardrail),
- latency by route and variant,
- evaluation scores (MIRAGE, PubMedQA, guardrail pass rate).[web:18]

### 9.3 Dashboards

Grafana dashboards should include:

- p50/p95 latency by route and variant,
- cache hit/miss rate over time,
- retrieval reuse hit/miss rate over time,
- route mix over time,
- eval metrics (MIRAGE accuracy, PubMedQA accuracy, guardrail pass rate) with pass/warn/fail bands.[web:18]
- external tool usage frequency (PubMed, MCP tools).

---

## 10. Evaluation Design

The evaluation loop should support:

- route accuracy,
- citation pass rate,
- groundedness / faithfulness,
- retrieval precision@k / recall@k,
- guardrail pass rate,
- average latency,
- MIRAGE benchmarking for the medical_knowledge_qa route (zeroâ€‘shot, questionâ€‘only, multiâ€‘choice).[web:3]
- PubMedQA accuracy for PubMedâ€‘style yes/no/maybe questions.[web:17]

Evaluation modes:

- **Offline, labeled test sets**:
  - Internal labeled cases for clinical_document_qa and nursing_care_qa.
  - MIRAGE multiâ€‘choice questions for medical_knowledge_qa.
  - PubMedQA for PubMedâ€‘style QA.
- **Configâ€‘aware**:
  - Support multiple `eval_config_id`s for different LLM/embedding/retrieval combinations.
  - For each config, run full eval and store metrics.
- **A/B comparison**:
  - Compare candidate config metrics against baseline and mark PASS/WARN/FAIL.

Every eval run is written to disk (JSON) and compared against the last passing baseline. If metrics regress beyond thresholds, mark run FAIL and surface in dashboards/CI.

---

## 11. Frontend Design

### 11.1 Web App

Build a simple, responsive chat UI with:

- Authentication (Supabase Auth).
- Workspace selector (in future).
- File upload components for PDFs/text/images.
- Voice input button:
  - pressâ€‘toâ€‘record,
  - visualization of STT transcript before sending.
- Streaming responses from backend.
- Citation panel:
  - clickable citations that open source passages in a side panel.
- Route indicator and disclaimers:
  - show active route (document vs education vs imaging),
  - show nonâ€‘diagnostic disclaimer prominently.
- Debug view (for dev):
  - show route, cache status, token counts, and trace link.

### 11.2 Browser Session Controls

All user interaction is via the web app. The browser UI supports session switching, local chat history, hover-to-delete chat cleanup, and trace/cache inspection against the backend APIs.

---

## 12. Implementation Phases

### Phase 1: Core RAG Backend

- FastAPI scaffold.
- Supabase integration (Auth, Postgres, storage, pgvector).
- Document upload â†’ text extraction â†’ chunking â†’ embeddings â†’ pgvector indexing.
- Basic clinical_document_qa RAG with citations.

### Phase 2: Router, Guardrails, Redis Cache

- Implement SupervisorAgent routing across all routes.
- Implement safety and injection guardrails.
- Integrate Redis for session + exact/semantic cache.
- Implement report comparison agent and MCP comparison tool.

### Phase 3: Evaluation & Observability

- Implement MIRAGE and PubMedQA eval harness for medical_knowledge_qa.[web:3][web:17]
- Implement baseline diffing and PASS/WARN/FAIL logic.
- Add LangSmith tracing.
- Expose Prometheus metrics and basic Grafana dashboards.

### Phase 4: Web UI

- Build chat UI with uploads, citations, route indicator, and disclaimers.
- Add debug mode for demos (show route, trace link, cache status).

### Phase 5: Voice Input & TTS

- Integrate ElevenLabs Scribe v2 batch STT for voice questions.
- Treat transcripts as standard text input with unchanged metadata.
- Store `detected_language` and `language_confidence` with FR-25 voice fields.
- Optional TTS for reading answers aloud.

### Phase 6: Imaging Mode

- Implement imaging_qa pipeline with MedGemma/Qwen2â€‘VL.
- Index MIMICâ€‘CXR/ROCOv2 text and integrate with imaging workflow.

### Phase 7: A/B Testing & Router Tuning

- Add multiple eval configs and offline A/B evaluation on MIRAGE/PubMedQA.
- Add online variant routing for medical_knowledge_qa (A/B models) with shared guardrails.
- LoRAâ€‘tune router and track route accuracy improvements.

---

## 13. Key Tradeoffs

- Prefer RAG over generation fineâ€‘tuning for factual grounding.
- Use Supabase pgvector initially for simplicity and close integration with relational data; revisit Pinecone only if vector scale or latency becomes a bottleneck.[web:19][web:24]
- Keep the router separate from the generator for cost, latency, and better eval.
- Keep voice as STTâ€‘first rather than native speechâ€‘toâ€‘speech for the MVP.
- Treat imaging as an extension once the text workflow and eval loop are stable.
- Keep synthetic demo data completely separate from credentialed MIMICâ€‘backed mode.

---

## 14. Nonâ€‘Goals

- Full PHI compliance certification.
- EHR/FHIR integration.
- DICOM support.
- Fully autonomous openâ€‘ended agents without routing constraints.
- Emergency triage automation.
- Paid GPU inference for text in the MVP.
- Fineâ€‘tuning the main generator as the default strategy.

---

## 15. Build Principles

- Make every important behavior inspectable (traces, logs, debug views).
- Make every important metric measurable (Prometheus, MIRAGE/PubMedQA scoring).[web:3][web:17][web:18]
- Make routing cheap, explicit, and testable (separate router model and dataset).
- Make safety explicit and enforced in code, not just prompt text.
- Make demos work on synthetic and open corpora, with MIMICâ€‘backed mode as a gated, credentialed option.
