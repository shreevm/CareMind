# CareMind Tasks

This checklist tracks implementation against `specs.md` and `plan.md` for the web‑only CareMind medical chatbot (Supabase + pgvector, NVIDIA, Redis, MIRAGE/PubMedQA eval).

---

## Phase 1: Foundation

- [x] Create repo structure.
- [x] Add environment variable handling.
- [x] Set up FastAPI app.
- [x] Add health endpoint.
- [x] Add logging and config modules (structured logging, pydantic settings).
- [ ] Add Supabase client initialization (Auth, Postgres, storage, pgvector).
- [ ] Add basic database migrations (profiles, workspaces, documents, chunks, conversations).

---

## Phase 2: Documents (Text Ingestion & Retrieval)

- [x] Implement PDF/text upload (web app → FastAPI).
- [x] Extract text from uploaded files.
- [x] Chunk documents.
- [x] Generate embeddings via NVIDIA.
- [x] Upsert vectors to Supabase pgvector (replace Pinecone upserts).
- [x] Implement Supabase pgvector retrieval (filter by workspace, document).
- [x] Add local fallback retrieval (SQLite/dev).
- [ ] Wire document ingestion to workspace + document metadata in Postgres.

---

## Phase 3: Agent & Routing

- [x] Implement query router (SupervisorAgent).
- [x] Add clinical_document_qa route.
- [x] Add medical_knowledge_qa route (MedCorp backend).
- [x] Add report_comparison route.
- [x] Add clarification route.
- [x] Add emergency_redirect route.
- [x] Add citation formatting.
- [x] Add nursing_care_qa route (nursing‑scope MedCorp subset).
- [x] Add imaging_qa route skeleton (delegating to imaging agent).
- [x] Integrate external literature tool (`pubmed_literature_search`) under medical_knowledge_qa.

---

## Phase 4: Memory and Cache

- [x] Add Redis session storage.
- [x] Add exact‑match response cache.
- [x] Add semantic response cache (cosine similarity threshold).
- [x] Add SQLite/Postgres fallback for chat history if Redis unavailable.
- [x] Make cache namespaces include route, workspace_revision, modality, top_k, vector_backend, embedding_model, generation_model, and min_retrieval_similarity.
- [x] Add explicit debug/bypass_cache handling in the agent graph.

---

## Phase 5: Safety & Guardrails

- [x] Add prompt injection screening (input).
- [x] Add medical safety screening (input/output).
- [x] Add disclaimer injection for medically relevant outputs.
- [x] Add output leak checks (system prompt / internal config leaks).
- [x] Add explicit emergency_redirect behavior (no LLM call, hardcoded refusal).
- [x] Log guardrail decisions for eval (pass/fail per query).

---

## Phase 6: Evaluation & A/B Harness

- [x] Create internal eval set (document QA, education QA, nursing QA).
- [ ] Create adversarial guardrail set (injection, jailbreak, emergency, dosing).
- [x] Implement `evaluate.py`.
- [x] Save eval runs to JSON (`backend/eval_runs/<timestamp>.json`).
- [x] Compare against baseline (`baseline.json`).
- [x] Emit PASS/WARN/FAIL results per metric.
- [ ] Add MIRAGE integration for medical_knowledge_qa (multi‑choice, question‑only retrieval).
- [ ] Add PubMedQA integration for PubMed‑style questions.
- [ ] Support multiple eval configs (`eval_config_id` with different LLM/embedding/retriever combos).
- [ ] Add A/B diff reporting between configs (offline).

---

## Phase 7: Web UI

- [x] Build web chat UI.
- [x] Add document upload UI.
- [x] Add citation display panel.
- [x] Add report comparison UI flow.
- [x] Add metrics page (basic Prometheus visualizations or links).
- [x] Add route indicator (document vs education vs imaging vs nursing).
- [x] Add explicit non‑diagnostic disclaimer banner.
- [x] Add debug/dev view (route, cache status, trace link).

---

## Phase 8: Browser Session Controls

> Browser-only milestone: chat sessions are managed from the web UI and backed by the same FastAPI APIs.

- [x] Add local browser chat history.
- [x] Add session switching in the web UI.
- [x] Add hover-to-delete chat cleanup in the web UI.
- [x] Add backend chat-session delete endpoint.
- [x] Expose trace/cache status for browser inspection.

---

## Phase 9: Voice

- [x] Add microphone input in web UI.
- [x] Transcribe speech to text with backend ElevenLabs Scribe v2 batch STT.
- [x] Use automatic language detection; do not manually configure input language.
- [x] Send verbatim transcript to agent with voice metadata.
- [x] Store `detected_language` and `language_confidence` with transcript, confidence, timestamps, and modality.
- [x] Default generation answer language to detected language unless session state has an override.
- [ ] Add optional TTS for reading answers aloud.
- [x] Use STT confidence to trigger clarification for low‑confidence transcripts.

---

## Phase 10: Imaging

- [x] Add image upload in web UI.
- [x] Store image metadata (modality, file path, linked report ID).
- [x] Implement image grounding route wiring (imaging_qa agent stub).
- [ ] Integrate MedGemma (primary) and Qwen2‑VL (fallback) for vision.
- [ ] Implement retrieval of nearest reports/captions from MIMIC‑CXR/ROCOv2.
- [ ] Add cited, grounded imaging answers with disclaimers.

---

## Phase 11: Observability

- [ ] Add LangSmith tracing to LangGraph runs (router, retrieval, tools, generation, guardrails).
- [x] Add Prometheus metrics (requests, latency, cache, tools, eval).
- [ ] Add Grafana‑ready dashboards/panels (latency, route mix, cache hit rate, eval trends).
- [ ] Tag traces and metrics with `workspace_id`, `route`, `cache_hit`, `eval_config_id`, and `variant` (for A/B).

---

## Phase 12: Persistence / Supabase Integration

- [ ] Define and migrate `profiles` table with upsert on first auth.
- [ ] Define and migrate `workspaces` table (default workspace per user).
- [ ] Define and migrate `documents` table (metadata + storage path).
- [ ] Define and migrate `caremind_document_chunks` table with pgvector column.
- [ ] Define and migrate `conversations` and `messages` tables.
- [ ] Ensure row‑level security and workspace scoping are configured.

---

## Phase 13: A/B Testing (Online)

- [ ] Add configuration for multiple LLM/embedding/retriever variants.
- [ ] Implement sticky variant assignment (hash over `user_id`/`workspace_id`).
- [ ] Log variant in Prometheus metrics (`variant` label).
- [ ] Optionally run secondary model in background for evaluation only (not user‑visible).
- [ ] Add internal dashboard panel comparing metrics by variant.

---

## Phase 14: Documentation & Demo

- [ ] Update `README.md` with architecture, setup, and demo steps.
- [ ] Document `specs.md`, `plan.md`, and `tasks.md` relationships.
- [ ] Add demo script (upload → QA → comparison → voice → imaging → eval run).
- [ ] Record or script a demo that showcases MIRAGE/PubMedQA evaluation and A/B testing.
