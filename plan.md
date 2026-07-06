# CareMind Plan

## 1. Purpose
This document defines the technical implementation plan for CareMind. It translates the product requirements in `specs.md` into an executable architecture, build sequence, and system design.

## 2. Technical Strategy
CareMind will be built as a modular agentic RAG platform with:
- FastAPI backend.
- LangGraph workflow orchestration.
- NVIDIA-hosted text inference and embeddings.
- Pinecone vector search.
- Redis session and cache state.
- LangSmith tracing.
- Prometheus metrics.
- Web frontend plus VS Code extension.
- STT-first voice input.
- Optional vision model for imaging mode.

The system will favor retrieval-grounded generation over generation-model fine-tuning for factual correctness.

## 3. Architecture Overview
### 3.1 Core request path
User input enters the frontend, is normalized, routed through a FastAPI backend, checked by guardrails, passed to the router, optionally served from cache, then sent through retrieval, tool use, and generation, and finally screened again before response.

### 3.2 Core services
- API service: chat, upload, compare, metrics, eval, health.
- Agent service: LangGraph router and workflow nodes.
- Retrieval service: document chunk search, image-report matching.
- MCP tool server: document search, comparison, timeline, education, imaging tools.
- Cache/session layer: Redis with SQLite fallback.
- Eval service: seeded tests, baseline diffing, and report generation.
- Observability layer: LangSmith + Prometheus.

## 4. Data Flow
### 4.1 Text workflow
Upload PDF/text → extract text → chunk → embed → index → route query → retrieve relevant chunks → generate grounded answer → store response and citations.

### 4.2 Imaging workflow
Upload image → store image metadata → retrieve nearest report/caption → vision model generates grounded description → cite matched source.

### 4.3 Voice workflow
Capture audio → speech-to-text → normalize transcript → route as standard text input → generate answer → optionally speak response with TTS.

## 5. Routing Design
The router will classify requests into:
- clinical_document_qa
- medical_knowledge_qa
- imaging_qa
- report_comparison
- nursing_care_qa
- emergency_redirect
- clarify

The router should be cheap, fast, and independently measurable. A small classifier or LoRA-tuned router is preferred over using the main generation model for routing.

## 6. Retrieval Design
### 6.1 Text retrieval
Use NVIDIA embeddings and Pinecone for the main retrieval path. For local development, allow a fallback vector store or SQLite-backed approximation.

### 6.2 Imaging retrieval
Store image/report associations and retrieve nearest relevant reports or captions before vision-based generation.

### 6.3 Citation policy
Every factual answer must include citations to the retrieved source passages, image-associated source report, or trusted education corpus passage.

## 7. Cache Design
Use Redis for:
- session state.
- exact-match response cache.
- semantic cache using embedding similarity.
- temporary agent state.

Cache keys should be scoped by workspace and document set. If Redis is unavailable, fall back to SQLite history and disable semantic cache.

## 8. Safety Design
### 8.1 Input guardrails
Screen uploads and queries for prompt injection, scope creep, and emergency/dosage requests before routing.

### 8.2 Output guardrails
Screen generated responses for diagnostic-sounding claims, unsupported advice, and prompt leakage. Add a non-diagnostic disclaimer for medically relevant outputs.

### 8.3 Emergency behavior
Emergency-like queries must bypass generation and go to emergency_redirect immediately.

## 9. Observability Design
### 9.1 Tracing
Use LangSmith to trace:
- router decisions.
- retrieval results.
- tool calls.
- generation prompts.
- guardrail outcomes.
- per-node latency.

### 9.2 Metrics
Expose Prometheus metrics for:
- request counts.
- route counts.
- tool calls.
- cache hits/misses.
- error counts.
- latency by route.
- eval scores.

### 9.3 Dashboards
Grafana should show:
- latency trends.
- cache hit rate.
- route distribution.
- eval pass/fail bands.
- guardrail pass rate.

## 10. Evaluation Design
The evaluation loop should support:
- route accuracy.
- citation pass rate.
- groundedness / faithfulness.
- retrieval precision@k / recall@k.
- guardrail pass rate.
- average latency.
- MIRAGE benchmarking for the medical knowledge route.

Every eval run should be written to disk and compared against the last passing baseline. If metrics regress past thresholds, mark the run FAIL.

## 11. Frontend Design
### 11.1 Web app
Build a simple chat UI with:
- upload support.
- streaming responses.
- citations panel.
- route indicator.
- metrics page.
- optional voice button.

### 11.2 VS Code extension
Build a sidebar chat that shares:
- session state.
- workspace context.
- cached responses.
- document history.

The VS Code client should call the same backend API as the web app.

## 12. Implementation Phases
### Phase 1
Backend scaffold, document upload, chunking, embeddings, retrieval, citations.

### Phase 2
Router, guardrails, Redis cache, session memory, comparison tool.

### Phase 3
Evaluation harness, baseline diffing, LangSmith tracing, Prometheus metrics.

### Phase 4
Web UI and VS Code extension.

### Phase 5
Voice input and optional TTS.

### Phase 6
Imaging mode and vision model integration.

### Phase 7
Grafana dashboard, MIRAGE benchmarking, router tuning.

## 13. Key Tradeoffs
- Prefer RAG over generation fine-tuning for factual grounding.
- Keep the router separate from the generator.
- Keep voice as STT-first rather than native speech-to-speech for the MVP.
- Keep imaging optional until the text workflow is stable.
- Keep synthetic demo data separate from credentialed MIMIC-backed mode.

## 14. Non-Goals
- Full PHI compliance certification.
- EHR/FHIR integration.
- DICOM support.
- Multi-agent systems.
- Emergency triage automation.
- Paid GPU inference for text.
- Fine-tuning the main generator as the default path.

## 15. Build Principles
- Make every important behavior inspectable.
- Make every important metric measurable.
- Make routing cheap and testable.
- Make safety explicit rather than implied.
- Make demos work without credentialed data.