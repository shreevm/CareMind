# CareMind Specs
Version: 0.6
Status: MVP + Observability/Eval Loop + Guardrails + Multimodal Clinical Agent

This is the full consolidated spec. It supersedes v0.4 and v0.5; nothing from those is dropped. This document is additive and complete on its own.

## 1. Product Summary

CareMind is an agentic, multimodal RAG assistant for medical and research documents, images, and voice, with two user interfaces:
- Web chat app.
- VS Code sidebar extension.

Users upload medical PDFs, text files, clinical images (e.g. chest X-rays), or speak into the app, ask questions, compare reports, and receive evidence-grounded answers with citations. It is designed for medical research, education, clinical document interpretation, and nursing/clinical-knowledge support — not autonomous diagnosis and not emergency triage.

Answer modes:
- Document-grounded mode: answers from uploaded reports, cites uploaded passages.
- General medical education mode: answers from a trusted education corpus (MedCorp), cites the passage. Non-diagnostic.
- Clinical knowledge QA mode: answers general medical/nursing knowledge questions grounded in MedCorp (PubMed, StatPearls, textbooks), benchmarked against MIRAGE.
- Imaging mode: accepts an uploaded clinical image (chest X-ray initially), returns a grounded description tied to retrieved similar reports, via a vision-capable model.
- Report comparison mode: diffs two documents and summarizes clinically relevant changes.
- Voice input mode: transcribes spoken questions into text and routes them through the same agent stack as typed input.

All modes carry a non-diagnostic disclaimer and route away from anything resembling emergency symptoms or dosage requests (see Section 10, emergency_redirect).

## 2. Goal

Ship a system that demonstrates:
- agentic planning and routing across multiple clinical knowledge domains,
- LangGraph-based workflow orchestration,
- multimodal retrieval-augmented generation (text, imaging, and voice-transcribed text),
- MCP-based tool use,
- evidence citations,
- shared web + VS Code experience,
- closed-loop evaluation (loop engineering) against a real benchmark (MIRAGE),
- guardrails appropriate for real, credentialed clinical data (MIMIC),
- no GPU cost for LLM/embedding inference (NVIDIA free endpoints) except where a vision model requires local/hosted inference.

## 3. Target Users

- Medical students.
- Researchers.
- Clinicians and nurses reviewing documents or images.
- Health-tech demo evaluators.
- AI/ML recruiter reviewers.

## 4. Core Use Case

User uploads one or more medical documents or images, or speaks a question, and asks:
- What are the key findings?
- What changed between reports?
- Explain this in simple language.
- What evidence supports this statement?
- Explain pneumonia, anemia, hypertension, diabetes, or chest pain in simple educational language.
- What does this chest X-ray show, grounded in similar reported cases?
- General nursing/clinical-knowledge questions such as standard care protocols, medication classes, and lab value interpretation at an educational level.

The system routes the query, retrieves relevant passages or images, reasons over them, and returns a cited response.

## 5. Datasets

### 5.1 Text corpora

| Purpose | Dataset | Access | Notes |
|---|---|---|---|
| Primary demo (synthetic, no credentialing) | PDF Deid Dataset (JohnSnowLabs/pdf-deid-dataset) | Open | Synthetic PDFs, Easy/Medium/Hard, ideal for ingestion/OCR testing |
| Primary demo (synthetic, no credentialing) | Synthetic Australian Medical Documents Sample | Open, CC-BY-NC 4.0 | 50 docs, 29 types, includes scanned variants |
| Document-grounded QA (real data) | MIMIC-IV-Note | PhysioNet credentialed | Discharge summaries and radiology report text; powers document-grounded mode on real clinical notes |
| Clinical knowledge QA / benchmarking corpus | MedCorp (PubMed + StatPearls + Textbooks + Wikipedia, via MedRAG toolkit) | Open | Backing corpus for medical_knowledge_qa route and MIRAGE benchmarking |
| Evaluation only (not a RAG corpus) | MIRAGE (MedQA, MedMCQA, PubMedQA, BioASQ, MMLU-Med) | Open | Used to score the agent, not to answer from |

### 5.2 Imaging corpora

| Purpose | Dataset | Access | Notes |
|---|---|---|---|
| Chest X-ray + report grounding | MIMIC-CXR | PhysioNet credentialed | Images paired with real radiology reports; primary imaging dataset |
| Broader modality coverage | ROCOv2 | Open, no credentialing | Captions are shorter/less structured than MIMIC reports; treat as secondary corpus |

MIMIC-IV itself has no images and is not directly ingested by the RAG pipeline; MIMIC-IV-Note and MIMIC-CXR are the text/image components actually used.

### 5.3 Data handling note

MIMIC-IV, MIMIC-IV-Note, and MIMIC-CXR are real, de-identified patient data under a PhysioNet Data Use Agreement, not synthetic. This changes the project's security posture from "defense in depth for a hypothetical" to load-bearing.

## 6. Models

### 6.1 LLM (NVIDIA free endpoint, text)

base_url = https://integrate.api.nvidia.com/v1

Recommended models:
- meta/llama-3.1-8b-instruct — default for MVP, recruiter-friendly demo.
- deepseek-ai/deepseek-v4-flash — strong general RAG option.
- nvidia/nemotron-3-ultra-500b — long-context, agentic reasoning.
- nvidia/nemotron-4-340b-instruct — general instruction-following.

### 6.2 Embedding model

- nvolveqa_40k — default for MVP.
- NV-EmbedQA-E5-v5 — QA-retrieval optimized alternative.
- NV-Embed-v2 — generalist, high MTEB rank.

### 6.3 Vision-language model (imaging mode)

Two candidates; verify current versions/availability before committing:
- MedGemma — trained specifically on medical imaging, chest-X-ray heavy; closer domain fit for MIMIC-CXR.
- Qwen2-VL — general-purpose vision-language, strong OCR/general image understanding, not medically fine-tuned.

Default: MedGemma for chest X-ray description; fall back to Qwen2-VL for other modalities where no medically tuned option is in scope.

### 6.4 Router model

A small model or LoRA-tuned classifier dedicated to route prediction, separate from the generation LLM. This keeps routing latency and cost low and makes routing behavior independently evaluable.

### 6.5 Optional reranker

NVIDIA rerank model if available, or simple similarity-based rerank.

## 7. Scope

### In scope
- Web chat interface.
- VS Code sidebar interface.
- PDF, text, image, and voice upload/input.
- Synthetic demo datasets for keyless/no-credential demos.
- Real clinical datasets for the credentialed deployment path.
- Chunking and embedding with NVIDIA.
- Vector retrieval with Pinecone.
- Agentic query routing across an expanded route taxonomy.
- MCP tools: document search, report comparison, timeline extraction, medical education search, imaging search.
- Citation-backed answers.
- Conversation history and session state in Redis.
- Exact-match and semantic response caching in Redis.
- LangSmith tracing of every agent run.
- Prometheus metrics, Grafana-compatible.
- Eval harness with baseline diffing against MIRAGE and internal test sets.
- Prompt-injection and medical-safety guardrails.
- LoRA-tuned router for route classification.
- NVIDIA free endpoints for text LLM/embeddings; vision model per Section 6.3.
- Speech-to-text preprocessing for voice input.
- Optional text-to-speech output.

### Out of scope
- Full PHI compliance certification.
- EHR/FHIR integration.
- DICOM support.
- Multi-agent orchestration beyond the single router + tool-call pattern.
- Voice-first real-time speech-to-speech as the primary architecture.
- Mobile app.
- Fine-tuning of the generation LLM.
- Real-time clinical decision automation or emergency triage.
- Complex user roles.
- Paid GPU or enterprise inference beyond what's needed for the vision model.

## 8. Functional Requirements

### FR-1 Authentication
Basic login for private workspace access. MVP may use simple auth or no-auth local sessions.

### FR-2 Document, Image, and Voice Upload
Users shall upload PDF, text, and image files through web and VS Code, and may submit voice input through the web app.

### FR-3 Document Processing
Extract text, split into chunks, and index for retrieval. For images: store alongside extracted metadata such as modality and associated report if present. For voice: transcribe to text and store transcript metadata.

### FR-4 Agentic Query Routing
The router shall classify each query into one of the routes defined in Section 10.1, including a mandatory emergency_redirect path.

### FR-5 Retrieval
Retrieve top relevant chunks (text) or nearest report matches (imaging) before generating answers. Vector search shall use Pinecone with NVIDIA embeddings.

Local dev may use SQLite-backed local vector search.

### FR-6 MCP Tool Use
Exposed MCP tools: document search, report comparison, timeline extraction, medical education search, imaging search. Implemented as a separate server the backend agent calls.

### FR-7 Answer Generation
Generate concise, evidence-grounded answers using retrieved text and/or image context via the NVIDIA LLM endpoint or vision model.

### FR-8 Citations
Every factual answer shall include citations to document passages, retrieved evidence, or the source report tied to an imaging match.

### FR-9 Memory
Store uploaded files, extracted text, conversation history, document summaries. Session and cache state in Redis. If Redis is unavailable, fall back to SQLite chat history and disable response caching.

### FR-10 Safety Checks
Avoid diagnostic claims, flag uncertainty, refuse unsupported medical advice, warn output is for educational use, and hard-redirect anything resembling an emergency or dosage request.

### FR-11 Evaluation
Repeatable evaluation script measuring route accuracy, citation pass rate, average latency, endpoint health, and MIRAGE-benchmarked accuracy for the clinical knowledge route.

### FR-12 Monitoring
/metrics in Prometheus format exposing request counts, route counts, tool call counts, cache hit/miss, and latency by route.

### FR-13 Semantic Cache
Redis semantic cache check using cosine similarity above a threshold before retrieval/generation, scoped per workspace/document set, in addition to exact-match cache.

### FR-14 Tracing
LangSmith tracing of every agent run: router decision, retrieval results, tool calls, generation prompts, guardrail outcomes.

### FR-15 Prometheus Metrics
/metrics in Prometheus exposition format: requests, latency, cache hit/miss, tool calls, errors.

### FR-16 Eval Harness with Baseline Diffing
Persist each eval run, diff against last passing baseline, report PASS/WARN/FAIL per metric with configurable thresholds.

### FR-17 Prompt Injection Guardrail
Screen uploaded documents/images and queries for injection patterns; treat all retrieved/document content as untrusted data, never instructions.

### FR-18 Medical Safety Guardrail
Screen outputs for diagnostic-sounding claims, enforce non-diagnostic disclaimers, flag low-confidence answers, hard-block emergency/dosage requests.

### FR-19 Guardrail Evaluation
Maintain an adversarial test set; report guardrail pass rate as a first-class eval metric subject to baseline diffing.

### FR-20 Multimodal Ingestion
Accept clinical images, extract modality metadata, and embed/index for retrieval alongside paired reports where available or captions where not.

### FR-21 Imaging QA
Given an uploaded image and a question, retrieve the nearest matching reports/captions and generate a grounded description via the vision-language model, with citations to the retrieved source.

### FR-22 Router Classification Quality
The router shall be evaluated and tuned as an independent component, with route accuracy tracked separately from downstream answer quality.

### FR-23 MIRAGE Benchmarking
The evaluation harness shall support running the agent against MIRAGE's five sub-datasets and reporting per-dataset and aggregate accuracy.

### FR-24 Voice Input
The system shall accept voice input in the web app and transcribe it before routing.

### FR-25 Voice Transcription Metadata
Voice transcripts shall carry metadata such as transcript text, confidence, language, timestamps, and input modality.

### FR-26 Optional TTS Output
The system may optionally read final answers aloud using text-to-speech.

## 9. Architecture

User (Web/VSCode) → FastAPI backend → input guardrail → router → Redis cache check → route-specific retrieval/tool/model path → output guardrail → response + metrics + cache

Offline:
- evaluate.py → internal test set + MIRAGE → scoring
- baseline diff → PASS/WARN/FAIL
- Prometheus Pushgateway → Grafana dashboard
- LangSmith trace links on failing examples

Router training:
- labeled route examples → LoRA fine-tune → router model → route accuracy eval

Storage & infrastructure:
- Pinecone: vector DB for text chunks and image-report embeddings, metadata-filtered by workspace/document.
- Redis: session state, exact-match cache, semantic cache, conversation memory.
- PostgreSQL/SQLite: metadata, user data, chat history.
- Object storage: local disk (demo) or S3-compatible cloud storage.

Voice pipeline:
- audio input → speech-to-text → normalized transcript → same backend agent path as text.

## 10. Routing Taxonomy and Agent Behavior

### 10.1 Routes

| Route | Purpose | Backing source |
|---|---|---|
| clinical_document_qa | Questions about an uploaded document | MIMIC-IV-Note or synthetic uploads |
| medical_knowledge_qa | General medical/clinical knowledge questions | MedCorp, benchmarked via MIRAGE |
| imaging_qa | Questions about an uploaded image | MIMIC-CXR (X-ray), ROCOv2 (other modalities) |
| report_comparison | Diff between two uploaded reports | MCP comparison tool |
| nursing_care_qa | Nursing-scope clinical knowledge | MedCorp subset, curated |
| emergency_redirect | Anything resembling emergency symptoms or dosage requests | Hard-coded refusal + redirect to professional/emergency care, no generation |
| clarify | Ambiguous query | Clarification prompt back to user |

### 10.2 Emergency redirect is non-negotiable
emergency_redirect bypasses generation entirely. It is not a prompt-level instruction to the LLM to "be careful"; it's a separate code path triggered before generation.

### 10.3 LoRA-tuned router
Build a labeled dataset of 200-500+ examples across all seven routes. LoRA fine-tune a small base model dedicated to route classification, kept separate from the generation LLM. Do not LoRA fine-tune the generation model on medical text by default; RAG grounding is preferred over parametric fine-tuning for factual claims.

## 11. Security & Guardrails

### 11.1 Prompt injection defense
Retrieved chunks and uploaded documents/images are treated as untrusted data; the system prompt explicitly instructs the model never to follow instructions found inside them. Retrieved content is delimiter-isolated in the prompt. Pre-check screens uploads/queries for injection patterns. Output-side check verifies no system-prompt leakage and no deviation from scope.

### 11.2 Medical safety guardrails
No diagnostic claims. Mandatory non-diagnostic disclaimer on every clinically relevant answer. Uncertainty flagging when retrieval confidence or citation coverage is low. Hard refusal + redirect for emergency_redirect.

### 11.3 Real PHI posture
Comply with the PhysioNet Data Use Agreement: no re-identification attempts, no data leaving the controlled environment, access logging enabled. No sensitive content in logs. Redis TTLs enforced on all cached content, including semantic cache entries. Pinecone entries scoped per workspace with metadata filtering. Synthetic datasets remain the default for public/keyless demos; MIMIC-backed mode is a separate, access-gated deployment path.

### 11.4 Guardrail test set
Adversarial eval set: direct injection attempts, jailbreak prompts embedded in uploaded documents, diagnostic-request edge cases, off-topic scope-creep, and simulated emergency-symptom queries to confirm emergency_redirect fires. Guardrail pass rate is tracked in the eval harness.

## 12. Loop Engineering (Eval Loop)

### 12.1 The problem it solves
Running an eval script once produces a number with nothing to compare it to, nothing to decide if it's a regression, and nothing to trigger action.

### 12.2 The loop
1. RUN → evaluate.py produces metrics.
2. COMPARE → diff current metrics against the last saved baseline.
3. DECIDE → flag PASS/WARN/FAIL per metric against a threshold.
4. ACT → surface FAIL in dashboard/CI/alert.
5. UPDATE → on PASS, current run becomes the new baseline.

### 12.3 Implementation
evaluate.py writes results to eval_runs/<timestamp>.json and to eval_runs/baseline.json on PASS. Comparison step loads baseline.json, computes per-metric deltas, applies thresholds. Results are pushed to Prometheus Pushgateway for Grafana trend lines. Optional GitHub Action runs the harness on PRs touching prompts/chunking/retrieval/routing code.

### 12.4 Testing "with loop" vs "without loop"
Fixed labeled test set plus MIRAGE for the knowledge route. Without loop: compare numbers manually. With loop: diff is programmatic and produces a PASS/FAIL table automatically.

## 13. Semantic Response Caching (Redis)

Embed each query with the same NVIDIA embedding model used for retrieval. Check Redis for a cached entry whose stored question-embedding has cosine similarity above threshold (default 0.92) to the incoming query, scoped to the same workspace/document set. Hit: return cached answer + citations, mark cache_hit=semantic, skip retrieval and generation. Miss: run full pipeline and store question, embedding, answer, citations, route, doc_ids. Exact-match cache is checked first. Invalidate on document re-upload/deletion. Fallback: if Redis or the vector index is unavailable, disable semantic cache, fall back to exact-match only.

## 14. LangSmith Tracing

Trace per run: router decision, retrieval step, MCP tool calls, generation step, guardrail step, total and per-node latency. Wrap the LangGraph invocation with LangSmith tracing. Tag traces with workspace_id, route, cache_hit. Failing eval examples link directly to traces.

## 15. Prometheus Metrics

- caremind_requests_total{route, status}
- caremind_latency_seconds{route}
- caremind_cache_hit_total{type="exact"|"semantic"} / caremind_cache_miss_total
- caremind_tool_calls_total{tool_name}
- caremind_errors_total{stage}
- caremind_eval_route_accuracy
- caremind_eval_citation_pass_rate
- caremind_eval_faithfulness
- caremind_eval_mirage_accuracy

Use prometheus_client to register metrics and expose /metrics. Grafana panels should show request rate by route, p50/p95 latency, cache hit rate over time, and eval metric trend lines with pass/warn/fail bands.

## 16. Eval Harness

### 16.1 Scoring dimensions
- Answer correctness.
- Relevance.
- Groundedness/faithfulness.
- Retrieval relevance.
- Route accuracy.
- Citation pass rate.
- Guardrail pass rate.
- MIRAGE accuracy.

### 16.2 Scoring method
Correctness/relevance/groundedness via LLM-as-judge against labeled data. Retrieval relevance, route accuracy, citation pass rate computed directly from agent output. MIRAGE scored using its standard zero-shot question-only retrieval protocol.

### 16.3 Output
eval_runs/<timestamp>.json and eval_runs/baseline.json. Failing examples link to LangSmith traces.

## 17. User Flows

### Flow A: Ask a Question
Upload → route to clinical_document_qa or medical_knowledge_qa → retrieve from Pinecone → LLM answers with citations → shown in web/VS Code → session/cache in Redis.

### Flow B: Compare Reports
Upload two reports → route to report_comparison → MCP comparison tool identifies differences → agent summarizes with citations.

### Flow C: Continue Across Interfaces
Start in web chat → open VS Code extension → same workspace/history via shared backend and Redis.

### Flow D: General Medical/Nursing Education
Ask a question → route to medical_knowledge_qa or nursing_care_qa → retrieve from MedCorp → answer with citations → safety layer adds disclaimer.

### Flow E: Cached Repeat or Similar Question
Ask a previously answered or semantically similar question → Redis exact or semantic cache hit → cached answer returned quickly → metrics record cache hit and latency.

### Flow F: Imaging Question
Upload a chest X-ray → route to imaging_qa → nearest report match retrieved from MIMIC-CXR or ROCOv2 caption → vision model generates grounded description with citation → safety layer adds disclaimer.

### Flow G: Emergency Symptom or Dosage Request
Query resembles an emergency or dosage request → router fires emergency_redirect → generation is bypassed → user receives a refusal and redirect.

### Flow H: Injection Attempt
Uploaded document or query contains an injection pattern → input guardrail flags it → request is blocked or routed to a stricter mode.

### Flow I: Voice Question
User speaks into the microphone → speech-to-text produces a transcript → transcript is routed through the same agent path as typed text → answer is returned with citations.

## 18. Tech Stack

- Frontend web: Next.js/React, Tailwind, streaming chat UI.
- Frontend VS Code: TypeScript, VS Code extension API, WebView sidebar.
- Backend: FastAPI, LangGraph StateGraph, Pydantic, uvicorn.
- Agent graph nodes: input guardrail, route, check cache, direct response, document retrieval, imaging retrieval, report comparison, medical/nursing education retrieval, clarification, output guardrail, finalization.
- RAG & retrieval: NVIDIA embeddings, Pinecone, local SQLite fallback, simple recursive-ish text splitter.
- Vector DB: Pinecone with metadata filtering by document/workspace ID.
- Cache & session: Redis.
- Database: SQLite or PostgreSQL.
- MCP: Python or TypeScript MCP server.
- LLM inference: NVIDIA free endpoint.
- Vision inference: MedGemma primary, Qwen2-VL fallback.
- Voice: speech-to-text and optional text-to-speech.
- Tracing: LangSmith.
- Metrics: prometheus_client, Grafana.
- Router fine-tuning: LoRA on a small base model.
- Deployment: Docker; Render/Cloudflare/Fly.io/simple VM.
- Evaluation: python evaluate.py with internal test set + MIRAGE.

## 19. Non-Functional Requirements

- MVP runs locally or on a simple cloud host.
- MIMIC-backed mode requires a controlled, access-logged environment.
- Responses stream where possible.
- Indexing completes in under a minute for small documents.
- UI is simple and responsive.
- System is easy to demo.
- Clear separation between keyless synthetic-data demo and credentialed MIMIC-backed deployment.
- Stack avoids GPU costs for text inference during prototyping.

## 20. Environment Variables

```env
# NVIDIA
NVIDIA_API_KEY=...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_CHAT_MODEL=meta/llama-3.1-8b-instruct
NVIDIA_EMBEDDING_MODEL=nvolveqa_40k

# Vision model
CAREMIND_VISION_MODEL=medgemma
CAREMIND_VISION_ENDPOINT=...

# Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=caremind-index
CAREMIND_VECTOR_BACKEND=pinecone

# Redis
REDIS_URL=redis://localhost:6379/0
CAREMIND_SEMANTIC_CACHE_ENABLED=true
CAREMIND_SEMANTIC_CACHE_THRESHOLD=0.92
CAREMIND_CACHE_TTL_SECONDS=86400

# Auth
CAREMIND_USERNAME=...
CAREMIND_PASSWORD=...

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=caremind-dev

# Prometheus
CAREMIND_PROMETHEUS_PUSHGATEWAY_URL=http://localhost:9091

# Eval / loop engineering
CAREMIND_EVAL_FAIL_THRESHOLD=0.05
CAREMIND_EVAL_WARN_THRESHOLD=0.02
CAREMIND_EVAL_BASELINE_PATH=eval_runs/baseline.json

# Guardrails
CAREMIND_GUARDRAILS_ENABLED=true

# MIMIC access
PHYSIONET_CREDENTIALED=true
CAREMIND_DATA_MODE=synthetic
```

## 21. Security Baseline

- HTTPS in deployment.
- Encrypt stored files where possible.
- Keep dev and prod data separate.
- Do not store unnecessary identifiers.
- Do not log sensitive content.
- Synthetic/de-identified documents for the public demo path; MIMIC data only in the credentialed, access-logged deployment path.
- Basic auth for workspace access.
- Session and cache state in Redis with sensible TTLs.
- Never commit .env or API keys.
- PhysioNet Data Use Agreement compliance for all MIMIC-derived data.

## 22. Success Criteria

The system is successful if:
- a user can upload a document, ask a question, and get a cited answer,
- a user can upload a chest X-ray and get a grounded, cited description,
- a user can compare two reports,
- a user can continue the session in both web and VS Code,
- MCP-based tool use is demonstrated,
- NVIDIA free endpoints are used for text inference/embeddings,
- a general medical education and a nursing-scope question both route and answer correctly,
- voice input is accepted and routed through the same agent stack,
- Redis cache state is visible when Redis is configured,
- Pinecone vector retrieval is visible when CAREMIND_VECTOR_BACKEND=pinecone,
- a full agent run is inspectable end-to-end in LangSmith,
- /metrics is scrapeable by Prometheus and renders in Grafana,
- running evaluate.py twice with an intentional regression produces a visible PASS/FAIL diff,
- the medical_knowledge_qa route can be scored against MIRAGE and reported per sub-dataset,
- the router's route accuracy is independently reported and improved via LoRA fine-tuning,
- an injection attempt embedded in an uploaded document/image is neutralized,
- a diagnostic-sounding or emergency-sounding query is redirected,
- guardrail pass rate is visible in the eval report.

## 23. Demo Script

- Upload a synthetic medical PDF, ask a question, show cited answer.
- Ask a paraphrased version of the same question, show semantic cache hit.
- Upload a chest X-ray, ask what it shows, show grounded description with citation to a matched report.
- Compare two reports.
- Open the VS Code extension, continue the same conversation.
- Show one MCP tool call in action.
- Ask a general medical education question and a nursing-scope question, show both route correctly.
- Use voice input to ask a question and show the transcript routed through the same agent.
- Open LangSmith, show the trace for a prior query.
- Open Grafana, show request/latency/cache panels.
- Run python evaluate.py, show baseline saved, including a MIRAGE sub-dataset score.
- Make a deliberate regression, run evaluate.py again, show FAIL diff table.
- Upload a document containing an injection attempt, show it neutralized.
- Ask an emergency-symptom or dosage-style question, show emergency_redirect firing.
- Show guardrail pass rate and router route-accuracy in the eval report.

## 24. Voice Input

CareMind shall support voice input as an optional modality in the web app and, later, in the VS Code extension.

### 24.1 Voice Input Behavior
- Users may speak into the microphone to ask questions.
- The system shall transcribe voice to text before routing.
- The transcribed text shall be treated the same as typed text for agent routing, retrieval, citations, and safety checks.
- Voice input shall carry metadata such as transcription confidence, language, timestamps, and input modality.

### 24.2 Voice Processing Pipeline
- Audio input is captured in the frontend.
- Speech-to-text converts audio into a transcript.
- The transcript is normalized and sent to the backend agent.
- The backend agent routes the query through the existing CareMind workflow.
- Optional text-to-speech may render the final answer aloud.

### 24.3 Agent Integration
- Voice input shall not create a separate reasoning stack.
- The same router, retrieval, tool use, safety, and citation pipeline shall handle both text and voice.
- The agent may use transcript confidence to decide whether clarification is needed.
- Low-confidence transcription may trigger a clarification response.

### 24.4 Implementation Notes
- The MVP may use a chained architecture: STT -> agent -> optional TTS.
- Live speech-to-speech realtime models are out of scope for the first release unless they are required for a demo.
- Voice support shall remain optional so the core text/document workflow stays the primary build path.

## 25. Future Extensions

- PHI-aware secure mode as a full compliance program.
- FHIR/EHR integration.
- DICOM support.
- Longitudinal patient timelines.
- Medical term simplification.
- Lab trend graphs.
- Multi-document reasoning.
- Audit logging.
- SSO and enterprise auth.
- Paid inference or private LLM/vision-model deployment.
- Expanded imaging modalities with a dedicated report-paired dataset.

## 26. Why This Is Worth Building

This demonstrates agentic orchestration across multiple knowledge domains, multimodal RAG engineering, tool use, closed-loop evaluation against a real benchmark, targeted fine-tuning, production observability, and guardrail design appropriate for real clinical data.

## 27. Reality Check on Scope

This is a significant expansion from a one-week synthetic-only MVP. Recommended sequencing:
1. Guardrails.
2. Expanded router + LoRA tuning.
3. Eval harness with baseline diffing.
4. MIMIC-IV-Note document QA.
5. LangSmith tracing.
6. MIMIC-CXR imaging mode.
7. MIRAGE benchmarking.
8. Prometheus/Grafana.

Keep the synthetic-data demo path fully functional throughout.