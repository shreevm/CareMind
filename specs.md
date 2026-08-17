# CareMind Specs
Version: 0.8.0
Status: Web Chatbot MVP + Observability/Eval Loop + Guardrails + Multimodal Clinical Agent + Routing/Cache Fixes + External Literature Tool + A/B Eval + Chat Attachments

---

## 1. Product Summary

CareMind is an agentic, multimodal, multi agent RAG assistant for medical and research documents, images, and voice, exposed as a web chat application. It is designed for medical research, education, clinical document interpretation, and nursing/clinicalâ€‘knowledge support â€” not autonomous diagnosis and not emergency triage.[web:15]

Users can:

- Upload medical PDFs, text files, and clinical images as first-class chat attachments.
- Speak questions (voice input) or type them.
- Ask about key findings, differences between reports, educational explanations, and nursing care questions.
- Request similar published cases or PubMed style literature enrichment.

Answer modes:

- **Documentâ€‘grounded mode**: answers from uploaded documents, with citations.
- **General medical education mode**: answers from a trusted education corpus (MedCorp), with citations.[web:15]
- **Clinical knowledge QA mode**: general medical/nursing knowledge grounded in MedCorp, benchmarked against MIRAGE.[web:7][web:15]
- **External literature mode**: for explicit similarâ€‘case / PubMed / literature requests, uses a sanitized PubMed/NCBI search to enrich MedCorp results.[web:11][web:15]
- **Imaging mode**: accepts an uploaded clinical image (chest Xâ€‘ray initially), returns a grounded description tied to retrieved similar reports.
- **Report comparison mode**: diffs two documents and summarizes clinically relevant changes.
- **Voice input mode**: records audio in the browser, sends it to the backend, transcribes it with ElevenLabs Scribe v2 batch STT, preserves the original transcript text, stores detected-language metadata, and routes through the same agent stack.

All modes carry a nonâ€‘diagnostic disclaimer and route away from anything resembling emergency symptoms or dosage requests via an emergency_redirect path.

---

## 2. Goal

Ship an endâ€‘toâ€‘end webâ€‘based AI medical chatbot that demonstrates:

- Agentic planning and routing across clinical knowledge domains.
- LangGraph based workflow orchestration.
- Multimodal RAG (text, imaging, and voiceâ€‘transcribed text).
- MCPâ€‘based tool use.
- Evidence citations.
- Closed loop evaluation against MIRAGE and PubMedQAa style questions.[web:7][web:15][web:17]
- Guardrails appropriate for real deâ€‘identified clinical data (MIMIC).
- Minimal GPU costs by relying on NVIDIA free endpoints for text LLM/embeddings, with optional vision model hosting.

---

## 3. Target Users

- Medical students.
- Researchers.
- Clinicians and nurses reviewing documents or images.
- Healthâ€‘tech demo evaluators.
- AI/ML recruiters.

---

## 4. Core Use Case

User (authenticated via CareMind web app) uploads one or more medical documents or images, or uses voice input, and asks:

- â€œWhat are the key findings?
- â€œWhat changed between these two reports?
- â€œExplain this in simple language.â€
- â€œWhat evidence supports this statement?â€
- â€œExplain pneumonia / anemia / hypertension / diabetes in simple educational language.â€
- â€œWhat does this chest Xâ€‘ray show, grounded in similar reported cases?â€
- General nursing/clinicalâ€‘knowledge questions: protocols, medication classes, lab interpretations.
- â€œIs there a similar case or PubMed literature for this?â€ (external literature).

The system routes the query, retrieves relevant passages or imageâ€‘linked reports, reasons over them, and returns cited responses.

---

## 5. Datasets

### 5.1 Text Corpora

| Purpose | Dataset | Access | Notes |
|---|---|---|---|
| Primary demo (synthetic) | PDF Deid Dataset (JohnSnowLabs/pdf-deid-dataset) | Open | Synthetic PDFs, Easy/Medium/Hard, ingestion/OCR testing |
| Primary demo (synthetic) | Synthetic Australian Medical Documents Sample | Open, CCâ€‘BYâ€‘NC 4.0 | 50 docs, 29 types, includes scanned variants |
| Documentâ€‘grounded QA (real) | MIMICâ€‘IVâ€‘Note | PhysioNet credentialed | Discharge summaries and radiology reports |
| Clinical knowledge QA / benchmark corpus | MedCorp (PubMed + StatPearls + textbooks + Wikipedia, via MedRAG) | Open | Backing corpus for medical_knowledge_qa, aligned with MIRAGE/MedRAG.[web:7][web:15] |
| Evaluation only | MIRAGE (MedQA, MedMCQA, PubMedQA*, BioASQâ€‘Y/N, MMLUâ€‘Med) | Open | Used to evaluate knowledge route with questionâ€‘only retrieval.[web:7][web:11][web:15] |
| Evaluation (PubMedâ€‘focused) | PubMedQA | Open | Biomedical QA dataset over PubMed abstracts.[web:17] |

### 5.2 Imaging Corpora

| Purpose | Dataset | Access | Notes |
|---|---|---|---|
| Chest Xâ€‘ray + report grounding | MIMICâ€‘CXR | PhysioNet credentialed | Images paired with real radiology reports |
| Broader modality coverage | ROCOv2 | Open | Captions shorter/less structured; secondary corpus |

### 5.3 Data Handling Note

MIMICâ€‘IV, MIMICâ€‘IVâ€‘Note, and MIMICâ€‘CXR are real, deâ€‘identified patient data under a PhysioNet DUA, not synthetic. This requires a stricter security posture: no reâ€‘identification, no uncontrolled export, access logging, and separation between public synthetic demo and credentialed MIMIC deployment.

---

## 6. Models

### 6.1 LLM (NVIDIA Free Endpoint, Text)

`base_url = https://integrate.api.nvidia.com/v1`

Recommended models:

- `nvidia/nemotron-3.5-lightning-30b-a3b` â€” default generation model.
- `meta/llama-3.1-8b-instruct` â€” MVP / recruiterâ€‘friendly fallback.
- `deepseek-ai/deepseek-v4-flash` â€” strong general RAG option.
- `nvidia/nemotron-3-ultra-500b` â€” longâ€‘context, agentic reasoning.
- `nvidia/nemotron-4-340b-instruct` â€” highâ€‘end instruction following.

If the configured generation model emits `reasoning_content` separately from final `content`, the backend must keep those fields separate. Raw reasoning must never be exposed to users, persisted, cached, logged, sent to LangSmith, or concatenated into the final answer.

### 6.2 Embedding Model

- `Qwen/Qwen3-Embedding-0.6B` - default local embedding model.
- Use SentenceTransformers `encode_query` for retrieval/cache queries.
- Use SentenceTransformers `encode_document` for uploaded-document chunks, structured document-understanding evidence, OCR-derived evidence, and searchable corpus passages.
- Normalize embeddings and store provider, model, dimension, and `embedding_index_version` with every vector.
- Keep `nvidia/nv-embedqa-e5-v5` only as a rollback or controlled evaluation baseline when its implementation remains configured.
- Do not compare vectors produced by different providers, models, dimensions, or embedding index versions.

### 6.3 Vision-Language Models

CareMind separates document understanding from radiology interpretation.

- **Primary document understanding model**: configurable Groq/OpenAI-compatible document vision endpoint.
  - Used for medical PDFs, lab reports, blood reports, ECG reports as documents, discharge summaries, prescriptions, referral letters, mobile phone photos of reports, and scanned clinical documents.
  - Produces structured JSON rather than free-form answers.
  - Runs once per uploaded document version through `DocumentUnderstandingAgent`.
  - OCR is fallback/support when the VLM is unavailable, fails, or returns low confidence.
- **Radiology model**: MedGemma or another dedicated medical vision model when configured.
  - Used for chest X-ray, CT, MRI, and other radiology images through `ImagingAgent`.
  - If unavailable, CareMind must not fake pixel interpretation; it answers from linked report evidence or explains the limitation.
  - Independent from the document-ingestion pipeline.

### 6.4 Router Model

A small base model (or LoRAâ€‘tuned variant) dedicated to route prediction, separate from generation LLM. Keeps routing cheap, lowâ€‘latency, and independently evalâ€‘able.

### 6.5 Optional Reranker

NVIDIA rerank model if available, or simple similarityâ€‘based rerank.

### 6.6 Speech-to-Text Model

- Use the ElevenLabs Speech-to-Text API with `scribe_v2` as the default STT model.
- Automatically detect the spoken language.
- Generate a transcript with word-level timestamps.
- Pass the transcript and detected-language metadata directly to the existing CareMind agent pipeline.
- Do not introduce a separate translation model or translation stage.
- The configured CareMind LLM processes the transcript and returns the response in the detected or user-selected language.
- Allow medical key-term prompting when supported to improve recognition of clinical terminology.
- Keep STT provider and model configurable through environment variables.

---

## 7. Scope

### In Scope

- Web chat interface (Next.js/React, streaming).
- PDF, text, image, and voice upload/input.
- Synthetic demo datasets for keyless/noâ€‘credential demos.
- Real clinical datasets (MIMIC) for credentialed deployment.
- Chunking and embedding with NVIDIA.
- Vector retrieval with Supabase pgvector.
- Agentic routing across expanded route taxonomy.
- MCP tools: document search, report comparison, timeline extraction, medical education search.
- Optional PubMed/NCBI external literature tool with query sanitization.[web:11]
- Citationâ€‘backed answers.
- Conversation history and session state in Redis, with Postgres backing.
- Exact and semantic response caching in Redis.
- LangSmith tracing for each agent run.
- Prometheus metrics, Grafana dashboards.
- Eval harness with baseline diffing against MIRAGE and internal test sets.
- Promptâ€‘injection and medicalâ€‘safety guardrails.
- LoRAâ€‘tuned router for route classification.
- NVIDIA free endpoints for text LLM/embeddings.
- Speechâ€‘toâ€‘text for voice input; optional TTS.
- Offline A/B evaluation across multiple model configurations on MIRAGE and PubMedQA.

### Out of Scope

- Full PHI compliance certification.
- EHR/FHIR integration.
- DICOM support.
- Fully autonomous openâ€‘ended agents beyond supervisorâ€‘specialist pattern.
- Voiceâ€‘first realâ€‘time speechâ€‘toâ€‘speech.
- Mobile app.
- Fineâ€‘tuning the generation LLM on medical text.
- Realâ€‘time decision automation or emergency triage.
- Complex user roles.
- Paid GPU beyond vision needs.

---

## 8. Functional Requirements

### FRâ€‘1 Authentication

- Use Supabase Auth (email/password, OAuth, or magic link).
- On first authenticated request:
  - Ensure migrations have created the `profiles` table.
  - Upsert user profile by `auth_user_id` (see Section 13).
- Workspaces are scoped to users (owner + shared users in future).

### FRâ€‘2 Document, Image, and Voice Upload

- Users can upload PDF, text, image files via the web app.
- Users can submit voice input through the web app.
- Production design targets Supabase Storage for binaries and Postgres for metadata. The current local implementation stores binaries in private local upload paths and records storage bucket/path metadata for Supabase compatibility.
- The current local implementation stores binaries in private local upload paths while keeping storage bucket/path metadata compatible with Supabase Storage.
- Upload responses include an `attachment_id`; chat requests send `attachment_ids` instead of relying on filenames in message text.

### FRâ€‘3 Document Processing

- Extract text, split into chunks, embed, and store in Supabase pgvector.
- Run `DocumentUnderstandingAgent` once per uploaded document version.
- The agent calls the configured Groq/OpenAI-compatible document VLM when configured, extracts structured JSON, computes confidence, generates a concise summary, stores raw JSON, and creates structured evidence for embeddings.
- For report photos/scans: store file and metadata in `image_assets`, use document vision first when configured, then OCR as fallback/support, and save/index the structured output as a linked document with `documents.source_image_id`.
- Image OCR must be additive and non-destructive: if no readable text is found, the image asset remains available for paired-report/caption grounding; existing PDF/text ingestion behavior is unchanged.
- For audio: use ElevenLabs Scribe v2 batch speech-to-text with `language_code=null`/automatic language detection. Store the verbatim transcript text, confidence, word timestamps, modality, `detected_language`, and `language_confidence`; do not replace the transcript with translated or normalized text.

### FRâ€‘4 Agentic Query Routing

- Router classifies each query into the routes in Section 10.1, including mandatory emergency_redirect.
- Existing uploaded documents must not force unrelated general questions into document retrieval.
- Patientâ€‘name questions, uploadedâ€‘report references, and followâ€‘ups with active patient/report context route to clinical_document_qa unless user explicitly switches to general education/literature.
- Standalone general medical concept questions route to medical education unless they explicitly mention uploaded evidence, a report/document/image, the patient, or a named report. The classifier uses common medical-intent signals and trusted education topics rather than disease-specific routing branches.
- Ambiguous pronoun follow-ups use active report/image context only when a prior cited document/image answer established that context; otherwise they clarify or route as general education when the user asks a standalone medical concept question.
- Current-message attachments have highest reference priority. If a user attaches image A and asks "What is this?", image A wins over an old active report or workspace search.
- Multiple current-message attachments are preserved as an exact set for comparison routes such as "Compare these."

### FRâ€‘5 Retrieval

- Retrieve top relevant chunks (text) or nearest reports (imaging) before generation.
- Vector search uses Supabase pgvector with NVIDIA embeddings.
- Local dev may use SQLite + local vector search.
- For long reports, retrieve child chunks by vector similarity and expand selected hits with neighboring parent context before generation.
- For documents parsed by `DocumentUnderstandingAgent`, retrieval uses stored structured evidence chunks instead of repeatedly invoking the vision model.
- When `attachment_ids` are present, retrieval loads the linked document/image evidence directly and bypasses vector similarity for selecting the target attachment.

### FRâ€‘6 MCP Tool Use

- MCP tools exposed:
  - document search
  - report comparison
  - timeline extraction
  - medical education search
- Implemented as a separate MCP server called by backend agents.

### FRâ€‘6a External Literature Tool

- Optional `medlineplus_health_topic_search` tool for general health-topic education enrichment.
- Optional `pubmed_literature_search` tool for explicit similarâ€‘case / PubMed / external / literature requests.
- Uses NCBI Eâ€‘utilities; removes obvious patient identifiers; caps results; fails closed to internal MedCorp if unavailable.[web:11]

### FRâ€‘7 Answer Generation

- Generate concise, evidenceâ€‘grounded answers using retrieved text/image context via NVIDIA LLM or vision model.
- Include nonâ€‘diagnostic disclaimer on clinically relevant outputs.

### FRâ€‘8 Citations

- Every factual answer includes citations to:
  - Uploaded document passages,
  - Retrieved MedCorp snippets,
  - Source report linked to imaging match.

### FRâ€‘9 Memory

- Store uploaded file metadata, extracted text, conversation history, document summaries.
- Redis for session and cache; Postgres for durable history.
- If Redis unavailable, fall back to Postgres/SQLite for chat history and disable response caching.
- Maintain session state: current patient, current document/report, current image, last route, last tool, last retrieved chunks, last summary, last citations, and conversation entities.
- Maintain attachment focus: `active_attachment_id`, `active_attachment_type`, `last_attachment_ids`, `last_structured_evidence_id`, `current_topic`, and `conversation_focus`.
- Maintain document-level memory separately from conversation history, including cached document summaries and last cited chunks per document.
- Users can delete a chat session from the browser history. Deletion removes local browser history, durable server messages/context for that `session_id`, and the matching Redis session key when Redis is available.

### FR-9A Conversational Grounding

- Before routing, resolve follow-up questions against conversation state, entity memory, active documents, and recent history.
- The resolver produces `resolved_query`, `conversation_entities`, `active_document_ids`, and session-state updates.
- The resolver also produces a deterministic RECAP-style `intent_rewrite` containing the latest actionable `current_intent`, a `route_hint`, confidence, extracted constraints, and context target.
- The router may accept the `route_hint` when confidence is high after hard safety checks; otherwise it falls back to supervisor routing or clarification.
- DocumentRAGAgent retrieves using the resolved query rather than the raw user utterance.
- General medical education keeps a separate `conversation_focus` and `current_topic`; pronoun or bare-topic follow-ups such as `What causes it?` or `symptoms?` resolve to the last general medical topic, while document-scoped wording such as `my report`, `patient`, or `uploaded document` still routes to uploaded evidence.
- The original user wording remains available in traces and displayed chat history.

Examples:

- `Tell me about Mr. Venkat Ramanujam.` activates that patient/report.
- `Why fatigue?` resolves to a patient/report-scoped fatigue question.
- `Did he have chest pain?` resolves `he` to the active patient.
- `Compare it with the previous report.` resolves `it` to the active report before comparison routing.
- `Is this worse than last time? focus on kidney markers` rewrites to a comparison intent with a kidney-marker constraint before route selection.

### FR-9B Retrieval Reuse

- If a resolved follow-up can be answered from active retrieved chunks with sufficient confidence, reuse those chunks and skip a new vector search.
- If confidence is low, perform normal retrieval.
- Trace and metrics expose the reuse decision, confidence, reason, and whether search was bypassed.

### FR-9C Conversation-Aware Cache

- Exact and semantic cache keys use the resolved query, workspace revision, current `attachment_ids`, active attachment IDs, active document IDs, active patient/report/image context, route, model, embedding, and retrieval settings.
- Semantic cache embeddings are created from the resolved query, not raw follow-up text.
- Workspace revision changes from document/image/chunk mutations invalidate stale response namespaces.
- Agent traces and `/cache/debug` expose Redis status, exact/semantic key patterns, cache namespace, cache source, and key counts without returning cached medical text.

### FRâ€‘10 Safety Checks

- Avoid diagnostic claims and dosage recommendations.
- Flag uncertainty, refuse unsupported advice, enforce educationalâ€‘only disclaimer.
- Hardâ€‘redirect emergencies via emergency_redirect.

### FRâ€‘11 Evaluation

- Repeatable eval script measuring:
  - Route accuracy.
  - Citation pass rate.
  - Average latency.
  - Endpoint health.
  - MIRAGE accuracy for medical_knowledge_qa.[web:7][web:15]
  - PubMedQA accuracy for PubMedâ€‘style questions.[web:17]

### FRâ€‘12 Monitoring

- `/metrics` in Prometheus format:
  - request counts,
  - latency by route,
  - tool call counts,
  - cache hit/miss and lookup reasons,
  - retrieval reuse hit/miss counts.

### FRâ€‘13 Semantic Cache

- Redis semantic cache using cosine similarity:
  - Threshold default 0.92.
  - Scoped per workspace and cache namespace.
- Cache namespace includes:
  - route, workspace_revision, modality, top_k, active patient/report/image context, active document IDs, vector_backend, embedding_model, generation_model, min_retrieval_similarity.
- Exact keys additionally include normalized resolved query.
- Avoid stale answers when routing, retrieval thresholds, vector backend, embeddings, or model change.
- Debug/bypass flags skip cache reads/writes.

### FRâ€‘14 Tracing

- LangSmith tracing:
  - resolved query,
  - entity-resolution metadata,
  - router decision,
  - retrieval,
  - retrieval reuse decisions,
  - MCP tool calls,
  - sanitized generation configuration and final-answer metadata,
  - guardrail outcomes,
  - exact/semantic cache keys and Redis availability,
  - TTFT for streaming responses,
  - perâ€‘node and total latency.
- Traces tagged with `workspace_id`, `route`, `cache_hit`, and `eval_config_id` for eval runs.
- Raw reasoning, patient identifiers, raw retrieved chunks, embedding vectors, credentials, local file paths, and sensitive prompt or document content must not be sent to LangSmith.

### FRâ€‘15 Prometheus Metrics

- `caremind_requests_total{route,status,variant}`
- `caremind_latency_seconds{route,variant}`
- `caremind_cache_hit_total{type="exact"|"semantic"}`
- `caremind_cache_miss_total`
- `caremind_tool_calls_total{tool_name}`
- `caremind_errors_total{stage}`
- `caremind_eval_route_accuracy{eval_config_id}`
- `caremind_eval_mirage_accuracy{eval_config_id}`
- `caremind_eval_pubmedqa_accuracy{eval_config_id}`
- `caremind_eval_guardrail_pass_rate{eval_config_id}`

### FRâ€‘16 Eval Harness with Baseline Diffing

- `backend/eval_runs/<timestamp>.json` and `backend/eval_runs/baseline.json`.
- Compare current metrics vs baseline per config.
- Report PASS/WARN/FAIL with configurable thresholds (`CAREMIND_EVAL_FAIL_THRESHOLD`, `CAREMIND_EVAL_WARN_THRESHOLD`).
- Support multiple eval configs (`eval_config_id`) corresponding to different LLM/embedding/retriever variants.

### FRâ€‘17 Prompt Injection Guardrail

- Treat retrieved chunks and uploads as untrusted data; system prompt instructs model to ignore instructions in retrieved content.
- Preâ€‘check uploads/queries for injection; output check ensures no system prompt leakage/scope deviation.

### FRâ€‘18 Medical Safety Guardrail

- Block diagnosticâ€‘sounding or emergency queries; enforce emergency_redirect.
- Add disclaimers, flag low confidence, and refuse explicit treatment/dosage planning.

### FRâ€‘19 Guardrail Evaluation

- Maintain adversarial test set:
  - injection attempts,
  - jailbreaks,
  - diagnostic request edges,
  - emergency symptoms.
- Track guardrail pass rate as firstâ€‘class eval metric with baseline diff.

### FRâ€‘20 Multimodal Ingestion

- Accept clinical images; extract modality metadata; store original image assets; route report photos/scans/prescriptions/lab screenshots through `DocumentUnderstandingAgent`.
- A configured Groq/OpenAI-compatible document VLM is the primary document-understanding model target. OCR is fallback/support when the document VLM is unavailable, fails, or low confidence.
- The structured JSON is normalized into evidence text and indexed as a normal document linked by `source_image_id`.
- All structured/OCR-derived chunks use the shared vector index/table with metadata such as `{"source": "document_understanding", "document_type": "...", "understanding_model": "..."}` or `{"source": "ocr", "image_id": "...", "ocr_engine": "..."}`. CareMind does not create one index per document.
- OCR and document VLM parsing do not diagnose raw pixels. True radiology-image interpretation requires a configured medical vision model in the independent radiology pipeline.

### FRâ€‘21 Imaging QA

- Given an uploaded image + question:
  - Retrieve OCR-derived linked document chunks and/or paired reports/captions from the workspace.
  - Retrieve nearest matching reports/captions from MIMICâ€‘CXR/ROCOv2 when configured.
- For CXR/CT/MRI support, call a dedicated radiology model such as MedGemma from `ImagingAgent` when configured.
  - Do not route ordinary lab reports, prescriptions, forms, discharge summaries, or scanned documents through the radiology model.
  - Use radiology vision only when explicitly configured; otherwise answer from linked text evidence and state that the pixels were not independently diagnosed.

### FRâ€‘22 Router Classification Quality

- Evaluate and tune router as an independent component; route accuracy tracked separately from downstream answer quality.

### FRâ€‘23 MIRAGE Benchmarking

- Eval harness supports MIRAGEâ€™s five subâ€‘datasets, with:
  - Zeroâ€‘shot, multiâ€‘choice, questionâ€‘only retrieval settings.[web:11][web:15]
- Report perâ€‘dataset and aggregate accuracy for medical_knowledge_qa.

### FRâ€‘24 Voice Input

- Accept recorded voice input through the web application.
- Send the audio to ElevenLabs Scribe v2 for transcription.
- Automatically detect the spoken language.
- Pass the resulting transcript and language metadata directly to the existing CareMind safety, routing, and agent pipeline.
- The voice request must follow the same processing path as a typed message after transcription.
- Voice input is asynchronous request-response transcription, not real-time speech-to-speech communication.

### FR-24a Multilingual Voice Input

- Use ElevenLabs Scribe v2 to automatically detect and transcribe supported spoken languages.
- Preserve the transcript in its original language.
- Pass the original transcript directly to the existing CareMind LLM pipeline without adding a separate translation stage.
- Pass the detected language code as request metadata.
- Return the response in the detected language or the language selected by the user, unless the user requests another language.
- Apply input guardrails and emergency detection to the transcript before normal agent routing.
- Test mixed-language and code-switched inputs, especially combinations such as Tamil-English and Hindi-English.

### FRâ€‘25 Voice Transcription Metadata

Store:

- Original transcript text.
- Detected language code.
- Language-detection probability.
- Word-level timestamps when returned.
- Audio duration.
- Input modality as `voice`.
- STT provider as `elevenlabs`.
- STT model as `scribe_v2`.
- Transcription processing status and error information.
- Overall transcription confidence only when the configured provider returns it.

Do not incorrectly treat `language_probability` as overall transcription confidence.

### FRâ€‘26 Optional TTS Output

- Optionally synthesize audio for final answers.

### FRâ€‘27 A/B Testing of Model Configs

- Support multiple eval configurations (LLM + embeddings + retriever) offline and online:
  - Offline: run MIRAGE and PubMedQA for each config, store metrics and deltas.
  - Online: route a subset of live traffic to different model variants (A/B) while keeping guardrails identical.
- Variant assignment sticky per user/workspace (hashâ€‘based), tracked in metrics as `variant`.

---

## 9. Architecture

User (Web) â†’ Next.js frontend â†’ FastAPI backend â†’ input guardrail â†’ ConversationResolverAgent â†’ RECAP-style intent rewrite â†’ SupervisorAgent/router â†’ Redis cache check â†’ specialist agent â†’ routeâ€‘specific retrieval/tool/model path â†’ output guardrail â†’ response + metrics + cache.

Specialist agents:

- SupervisorAgent (router).
- DocumentUnderstandingAgent (upload-time only; configured document VLM structured extraction with OCR fallback; not a chat responder).
- ConversationResolverAgent (context grounding + current-intent rewrite).
- DirectResponseAgent (product/help questions).
- DocumentRAGAgent.
- MedicalEducationAgent (MedCorp, MIRAGE/PubMedQA eval).
- NursingEducationAgent (nursing subset of MedCorp).
- ReportComparisonAgent.
- ImagingAgent (image + report grounding; future MedGemma-style radiology model plug-in).
- ClarificationAgent.
- SafetyLayer (guardrails).
- ConversationMemory.

Offline:

- `evaluate.py` â†’ internal test set + MIRAGE + PubMedQA â†’ scoring, multiâ€‘config A/B.[web:7][web:15][web:17]
- Baseline diff â†’ PASS/WARN/FAIL.
- Prometheus Pushgateway â†’ Grafana dashboards.
- LangSmith trace links on failing examples.

Router training:

- Labeled route examples â†’ LoRA fineâ€‘tune â†’ router model â†’ route accuracy eval.

Storage/infrastructure:

- Supabase Postgres + storage + pgvector.[web:2][web:9]
- Redis for session state and caches.
- SQLite or Postgres for local dev metadata.
- Object storage: Supabase buckets or S3â€‘compatible.
- PubMed/NCBI Eâ€‘utilities as external literature source, with query sanitization.[web:11]

Voice pipeline:

Audio input → ElevenLabs Scribe v2 → original-language transcript + language metadata → input guardrail → ConversationResolverAgent → SupervisorAgent/router → appropriate specialist agent → response in detected or user-selected language.

---

## 9A. Final UI and Tracing Constraint

This section overrides Sections 4, 5, 6, 8, and 9 wherever they conflict.

### 9A.1 Preserve the Existing Agent Activity Trace

CareMind must preserve the current agent activity trace UI exactly as it works today.

Do not:

- Modify the existing agent activity trace layout, labels, events, state management, rendering, persistence, or styling.
- Add reasoning events to the existing agent activity trace.
- Add new frontend SSE or WebSocket event types for reasoning.
- Add a "Show reasoning" button, reasoning panel, reasoning-status indicator, collapsible reasoning section, or additional trace toggle.
- Modify frontend chat components unless a minimal compatibility fix is strictly required for final-answer streaming.

If a frontend compatibility change is required, flag it in the proposed diff and explain why before applying it. Final-answer tokens, citations, attachments, disclaimers, and existing agent-activity behavior must remain unchanged.

### 9A.2 Backend-Only Reasoning Handling

The selected model may emit `reasoning_content` separately from final `content`.

The backend must:

- Detect and consume `reasoning_content`.
- Never concatenate `reasoning_content` into final answer content.
- Stream only final `content` through the existing frontend answer event format.
- Never send raw `reasoning_content` to the frontend.
- Never add reasoning text to citations.
- Never add reasoning text to conversation history.
- Never add reasoning text to Redis or semantic caches.
- Never add reasoning text to stored assistant messages.
- Never add reasoning text to application logs.
- Never include reasoning text in user-visible errors.
- Discard raw reasoning after collecting permitted technical metadata.

Do not change the frontend event contract merely because the model emits a separate reasoning field.

### 9A.3 LangSmith Reasoning Metadata

Preserve the existing LangSmith technical workflow trace. Extend it only with sanitized reasoning metadata when measurable:

```json
{
  "reasoning_enabled": true,
  "reasoning_status": "complete",
  "reasoning_token_count": 0,
  "reasoning_duration_ms": 0,
  "generation_model": "nvidia/nemotron-3.5-lightning-30b-a3b"
}
```

LangSmith may continue tracing:

- Conversation resolution.
- Router decision.
- Selected specialist agent.
- Retrieval execution and result counts.
- MCP tool calls.
- Cache decisions.
- Generation configuration.
- Final-answer generation.
- Citation validation.
- Guardrail outcomes.
- Per-node and total latency.
- Errors and retries.

Do not send raw `reasoning_content`, private chain-of-thought, patient identifiers, raw retrieved medical chunks, embedding vectors, credentials, or local file paths to LangSmith.

If the existing LangSmith setup currently sends sensitive prompt or document content, treat that as a separate security concern. Do not expand the scope of a model-swap patch unless required to prevent new reasoning content from being exported.

### 9A.4 Guardrails

Existing input and output guardrails must remain unchanged.

The output guardrail must continue to inspect final answer content only. Raw reasoning is not user-facing output. Do not add a new reasoning guardrail in a model-swap patch.

Reasoning safety inspection remains an open architectural question.

### 9A.5 Documentation Correction

Replace any statement that says the UI displays reasoning or reasoning status with:

> The model may generate intermediate reasoning separately from final answer content. Raw reasoning is not exposed to users, persisted in conversation state, cached, logged, or sent to LangSmith. The existing CareMind agent activity UI remains unchanged. Sanitized reasoning metadata may be recorded in the backend LangSmith trace for debugging and evaluation.

Do not document a reasoning panel, "Show reasoning" control, frontend reasoning event, or reasoning status UI.

### 9A.6 Diff Scope

The model-swap patch should normally be limited to:

- NVIDIA generation client and configuration.
- Backend handling that separates reasoning from final content.
- Sanitized LangSmith reasoning metadata.
- Environment examples.
- Relevant model documentation.
- Backend tests.

Frontend files must not be changed unless there is a concrete final-answer streaming compatibility issue. If frontend files appear in the proposed diff, explain the necessity of each change before requesting approval.

---

## 10. Routing Taxonomy and Agent Behavior

### 10.1 Routes

| Route | Purpose | Backing source |
|---|---|---|
| clinical_document_qa | Questions about uploaded documents | MIMICâ€‘IVâ€‘Note or synthetic uploads |
| medical_knowledge_qa | General medical/clinical knowledge; MIRAGE/PubMedQA benchmarked | MedCorp + MedlinePlus + optional PubMed/model context |
| imaging_qa | Questions about uploaded images | MIMICâ€‘CXR, ROCOv2 |
| report_comparison | Diff between two uploaded reports | MCP comparison tool |
| nursing_care_qa | Nursingâ€‘scope clinical knowledge | MedCorp nursing subset |
| emergency_redirect | Emergency symptoms or dosage requests | Hardâ€‘coded refusal + redirect |
| clarify | Ambiguous query | Clarification prompt |

### 10.2 Emergency Redirect

- emergency_redirect bypasses generation entirely.
- Implemented as separate code path triggered before any LLM call.

### 10.3 LoRAâ€‘Tuned Router

- Labeled dataset of 200â€“500+ examples across all routes.
- LoRAâ€‘fineâ€‘tune small model dedicated to route classification.
- Generation LLM remains unfineâ€‘tuned; RAG grounding preferred.

### 10.4 Routing Cache and Context Safeguards

- Routing decision computed before cache lookup.
- Retrieval uses the standalone resolved query only. Conversation history may resolve references, but raw prior user/assistant messages are not appended to embedding queries.
- Retrieval reuse requires the same active evidence need, not merely the same active patient or document.
- Results below the configured minimum similarity are not passed to generation. If every candidate is below threshold, the evidence set is empty.
- Retrieval diagnostics enforce `kept_count + filtered_count == returned_count` and track below-threshold and duplicate filters separately.
- Qwen retrieval is scoped to `embedding_provider=qwen`, `embedding_model=Qwen/Qwen3-Embedding-0.6B`, `embedding_dimension=1024`, and `embedding_index_version=qwen3-0.6b-v1-1024`.

### 10.5 Report-Guidance Questions

- For questions asking what a patient should do, first search the complete active document chunks for explicit recommendations, follow-up plans, instructions, advice, conclusions, impressions or treatment guidance.
- Do not reuse prior key-finding chunks solely because the same document is active.
- Deduplicate repeated chunks before generation.
- If no explicit report guidance is found, state that no instructions were found in the retrieved sections.
- Add a separate "General next steps" section only when trusted medical-education evidence is retrieved.
- Cite uploaded-report statements only with document citations and general suggestions only with education citations.
- Do not use uncited model knowledge for patient next steps.
- Cache namespaces include route and retrieval/model settings.
- Contextual followâ€‘ups use active patient/report context.
- Phrases like â€œin generalâ€, â€œnot about the patientâ€, â€œliteratureâ€, â€œPubMedâ€, â€œsimilar caseâ€ switch away from document retrieval.

---

## 11. Security & Guardrails

### 11.1 Prompt Injection Defense

- Treat retrieved content as untrusted data.
- System prompt enforces ignoring instructions inside documents/snippets.
- Preâ€‘check inputs for injection patterns.
- Outputâ€‘side check for system prompt leakage and scope violations.

### 11.2 Medical Safety Guardrails

- No diagnostic claims or dosing advice.
- Mandatory educational disclaimer.
- Uncertainty flagging and safe failure behavior.
- Hard refusal + redirect for emergency/dosage queries.

### 11.3 Real PHI Posture

- PhysioNet DUA compliance:
  - No reâ€‘identification, no uncontrolled data export.
  - Access logging and environment controls.
- No sensitive content in logs.
- Redis TTLs on cached content.
- Supabase vector rows scoped per workspace with RLS/RBAC.
- Synthetic datasets as default for public demo; MIMIC mode as separate accessâ€‘gated deployment.
- External literature queries must never include patient identifiers; outbound query reduced to clinical pattern + literature intent.[web:11]

### 11.4 Guardrail Test Set

- Adversarial set includes:
  - injections,
  - jailbreaks,
  - diagnostic and emergency edge cases,
  - offâ€‘scope content.
- Guardrail pass rate tracked in eval reports.

### 11.5 Voice STT Security Requirements

- Validate audio MIME type, extension, duration, and file size before sending it to the STT provider.
- Do not place audio contents or transcripts in application logs.
- The public demo must accept only synthetic or non-sensitive voice input.
- Do not send PHI to ElevenLabs through a standard account.
- Any deployment processing PHI requires appropriate contractual and technical controls, including an applicable BAA and Zero Retention Mode.
- Store audio privately and associate it with the user, workspace, session, and attachment ID.
- Define a retention period for stored audio and permit deletion when no longer needed.

---

## 12. Loop Engineering (Eval Loop)

### 12.1 Loop

1. RUN â†’ `evaluate.py` produces metrics for one or more eval configs (MIRAGE + PubMedQA + internal tests).
2. COMPARE â†’ diff metrics against baseline per config.
3. DECIDE â†’ PASS/WARN/FAIL per metric.
4. ACT â†’ surface FAIL in dashboard/CI/alerts.
5. UPDATE â†’ on PASS, new baseline stored.

### 12.2 Implementation

- Results written to `backend/eval_runs/<timestamp>.json` and `backend/eval_runs/baseline.json`.
- Comparison step applies thresholds (`CAREMIND_EVAL_FAIL_THRESHOLD`, `CAREMIND_EVAL_WARN_THRESHOLD`).
- Optional GitHub Action runs harness on PRs.

---

## 13. Persistence Model (Supabase + Upsert Behavior)

### 13.1 User & Profiles

- Authentication via Supabase Auth (`auth.users`).[web:2][web:9]
- `profiles` table:

  ```sql
  CREATE TABLE IF NOT EXISTS profiles (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    auth_user_id uuid UNIQUE NOT NULL,
    email text,
    created_at timestamptz DEFAULT now(),
    last_seen_at timestamptz DEFAULT now(),
    default_workspace_id uuid,
    feature_flags jsonb DEFAULT '{}'::jsonb
  );
  ```

- On every authenticated request:
  - Ensure migrations have run (handled at startup, not per request).
  - Upsert profile:

    ```sql
    INSERT INTO profiles (auth_user_id, email, created_at, last_seen_at)
    VALUES (:auth_user_id, :email, now(), now())
    ON CONFLICT (auth_user_id) DO UPDATE
    SET last_seen_at = EXCLUDED.last_seen_at;
    ```

### 13.2 Workspaces & Documents

- `workspaces(id, owner_id, name, data_mode, created_at)`.
- On first request for a user with no workspace:
  - Create default workspace; store `default_workspace_id` in `profiles`.
- `image_assets(image_id, workspace_id, filename, content_type, file_path, modality, uploaded_at, report_text_summary, ocr_document_id)`.
- `message_attachments(id, message_id, conversation_id, workspace_id, user_id, attachment_type, filename, mime_type, file_size, storage_bucket, storage_path, image_asset_id, document_id, processing_status, processing_error, created_at, expires_at, persistence_mode)`.
- `documents(id, workspace_id, file_path, mime_type, title, created_at, source_dataset, source_image_id NULLABLE, document_type, understanding_confidence)`.
- `document_understanding(document_id, workspace_id, filename, content_type, document_type, summary, confidence, model_name, extraction_source, raw_json, index_text, created_at, updated_at)`.
- OCR-derived image documents use `source_image_id` to link back to `image_assets.image_id`; their content type is `application/x-ocr-json` or `text/plain`.
- `image_assets.ocr_document_id` links forward to the OCR-derived document so deleting the image can also remove the derived text/index rows.
- Raw files are stored in private local upload paths in the current implementation; the attachment metadata includes `storage_bucket` and `storage_path` for Supabase Storage migration.

### 13.3 Chunks & Vectors

- `chunks(chunk_id, document_id, workspace_id, document_name, page, position, text, embedding_json, embedding_provider, embedding_model, embedding_dimension, embedding_index_version, metadata_json)`.
- `chunk_embeddings(chunk_id, document_id, workspace_id, embedding_json, embedding_provider, embedding_model, embedding_dimension, embedding_index_version, created_at, updated_at)` stores versioned local embeddings for side-by-side migration.
- Default embeddings use local `Qwen/Qwen3-Embedding-0.6B` through SentenceTransformers with `encode_query` for retrieval/cache queries and `encode_document` for uploaded-document chunks, document-understanding evidence, OCR-derived evidence, and searchable corpus passages.
- Supabase Qwen vectors are stored in `caremind_document_chunks_qwen3_1024` with `embedding vector(1024)`, provider/model/dimension/index-version metadata, and a matching HNSW cosine index.
- Existing NVIDIA `nvidia/nv-embedqa-e5-v5` vectors remain recoverable in their legacy 1024-dimensional tables. Do not compare Qwen vectors with NVIDIA vectors or any vectors from another provider/model/index version.
- Filtered by workspace, provider, model, dimension, and embedding index version. Structured evidence rows include document-understanding metadata; OCR rows preserve `source="ocr"` and `image_id`.
- OCR-derived chunks are stored with the same embedding provider/model/index-version metadata as normal documents; document boundaries are represented by `document_id`, not by separate indexes.

### 13.4 Conversations & Messages

- `conversations(id, workspace_id, user_id, created_at, route_default, eval_config_id NULLABLE)`.
- `messages(id, conversation_id, role, content, route, created_at, citations jsonb, tool_calls jsonb)`.

### 13.5 Eval Runs

- `eval_runs(id, created_at, git_sha, eval_config_id, metrics jsonb, is_baseline boolean)` (optional if you want DBâ€‘backed eval history).

---

## 14. LangSmith Tracing

- Trace every run:
  - router decision,
  - retrieval,
  - MCP calls,
  - generation,
  - guardrail,
  - cache hits/misses,
  - cache key/namespace and Redis debug metadata,
  - TTFT and total stream latency when streaming.
- Tag with `workspace_id`, `route`, `cache_hit`, `eval_config_id`, `variant`.
- Preserve the existing technical trace while excluding raw `reasoning_content`, private chain-of-thought, patient identifiers, raw retrieved medical chunks, embedding vectors, credentials, local file paths, and sensitive prompt or document content.
- Record sanitized reasoning metadata only when measurable, such as `reasoning_enabled`, `reasoning_status`, `reasoning_token_count`, `reasoning_duration_ms`, and `generation_model`.

---

## 15. Prometheus Metrics

As in FRâ€‘15, exposed at `/metrics` via `prometheus_client`.[web:18]

---

## 16. Eval Harness

### 16.1 Scoring Dimensions

- Answer correctness.
- Relevance.
- Groundedness/faithfulness.
- Retrieval relevance.
- Route accuracy.
- Citation pass rate.
- Guardrail pass rate.
- MIRAGE accuracy (per subâ€‘dataset and aggregate).[web:7][web:15]
- PubMedQA accuracy (yes/no/maybe).[web:17]

### 16.2 Methods

- LLMâ€‘asâ€‘judge for correctness, relevance, groundedness.
- Direct computation for retrieval relevance, route accuracy, citation pass rate.
- MIRAGE:
  - Multiâ€‘choice, zeroâ€‘shot, questionâ€‘only retrieval; must pick among A/B/C/D.[web:11][web:15]
- PubMedQA:
  - Yes/no/maybe classification; optionally compare â€œRAG vs noâ€‘RAGâ€ for PubMed questions.[web:17]

### 16.3 Voice Acceptance Tests

- English audio is transcribed and routed correctly.
- Tamil audio is automatically detected, transcribed, and answered in Tamil.
- Hindi audio is automatically detected, transcribed, and answered in Hindi.
- Code-switched audio such as Tamil-English is transcribed without crashing.
- The transcript reaches the same router used for typed messages.
- Emergency language spoken through voice reaches `emergency_redirect`.
- A failed or timed-out STT request returns a safe retry message.
- An unsupported or low-quality recording does not produce a fabricated transcript.
- The ElevenLabs API key is never present in frontend code, API responses, logs, or browser network requests.

### 16.4 Reasoning Separation Acceptance Tests

- The new NVIDIA generation model is selected by default.
- Environment variables override model parameters.
- Only provider-supported reasoning parameters are sent.
- `reasoning_content` is never concatenated into final answer content.
- Final answer tokens continue using the existing streaming event contract.
- Existing agent activity trace events remain unchanged.
- No new reasoning event is sent to the frontend.
- Citations and disclaimers remain unchanged.
- Output guardrails receive only final answer content.
- Raw reasoning is not written to logs.
- Raw reasoning is not stored in conversation history.
- Raw reasoning is not stored in Redis or semantic caches.
- Raw reasoning is not sent to LangSmith.
- Sanitized reasoning metadata is traced separately when measurable.
- Non-reasoning models remain backward-compatible.
- Existing frontend tests pass without changing agent activity rendering.
- Unsupported provider parameters produce a clear configuration error.
- External NVIDIA calls are mocked; automated tests must not call the live NVIDIA endpoint.

---

## 17. User Flows

- Flow A: Ask question on uploaded docs â†’ clinical_document_qa â†’ RAG + citations.
- Flow B: Compare reports â†’ report_comparison â†’ MCP compare tool + summary.
- Flow C: General education â†’ medical_knowledge_qa or nursing_care_qa â†’ MedCorp + MedlinePlus retrieval, with controlled model context when evidence is thin.
- Flow D: PubMed/similar case â†’ medical_knowledge_qa with external intent â†’ sanitized PubMed search + MedCorp mixing.[web:11]
- Flow E: Cached repeat â†’ exact/semantic cache â†’ quick answer.
- Flow F: Debug/bypass cache â†’ retrieval/generation fresh, trace shows cache_bypassed.
- Flow G: Imaging â†’ imaging_qa â†’ closest reports + vision model + citations.
- Flow H: Emergency/dosage â†’ emergency_redirect â†’ refusal + redirect.
- Flow I: Injection attempt â†’ guardrail neutralization.
- Flow J: Multilingual voice input → ElevenLabs Scribe v2 language detection and transcription → transcript passed directly to the existing safety and agent pipeline → response returned in detected or selected language.
- Flow K: Offline eval â†’ `evaluate.py` on MIRAGE/PubMedQA â†’ metrics + baseline diff.[web:7][web:15][web:17]
- Flow L: A/B test â†’ variant A/B for medical_knowledge_qa â†’ metrics per variant.

---

## 18. Tech Stack

- Frontend: Next.js/React, Tailwind, streaming chat UI.
- Backend: FastAPI, LangGraph, Pydantic, uvicorn.
- Storage: Supabase Postgres + storage + pgvector.[web:2][web:9]
- Cache: Redis (session + exact/semantic cache).
- DB: Supabase Postgres (SQLite for local dev).
- MCP: Python/TypeScript MCP server.
- External tools: MedlinePlus health-topic search for general education and PubMed/NCBI Eâ€‘utilities for explicit literature requests.[web:11]
- LLM inference: NVIDIA integrate API.
- Document vision: configured Groq/OpenAI-compatible VLM primary for document understanding.
- Radiology vision: MedGemma-style model is an optional ImagingAgent plug-in for CXR/CT/MRI.
- Voice STT: ElevenLabs Scribe v2.
- Optional TTS: configurable ElevenLabs TTS.
- Backend integration: ElevenLabs API called through FastAPI.
- The ElevenLabs API key must remain server-side and must never be exposed to the Next.js client.
- Tracing: LangSmith.
- Metrics: `prometheus_client`, Grafana.[web:18]
- Router fineâ€‘tuning: LoRA on a small base model.
- Deployment: Docker on Render/Cloudflare/Fly.io/simple VM.
- Evaluation: `uv run python backend/evaluate.py`.

---

## 19. Nonâ€‘Functional Requirements

- Runs locally or on simple cloud host.
- MIMIC mode requires controlled, accessâ€‘logged environment.
- Streaming responses where possible.
- Indexing < 1 minute for small docs.
- Simple, responsive UI.
- Clear separation between:
  - Synthetic, keyless demo path.
  - Credentialed, accessâ€‘gated MIMIC deployment.
- Avoid GPU costs for text inference during prototyping.

---

## 20. Environment Variables

Key variables (subset):

```env
# NVIDIA
NVIDIA_API_KEY=
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

# Document understanding model
CAREMIND_DOCUMENT_UNDERSTANDING_ENABLED=true
CAREMIND_DOCUMENT_VLM_BASE_URL=
CAREMIND_DOCUMENT_VLM_MODEL=llama-4-scout-17b-16e-instruct
CAREMIND_DOCUMENT_VLM_MIN_CONFIDENCE=0.55
CAREMIND_ATTACHMENT_STORAGE_BUCKET=local-uploads
CAREMIND_TEMP_ATTACHMENT_TTL_SECONDS=604800
CAREMIND_DOCUMENT_VLM_API_KEY=
CAREMIND_DOCUMENT_VLM_TIMEOUT_SECONDS=60

# Future radiology model
CAREMIND_RADIOLOGY_VISION_BASE_URL=
CAREMIND_RADIOLOGY_VISION_MODEL=google/medgemma-4b-it

# Supabase / pgvector
SUPABASE_URL=
SUPABASE_PUBLISHABLE_KEY=...
SUPABASE_SECRET_KEY=...
SUPABASE_DB_URL=...
SUPABASE_VECTOR_TABLE=caremind_document_chunks_qwen3_1024
SUPABASE_MATCH_FUNCTION=match_caremind_document_chunks_qwen3_1024
CAREMIND_VECTOR_BACKEND=supabase

# Image OCR
CAREMIND_IMAGE_OCR_ENABLED=true
CAREMIND_IMAGE_OCR_ENGINE=auto
CAREMIND_IMAGE_OCR_MIN_CHARS=30
CAREMIND_IMAGE_OCR_STRUCTURING_ENABLED=true

# Redis
REDIS_URL=redis://localhost:6379/0
CAREMIND_RESPONSE_CACHE_ENABLED=true
CAREMIND_SEMANTIC_CACHE_ENABLED=true
CAREMIND_SEMANTIC_CACHE_THRESHOLD=0.92
CAREMIND_CACHE_TTL_SECONDS=86400

# External literature search
CAREMIND_EXTERNAL_SEARCH_ENABLED=true
NCBI_EUTILS_BASE_URL=https://eutils.ncbi.nlm.nih.gov/entrez/eutils
NCBI_TOOL=caremind
NCBI_EMAIL=...
NCBI_API_KEY=...
CAREMIND_EXTERNAL_SEARCH_TIMEOUT_SECONDS=12
CAREMIND_EXTERNAL_SEARCH_MAX_RESULTS=4
CAREMIND_MEDLINEPLUS_ENABLED=true
MEDLINEPLUS_BASE_URL=https://wsearch.nlm.nih.gov/ws/query
MEDLINEPLUS_TOOL=caremind
MEDLINEPLUS_EMAIL=...
CAREMIND_MEDLINEPLUS_TIMEOUT_SECONDS=8
CAREMIND_MEDLINEPLUS_MAX_RESULTS=3
CAREMIND_MEDICAL_EDUCATION_MODEL_CONTEXT_ENABLED=true
CAREMIND_MEDICAL_EDUCATION_MODEL_CONTEXT_MIN_CHARS=900

# Speech-to-text and text-to-speech
STT_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=
ELEVENLABS_STT_MODEL=scribe_v2
ELEVENLABS_STT_TIMESTAMPS=word
ELEVENLABS_STT_DIARIZE=false
TTS_ENABLED=false

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
CAREMIND_EVAL_BASELINE_PATH=backend/eval_runs/baseline.json
CAREMIND_EVAL_CONFIG_ID=default

# Guardrails
CAREMIND_GUARDRAILS_ENABLED=true

# MIMIC access
PHYSIONET_CREDENTIALED=true
CAREMIND_DATA_MODE=synthetic

# A/B testing
CAREMIND_AB_TEST_ENABLED=true
CAREMIND_AB_TEST_VARIANTS=A,B
```

---

## 21. Security Baseline

- HTTPS in deployment.
- Encrypt files where possible.
- Separate dev/prod data.
- Do not store unnecessary identifiers.
- Do not log sensitive content.
- Synthetic/deâ€‘identified documents for public demo; MIMIC only in credentialed mode.
- Supabase Auth for workspace access.[web:2][web:9]
- Redis TTLs.
- Never commit `.env` or keys.
- PhysioNet DUA compliance for MIMIC data.

---

## 22. Success Criteria

The system is successful if:

- A user can sign up, create a workspace, upload a document, and get a cited answer.
- A user can upload a chest Xâ€‘ray and get a grounded, cited description.
- A user can compare two reports.
- MCPâ€‘based tool use is demonstrated.
- An explicit similarâ€‘case or PubMed/literature request uses sanitized external literature search when enabled.[web:11]
- NVIDIA free endpoints are used for text inference/embeddings.
- A general medical education and a nursingâ€‘scope question both route and answer correctly.
- Voice input is accepted and routed through the same agent stack.
- Redis cache state is visible when Redis is configured.
- Debug/bypass_cache requests skip exact and semantic cache and record that bypass in the trace.
- Supabase pgvector retrieval is required and visible in runtime health.
- A full agent run is inspectable endâ€‘toâ€‘end in LangSmith.
- `/metrics` is scrapeable by Prometheus and renders in Grafana.[web:18]
- Running `evaluate.py` twice with an intentional regression produces a visible PASS/FAIL diff.
- The medical_knowledge_qa route can be scored against MIRAGE and PubMedQA and reported per subâ€‘dataset.[web:7][web:15][web:17]
- The router's route accuracy is independently reported and improved via LoRA fineâ€‘tuning.
- An injection attempt embedded in an uploaded document/image is neutralized.
- A diagnosticâ€‘sounding or emergencyâ€‘sounding query is redirected.
- Guardrail pass rate is visible in the eval report.
- A/B experimentation between model configs is visible in metrics and eval JSON.
