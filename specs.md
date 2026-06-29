# CareMind MVP Specs
Version: 0.4
Status: Viable MVP with deployment and evaluation pipeline

## 1. Product Summary
CareMind is a lightweight Slack-first agentic RAG assistant for medical and research documents with two user surfaces:
- Slack agent/app for chat, report Q&A, report comparison, and workflow notifications.
- Web app for document upload, demo chat, admin review, metrics, and evaluation dashboards.

The MVP helps users upload medical PDFs or text files, ask questions, compare reports, and receive evidence-grounded answers with citations. It is designed for medical research, education, and document interpretation, not autonomous diagnosis.

The product has two answer modes:
- **Document-grounded mode**: answers from uploaded reports and cites uploaded passages.
- **General medical education mode**: answers from a small trusted education corpus and cites the education passage. It is non-diagnostic and does not replace clinician advice.

## 2. MVP Goal
Ship a usable product in one week that demonstrates:
- agentic planning,
- LangGraph-based workflow orchestration,
- retrieval-augmented generation,
- MCP-based tool use,
- evidence citations,
- Slack agent experience with a supporting web dashboard,
- Slack hackathon alignment through MCP integration and agentic workflow automation,
- no GPU cost (NVIDIA free endpoints),
- synthetic medical datasets.

## 3. Target Users
- Medical students.
- Researchers.
- Clinicians reviewing documents.
- Health-tech demo evaluators.
- AI/ML recruiter reviewers.
- Slack workspace users who need cited answers from trusted medical/research documents.

## 4. Core Use Case
User uploads one or more medical documents and asks:
- What are the key findings?
- What changed between reports?
- Explain this in simple language.
- What evidence supports this statement?
- Explain pneumonia, anemia, hypertension, diabetes, or chest pain in simple educational language.

The system retrieves relevant passages, reasons over them, and returns a cited response.

In the Slack experience, users can ask CareMind questions in a channel or direct message, trigger report comparison, receive citation-backed summaries, and get safety-aware educational responses without leaving Slack.

## 5. Datasets

### 5.1 For Development & Demo
Use **synthetic or open datasets** to avoid PHI and compliance issues:

- **PDF Deid Dataset** (JohnSnowLabs/pdf-deid-dataset)
  - Fully synthetic medical-style PDF documents.
  - Easy, Medium, Hard levels.
  - Ideal for OCR, de-identification testing, and document ingestion.
  - No real patient data. [web:168]

- **Synthetic Australian Medical Documents Sample** (RootCauseAnalytics/synthetic-australian-medical-documents-sample)
  - 50-document sample of synthetic NSW Health-style PDFs.
  - 29 document types.
  - PHI-free, CC-BY-NC 4.0.
  - Includes scanned variants for OCR robustness. [web:170]

- **Medical Lab Report Dataset** (Kaggle)
  - Medical report images.
  - Good for OCR, NLP, and extraction testing. [web:172][web:181]

- **Medical RAG Corpus** (Sagarika-Singh-99/medical-rag-corpus on Hugging Face)
  - 216,102 medical document samples.
  - Preprocessed for RAG pipelines.
  - Includes BM25 tokens and dense embeddings. [web:157]

- **PubMed Corpus** (MedRAG/pubmed on Hugging Face)
  - 23.9M PubMed snippets (titles + abstracts).
  - Ready for medical RAG. [web:160]

### 5.2 For MVP Demo
For the MVP, you should:
- Use PDF Deid or Synthetic Australian documents as your **primary demo dataset**.
- Add a few classic medical Q&A datasets (e.g., MedQA, PubMedQA) for evaluation.
- Avoid real PHI in demos.

## 6. Models

### 6.1 LLM (NVIDIA Free Endpoint)
Use NVIDIA’s free NIM endpoints at `https://integrate.api.nvidia.com/v1`.

Recommended models:
- **`meta/llama-3.1-8b-instruct`** — Llama-family NVIDIA-hosted option, good for a recruiter-friendly demo.
- **`deepseek-ai/deepseek-v4-flash`** — strong for most RAG scenarios. [web:174]
- **`nvidia/nemotron-3-ultra-500b`** — extreme long-context, agentic reasoning. [web:174]
- **`nvidia/nemotron-4-340b-instruct`** — good general instruction-following.
- **`meta/llama-3.1-8b-instruct`** or **`mistralai/mistral-7b-instruct`** — smaller, fast options.

For MVP, use:
```python
model = "meta/llama-3.1-8b-instruct"
base_url = "https://integrate.api.nvidia.com/v1"
```

### 6.2 Embedding Model
Use NVIDIA embedding models via their API:

- **`nvolveqa_40k`** — GPU-accelerated question-answer retrieval embedding. [web:167][web:176]
- **`NV-EmbedQA-E5-v5`** — optimized for text QA retrieval. [web:179]
- **`NV-Embed-v2`** — generalist embedding, ranks No. 1 on MTEB. [web:182]

For MVP, use:
```python
model = "nvolveqa_40k"
```

Implementation note:
- The current MVP calls NVIDIA's OpenAI-compatible HTTP endpoints directly through `httpx`.
- LangGraph is used for agent workflow orchestration.
- LangChain wrappers are optional and can be added later if the project needs a larger integration ecosystem.

### 6.3 Optional Reranker
Optional:
- Use NVIDIA’s rerank models if available.
- Or use a simple similarity-based rerank in your pipeline.

## 7. Scope

### In Scope
- Web chat interface.
- Slack agent/app interface for direct messages, app mentions, document Q&A, and report comparison.
- Web admin/demo interface for upload, seeded demos, metrics, and evaluation dashboards.
- PDF and text upload.
- Synthetic medical PDFs (PDF Deid, Synthetic Australian).
- Chunking and embedding with NVIDIA.
- Vector retrieval with Pinecone.
- Agentic query routing.
- MCP-style tools for document search, report comparison, timeline extraction, and medical education search.
- Citation-backed answers.
- Conversation history and session state with Redis.
- Redis response cache for repeated retrieval, comparison, and education questions.
- Basic document-level memory.
- Basic evaluation and monitoring endpoints.
- Performance evaluation dashboard for offline eval runs and quality trends.
- Simple safety checks.
- NVIDIA free endpoint for LLM and embeddings.

### Out of Scope
- Full PHI compliance program.
- EHR/FHIR integration.
- DICOM support.
- Multi-agent orchestration.
- Voice.
- Mobile app.
- VS Code extension as a required MVP surface.
- Fine-tuning.
- Real-time clinical decision automation.
- Complex user roles.
- Paid GPU or enterprise inference.

## 8. Functional Requirements

### FR-1 Authentication
The product shall support basic login for private workspace access (MVP can use simple auth or no auth with local sessions).

### FR-2 Document Upload
Users shall upload PDF and text files through the web app. Slack users may attach documents or reference already-uploaded demo documents when Slack file ingestion is enabled.

### FR-3 Document Processing
The system shall extract text, split it into chunks, and index it for retrieval.

### FR-4 Agentic Query Routing
The assistant shall decide whether to:
- answer directly,
- retrieve from documents,
- call an MCP tool,
- answer from the medical education corpus,
- ask a clarification question.

### FR-5 Retrieval
The system shall retrieve top relevant chunks from uploaded documents before generating answers when needed. Vector search is done via Pinecone with NVIDIA embeddings.

Local development may use SQLite-backed local vector search. Production/demo-with-keys uses Pinecone by setting:
```env
CAREMIND_VECTOR_BACKEND=pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=caremind-index
```

### FR-6 MCP Tool Use
The system shall expose at least one MCP tool such as:
- report comparison,
- document search,
- timeline extraction.
- medical education search.

MCP tools are implemented as a separate server that the backend agent can call.

### FR-7 Answer Generation
The system shall generate concise, clinically grounded answers using retrieved evidence and the NVIDIA LLM endpoint.

### FR-8 Citations
Every factual answer shall include citations to document passages or retrieved evidence.

### FR-9 Memory
The system shall store:
- uploaded files,
- extracted document text,
- conversation history,
- basic document summaries.

Session and cache state are stored in Redis.

If Redis is unavailable, the MVP falls back to SQLite chat history and disables Redis response cache.

### FR-10 Safety Checks
The system shall:
- avoid claiming diagnosis,
- flag uncertainty,
- refuse unsupported medical advice,
- warn that the output is for educational use.

### FR-11 Evaluation
The system shall include a repeatable evaluation script that measures:
- route accuracy,
- citation pass rate,
- average latency,
- basic endpoint health.

### FR-12 Monitoring
The system shall expose a metrics endpoint containing:
- request counts,
- agent route counts,
- tool call counts,
- cache hit/miss counts,
- average and max latency by route/path.

### FR-13 Slack Agent Interface
The system shall expose a Slack app/agent that supports:
- direct messages and app mentions,
- asking questions over uploaded or seeded documents,
- report comparison requests,
- general medical education questions,
- citation-backed responses in Slack messages,
- safety notes and non-diagnostic disclaimers,
- optional buttons or shortcuts for seed demo, compare reports, and open dashboard.

### FR-14 Hackathon Submission Readiness
The MVP shall be prepared for the Slack Agent Builder Challenge by demonstrating at least one qualifying Slack technology path:
- MCP server integration for document search, report comparison, timeline extraction, and medical education tools.
- Slack agent workflow automation for question routing, retrieval, and response generation.
- Optional Real-Time Search API integration only if time permits.

The target hackathon track is **Slack Agent for Good**, with **New Slack Agent** as the backup track.

## 9. User Flows

### Flow A: Ask a Question
1. User uploads a medical PDF (e.g., PDF Deid).
2. User asks a question.
3. Agent decides retrieval is needed.
4. Retrieval finds relevant chunks in Pinecone.
5. LLM (via NVIDIA endpoint) answers with citations.
6. Response is shown in Slack and the web chat.
7. Session and cache are stored in Redis.

### Flow B: Compare Reports
1. User uploads two reports.
2. User asks what changed.
3. MCP tool or comparison logic identifies differences.
4. Agent summarizes changes with citations.

### Flow C: Continue Across Slack and Web
1. User starts in Slack by asking CareMind a question in a direct message or app mention.
2. User opens the web dashboard to upload documents, inspect citations, or view evaluation metrics.
3. Same workspace and history are available via shared backend and Redis.
4. User continues the conversation in Slack or the web app.

### Flow D: General Medical Education
1. User asks a general education question such as "What is hypertension?"
2. Agent routes to `medical_education`.
3. Medical education search retrieves trusted built-in education passages.
4. LLM produces a concise educational answer with citations.
5. Safety layer adds the educational/non-diagnostic disclaimer.

### Flow E: Cached Repeat Question
1. User asks a previously answered retrieval, comparison, or education question.
2. Agent checks Redis response cache.
3. If a cache hit exists, the cached answer is returned quickly.
4. Metrics records cache hit and route latency.

### Flow F: Slack Hackathon Demo
1. User opens the CareMind Slack agent in a sandbox workspace.
2. User asks CareMind to seed synthetic medical reports or references a preloaded demo workspace.
3. User asks "What are the key findings?"
4. CareMind routes the question, retrieves evidence, calls MCP-style document search, and posts a cited answer.
5. User asks "What changed between reports?"
6. CareMind calls the report comparison tool and posts a cited summary.
7. User opens the web dashboard from Slack to show metrics, route distribution, cache state, and evaluation results.

## 10. Architecture

### Frontend
- Web app with chat and upload (Next.js or React).
- Slack app/agent for workspace chat, direct messages, app mentions, and workflow actions.
- Web dashboard for document upload, demo chat, metrics, and evaluation results.

### Backend
- FastAPI service for chat, upload, retrieval, and memory.
- Slack adapter service for events, commands, interactive actions, and response formatting.
- LangGraph agent planner (agentic RAG).
- RAG pipeline using Pinecone + NVIDIA embeddings.
- MCP server for tools.
- Safety layer for grounding and policy checks.
- Metrics layer for route/tool/cache/request observability.
- Evaluation script for repeatable smoke-quality checks.

### Storage & Infrastructure
- **Pinecone**: vector database for document embeddings and retrieval.
- **Redis**: cache, session state, and conversation memory.
- **PostgreSQL/SQLite**: metadata, user data, and chat history.
- **Object storage**: local file storage or S3 for uploaded documents.

### Models
- **LLM**: NVIDIA free endpoint (e.g., `deepseek-ai/deepseek-v4-flash`).
- **Embedding**: NVIDIA endpoint (`nvolveqa_40k` or `NV-EmbedQA-E5-v5`).
- **Optional reranker**: NVIDIA rerank or simple similarity.

## 11. Tech Stack

### Frontend (Web)
- Next.js or React.
- Tailwind CSS for UI.
- Fetch/axios for API calls.
- Streaming chat UI.
- Admin/demo dashboard for documents, metrics, and evaluation reports.

### Slack App / Agent
- Slack app for direct messages, app mentions, commands, and interactive buttons.
- Slack Bolt for Python or Slack SDK integration with the FastAPI backend.
- Slack message formatting with citations, safety notes, and dashboard links.
- Slack developer sandbox for hackathon judging and demo access.

### Backend
- FastAPI (Python).
- LangGraph `StateGraph` for agent routing and workflow orchestration.
- Pydantic for data validation.
- uvicorn for server.

### Agent Graph
- LangGraph workflow:
  - route,
  - check cache,
  - direct response,
  - document retrieval,
  - report comparison,
  - medical education retrieval,
  - clarification,
  - safety/finalization.

### RAG & Retrieval
- Custom RAG pipeline using NVIDIA embeddings, Pinecone, and local SQLite fallback.
- Chunking: simple recursive-ish text splitter.
- NVIDIA embeddings via direct HTTP calls to the NVIDIA endpoint.

### Vector DB
- **Pinecone**:
  - Cloud-hosted.
  - Index for document chunks.
  - Metadata filtering by document ID.

### Cache & Session
- **Redis**:
  - Conversation sessions.
  - Query cache.
  - Temporary agent state.

### Database
- **SQLite** (MVP) or **PostgreSQL**.
- Stores:
  - user data,
  - workspace metadata,
  - chat history,
  - document metadata.

### MCP
- Python or TypeScript MCP server.
- Tools:
  - document search,
  - report comparison,
  - timeline extraction.
  - medical education search.

### LLM Inference
- NVIDIA free endpoint:
  - base_url: `https://integrate.api.nvidia.com/v1`
  - models:
    - `deepseek-ai/deepseek-v4-flash`
    - `nvidia/nemotron-4-340b-instruct`
    - `meta/llama-3.1-8b-instruct`

### Embeddings
- NVIDIA endpoint:
  - `nvolveqa_40k` or `NV-EmbedQA-E5-v5` via direct HTTP calls.
- Local fallback:
  - deterministic hashing embeddings for keyless demos.

### Deployment
- Docker for containerization.
- Optional:
  - Render,
  - Hugging Face Spaces with Docker,
  - Cloudflare,
  - Fly.io,
  - Azure Container Apps,
  - or simple VM.
- Slack app configured with production callback URLs and environment secrets.

### Evaluation & Testing
- `python evaluate.py` runs a local seeded evaluation.
- Metrics:
  - route accuracy,
  - citation pass rate,
  - average latency,
  - endpoint success/failure.
- Smoke checks:
  - `/health`,
  - `/demo/seed`,
  - `/chat`,
  - `/compare`,
  - `/metrics`.

### Monitoring
- `/metrics` exposes JSON metrics for MVP demos.
- Production can scrape or forward these metrics to Grafana/Prometheus, hosted logging platforms, or cloud provider monitoring.
- Minimum monitored signals:
  - request count by status,
  - latency by endpoint,
  - route distribution,
  - tool-call distribution,
  - Redis cache hit/miss,
  - citation pass rate from scheduled evaluation.

## 12. Non-Functional Requirements
- MVP should run locally or on a simple cloud host.
- Responses should stream where possible.
- Indexing should complete in under a minute for small documents.
- The UI should be simple and responsive.
- The system should be easy to demo.
- The stack should avoid GPU costs during prototyping.

## 13. Security Baseline
- Use HTTPS in deployment.
- Encrypt stored files if possible.
- Keep dev and prod data separate.
- Do not store unnecessary identifiers.
- Do not log sensitive content.
- Use synthetic or de-identified medical documents for the MVP demo.
- Basic auth for workspace access.
- Session and cache state in Redis with sensible TTLs.
- Never commit `.env` or API keys.
- Rotate any key that is accidentally shared.

## 14. Success Criteria
The MVP is successful if:
- a user can upload a document,
- ask a question,
- get a cited answer,
- compare two reports,
- use CareMind from Slack through direct messages or app mentions,
- continue the same workspace across Slack and the web dashboard,
- demonstrate MCP-based tool use,
- meet Slack hackathon requirements through MCP integration or Slack agent workflow automation,
- use NVIDIA free endpoints for inference and embeddings,
- demonstrate a general medical education route,
- show Redis cache state when Redis is configured,
- show Pinecone vector retrieval when `CAREMIND_VECTOR_BACKEND=pinecone`,
- run evaluation with route accuracy and citation pass-rate metrics,
- view runtime metrics from `/metrics`,
- view performance evaluation results in the dashboard.

## 15. Demo Script
Demo should show:
1. Open the CareMind Slack agent in the hackathon sandbox workspace.
2. Seed or upload synthetic medical reports through the web app.
3. In Slack, ask "What are the key findings?"
4. Show a cited answer posted back into Slack.
5. In Slack, ask "What changed between reports?"
6. Show the report comparison response and cite the source documents.
7. Show one MCP-style tool call in the backend logs, metrics, or dashboard.
8. Ask "What is hypertension?" to show general medical education RAG.
9. Open the web dashboard from Slack to show route, latency, cache, and tool metrics.
10. Run `python evaluate.py` and show the evaluation dashboard output.
11. Mention that the LLM and embeddings use NVIDIA free endpoints.

## 16. Future Extensions
- PHI-aware secure mode.
- FHIR integration.
- DICOM support.
- VS Code extension for developer/research workflows.
- Longitudinal patient timelines.
- Medical term simplification.
- Lab trend graphs.
- Multi-document reasoning.
- Audit logging.
- SSO and enterprise auth.
- Paid inference or private LLM deployment.

## 17. Similar Projects
Similar projects exist in parts:
- Slack AI agents and workflow bots.
- Slack Marketplace apps for knowledge search and team automation.
- VS Code RAG extensions (e.g. Knowledge RAG, StackRAG).
- MCP-enabled VS Code agents.
- Medical agentic RAG systems (e.g., MED-COPILOT-style work).
- Agentic RAG chatbots over PDFs and docs.

The competitive edge is combining:
- Slack + web dashboard,
- agentic RAG,
- MCP tool use,
- citation-backed answers,
- a medical/research domain,
- performance evaluation visibility for demo and regression tracking.

## 18. Is It Worth Building
Yes, it is worth building for AI/ML engineer jobs because it shows:
- agentic orchestration,
- RAG engineering,
- tool use (MCP),
- Slack app + frontend + backend integration,
- product thinking,
- deployment on a cloud stack without GPU costs.

## 19. One-Week Reality Check
This is feasible in one week only if you stay strict about scope:
- one document workflow,
- one primary Slack chat experience,
- one supporting web dashboard,
- one comparison tool,
- one shared backend,
- no compliance-heavy features.

With NVIDIA free endpoints, Pinecone, and Redis, the stack is clean and cost-effective for prototyping.

## 20. End-to-End Pipeline

### 20.1 Local Keyless Demo
1. Install dependencies with `pip install -r requirements.txt`.
2. Start the API with `python app.py`.
3. Open `http://127.0.0.1:8000`.
4. Upload sample reports or click seed demo.
5. Ask document questions and compare reports.
6. Ask a general education question such as "What is pneumonia?"
7. Inspect `/metrics` and the evaluation dashboard.
8. Run `python evaluate.py`.

Optional local Slack demo:
1. Start the API with a public tunnel URL.
2. Configure the Slack app event, command, and interactivity URLs to point at the tunnel.
3. Install the Slack app into a developer sandbox workspace.
4. Ask CareMind questions in Slack while the backend handles retrieval and tool calls.

### 20.2 API-Key Production Demo
Configure:
```env
NVIDIA_API_KEY=...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_CHAT_MODEL=meta/llama-3.1-8b-instruct
NVIDIA_EMBEDDING_MODEL=nvolveqa_40k

PINECONE_API_KEY=...
PINECONE_INDEX_NAME=caremind-index
CAREMIND_VECTOR_BACKEND=pinecone

REDIS_URL=redis://localhost:6379/0
CAREMIND_USERNAME=...
CAREMIND_PASSWORD=...

SLACK_BOT_TOKEN=...
SLACK_SIGNING_SECRET=...
SLACK_APP_TOKEN=...
CAREMIND_PUBLIC_BASE_URL=https://your-caremind-demo.example.com
```

Pipeline:
1. Upload document.
2. Extract text from PDF/text.
3. Chunk text.
4. Embed chunks using NVIDIA embeddings.
5. Upsert vectors into Pinecone with workspace/document metadata.
6. Store metadata and chat history in SQLite/PostgreSQL.
7. Cache session and repeat responses in Redis.
8. Agent routes each query to direct, document RAG, report comparison, medical education, or clarification.
9. LLM generates grounded answer.
10. Safety layer adds uncertainty and education disclaimer.
11. API returns answer, route, tool calls, citations, and safety notes.
12. Slack adapter formats the response for direct messages, app mentions, or interactive actions.
13. Metrics records latency, tool calls, route counts, request counts, and cache hit/miss.
14. Evaluation output is written to JSON/CSV and rendered in the dashboard.

### 20.3 Deployment
Recommended MVP deployment:
- Dockerized FastAPI app.
- Managed Redis.
- Pinecone serverless index.
- NVIDIA NIM endpoint.
- SQLite for demo or PostgreSQL for shared/stable deployment.
- Local disk for demo uploads or S3-compatible object storage for cloud uploads.
- HTTPS via platform proxy.
- Basic auth enabled for demo privacy.
- Slack app installed in a developer sandbox workspace.
- Public callback URL configured for Slack events, commands, and interactivity.

Commands:
```bash
docker compose up --build
python evaluate.py
```

### 20.4 Evaluation Metrics
Core MVP metrics:
- **Route accuracy**: predicted route equals expected route.
- **Citation pass rate**: factual answers include at least one citation.
- **Average latency**: mean end-to-end `/chat` response time.
- **Tool-call coverage**: document search, comparison, and education tools are exercised.
- **Cache hit rate**: Redis hits divided by total cacheable questions.
- **Endpoint health**: `/health` and `/metrics` return 200.

Future quality metrics:
- retrieval recall@k,
- answer faithfulness,
- citation precision,
- hallucination rate,
- safety refusal accuracy,
- user task success rate.

## 21. Performance Evaluation Dashboard

Build a CareMind performance evaluation dashboard to track quality over time across test sets, routes, and Slack/web interaction surfaces.

### 21.1 Dashboard Goals
- Track RAG accuracy, citation precision, retrieval latency, route accuracy, and cache hit rate.
- Compare metrics across synthetic test sets, document types, query types, and model configurations.
- Surface regressions after prompt, chunking, embedding, retrieval, Slack response formatting, or safety changes.
- Support both offline eval runs and sampled live-traffic checks.
- Provide a demo-friendly view for hackathon judges showing quality, grounding, and operational maturity.

### 21.2 Core Metrics
The dashboard shall include at least:
- RAG accuracy: end-to-end answer correctness against labeled references.
- Citation precision: fraction of cited claims that are supported by the cited passage.
- Citation completeness: fraction of factual claims with citations.
- Retrieval latency: time spent in vector search and reranking.
- End-to-end latency: total `/chat` or Slack request response time.
- Route accuracy: predicted route versus expected route.
- Retrieval precision@k and recall@k for gold passages.
- Faithfulness: answer claims supported by retrieved context.
- Slack interaction success rate: percentage of Slack requests that receive a successful response.

### 21.3 Views
- Trend lines for each metric by date.
- Breakdown by test set, question type, document type, route, and surface (`web` or `slack`).
- Top failing examples with gold answer, retrieved chunks, model answer, and citations.
- Comparison view for model, prompt, embedding, and Slack formatting experiments.
- Threshold indicators showing pass, warn, and fail states.
- Hackathon summary panel showing route accuracy, citation pass rate, latency, cache hit rate, and tool-call coverage.

### 21.4 Evaluation Pipeline
- `python evaluate.py` shall generate a repeatable offline report.
- The report shall be written to JSON or CSV and exposed through the dashboard.
- Scheduled runs may compare current results with the previous baseline.
- The dashboard shall make regressions visible without manual log inspection.
- Slack demo interactions may be sampled into anonymized evaluation logs when using only synthetic or de-identified data.

### 21.5 Deployment Surface
- Expose the dashboard in the web app as an admin or demo page.
- Link to the dashboard from Slack messages or shortcuts when appropriate.
- Keep the implementation simple enough for the one-week MVP scope.
- Do not expose private documents, full Slack messages, or user identifiers in public dashboard views.

## 22. Slack Hackathon Submission Plan

CareMind can be submitted to the Slack Agent Builder Challenge as a **Slack Agent for Good** project, with **New Slack Agent** as a backup track. The public Devpost page lists a deadline of July 13, 2026 at 5:00 PM PDT.

### 22.1 Submission Positioning
- Project name: CareMind Slack Agent.
- One-line pitch: A Slack agent that helps teams ask citation-backed questions over synthetic or de-identified medical documents.
- Impact angle: Safer medical education, research review, and document understanding inside the collaboration tool teams already use.
- Technical angle: Slack agent workflow + LangGraph RAG backend + MCP-style document tools + evaluation dashboard.

### 22.2 Hackathon Requirements Mapping
- Primary qualifying technology: MCP server integration for document search, report comparison, timeline extraction, and medical education tools.
- Slack agent/app workflow: Slack is the primary interaction surface for the working project.
- Optional qualifying technology: Slack AI capabilities or Real-Time Search API may be added only if they fit the final implementation without bloating scope.
- Working project footage: demo video shows Slack Q&A, report comparison, citations, and dashboard metrics.
- Architecture diagram: show Slack app, FastAPI adapter, LangGraph agent, MCP tools, Pinecone/SQLite, Redis, NVIDIA endpoints, and dashboard.
- Sandbox URL/access: provide Slack developer sandbox access for judging.

### 22.3 MVP Build Order
1. Ship web upload, seeded demo, chat, comparison, metrics, and eval dashboard.
2. Add Slack app authentication, event handling, and message formatting.
3. Route Slack messages into the existing `/chat` and `/compare` backend flows.
4. Add Slack shortcuts/buttons for seed demo, compare reports, and open dashboard.
5. Record the demo video and include architecture diagram plus evaluation results.
