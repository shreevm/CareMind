# CareMind — Project Specification

## Overview

CareMind is a Retrieval-Augmented Generation (RAG) clinical QA system that answers natural language questions about patient data from the MIMIC-IV Clinical Database Demo. It runs a local LLM, stores conversation sessions in Redis, and retrieves relevant patient summaries from a Pinecone vector index built with BioBERT embeddings.

---

## Architecture

```
User (Gradio Chat UI)
        │
        ▼
 gradio.py  ──── session history ────► Redis (caremind:session:<uuid>)
        │
        ▼
 retriever.py  ──── BioBERT embed query ──► Pinecone Index (caremind-index)
        │                                         │
        │                                   top-k patient summaries
        ▼
 gradio.py  ──── prompt + history + context ──► Local LLM (flan-t5-large)
        │
        ▼
 Streamed answer (TextIteratorStreamer)
        │
        ▼
 guadrails_apply.py  ──► Zero-shot NLI check (bart-large-mnli)
        │
        ▼
 Final response rendered in chat
```

### Offline / Index-build path

```
dataloader.py
  ├── download_and_extract_mimic_data()   — Kaggle API → local CSV
  └── prepare_patient_documents()         — one text doc per patient

build_vectorstore.py
  └── build_vectorstore()
        ├── BioBERT encode each doc
        └── upsert to Pinecone (caremind-index)
```

---

## Modules

| File | Role |
|---|---|
| [gradio.py](gradio.py) | Gradio chat UI, streaming, Redis session management |
| [retriever.py](retriever.py) | Pinecone similarity search using BioBERT |
| [dataloader.py](dataloader.py) | Load MIMIC-IV CSVs; build per-patient text documents |
| [build_vectorstore.py](build_vectorstore.py) | One-time index build: embed docs → upsert to Pinecone |
| [agentic_qa.py](agentic_qa.py) | LangChain agent with two tools: clinical search + guidelines |
| [llm_chain.py](llm_chain.py) | Llama-2-7b prompt formatter + generator (alternative LLM path) |
| [llm_qa.py](llm_qa.py) | Flan-T5-XL pipeline wrapper (alternative QA path) |
| [guadrails_apply.py](guadrails_apply.py) | NLI-based hallucination filter |
| [hallucination_eval.py](hallucination_eval.py) | Extended guardrail with NLI + ICD rule checks (stub) |

---

## Data Source

**MIMIC-IV Clinical Database Demo v2.2**

- 100 de-identified patients from Beth Israel Deaconess Medical Center
- No free-text clinical notes (excluded from demo)
- Stored at `data/mimic-iv-clinical-database-demo-2.2/`
- Kaggle mirror: `montassarba/mimic-iv-clinical-database-demo-2-2`

### Loaded Tables

| Key | Path |
|---|---|
| `patients` | `hosp/patients.csv` |
| `admissions` | `hosp/admissions.csv` |
| `diagnoses_icd` | `hosp/diagnoses_icd.csv` |
| `drgcodes` | `hosp/drgcodes.csv` |
| `labevents` | `hosp/labevents.csv` |
| `pharmacy` | `hosp/pharmacy.csv` |
| `prescriptions` | `hosp/prescriptions.csv` |
| `procedures_icd` | `hosp/procedures_icd.csv` |
| `transfers` | `hosp/transfers.csv` |
| `icustays` | `icu/icustays.csv` |
| `chartevents` | `icu/chartevents.csv` |
| `datetimeevents` | `icu/datetimeevents.csv` |
| `inputevents` | `icu/inputevents.csv` |
| `outputevents` | `icu/outputevents.csv` |
| `procedureevents` | `icu/procedureevents.csv` |

### Patient Document Schema

`prepare_patient_documents()` produces one document per `subject_id`:

```
Subject <id>
Summary:
Diagnoses: <ICD long titles, semicolon-separated>
ICU stay from <intime> to <outtime>, firstcare = <unit>
Prescriptions: <drug names, comma-separated>
Procedures: <ICD long titles, semicolon-separated>
Recent Labs: <itemid = value, top 3 most recent>
```

---

## Models

| Model | Used in | Purpose |
|---|---|---|
| `google/flan-t5-large` | `gradio.py` | Primary chat LLM (local, default) |
| `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` | `retriever.py`, `build_vectorstore.py` | Embedding for semantic search |
| `facebook/bart-large-mnli` | `guadrails_apply.py` | Zero-shot NLI guardrail |
| `meta-llama/Llama-2-7b-chat-hf` | `llm_chain.py` | Alternative causal LLM (requires HF access) |
| `google/flan-t5-xl` | `llm_qa.py` | Alternative larger Flan-T5 path |
| `deepset/roberta-base-squad2` | `app.py` | Extractive QA fallback |

Override the primary LLM via env var: `CAREMIND_MODEL=google/flan-t5-xl`

---

## Vector Store

- **Provider:** Pinecone (serverless)
- **Index name:** `caremind-index`
- **Embedding model:** BioBERT (768-dim)
- **Metadata stored per vector:**
  - `subject_id`
  - `icu_stay`
  - `Diagnosis` (truncated to 1000 chars)
  - `procedures` (truncated to 1000 chars)
- **Retrieval:** top-5 nearest neighbors; top-3 used in prompt

---

## Session Management

- **Store:** Redis at `localhost:6379` (db 0)
- **Key pattern:** `caremind:session:<uuid4>`
- **Value:** JSON array of `{"role": "user"|"assistant", "content": "..."}` messages
- **TTL:** 3600 seconds (1 hour)
- **Fallback:** in-memory dict if Redis is unreachable
- **Session ID:** generated per browser session via `gr.State`; reset on "Clear Chat"

---

## Prompt Structure

```
You are CareMind, a clinical decision-support assistant.
Answer using only the patient records provided.
If the answer is not in the records, say 'I don't know.'

--- PATIENT RECORDS ---
<top-3 retrieved summaries>

--- CONVERSATION HISTORY ---
User: <turn n-2>
CareMind: <turn n-1>

User: <current question>
CareMind:
```

- History window: last 4 messages (2 full turns)
- Input truncated to 512 tokens

---

## Guardrails

`guadrails_apply.py` — runs after every answer:

1. Zero-shot classification via `bart-large-mnli` on labels `["accurate", "hallucinated"]`
2. If top label is `"accurate"` with score > 0.7 → return answer as-is
3. Otherwise → append: *"This response may be inaccurate. Please consult a clinician."*

Extended guardrail in `hallucination_eval.py` (currently stubbed):
- NLI entailment check (context as premise, answer as hypothesis)
- ICD code rule check (e.g. "Type 1 diabetes" requires ICD E10)

---

## Agentic Path (agentic_qa.py)

LangChain zero-shot ReAct agent with two tools:

| Tool | Function | Description |
|---|---|---|
| `ClinicalRecordSearch` | `search_clinical_data()` | RAG over Pinecone for patient-specific answers |
| `MedicalGuidelines` | `medical_guideline_lookup()` | Keyword-matched stub for hypertension/diabetes guidelines |

LLM: OpenAI (requires `OPENAI_API_KEY`). This path is separate from the Gradio chat UI which uses the local Flan-T5.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `PINECONE_API_KEY` | Yes | — | Pinecone project API key |
| `OPENAI_API_KEY` | agentic path only | — | For `agentic_qa.py` |
| `HF_TOKEN` | gated models only | — | HuggingFace token (needed for Llama-2) |
| `CAREMIND_MODEL` | No | `google/flan-t5-large` | Override local LLM |
| `REDIS_HOST` | No | `localhost` | Redis hostname |
| `REDIS_PORT` | No | `6379` | Redis port |

Set via `.env` file (loaded with `python-dotenv`).

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env   # add PINECONE_API_KEY, HF_TOKEN, etc.
```

### 3. Build the vector index (one-time)
```bash
python build_vectorstore.py
```

### 4. Start Redis (optional but recommended)
```bash
redis-server
```

### 5. Launch the chat UI
```bash
python gradio.py
```

---

## Dependencies

```
torch
transformers
gradio
pinecone-client
sentence-transformers
langchain
langchain-community
huggingface_hub
scikit-learn
tqdm
python-dotenv
redis
kaggle
pandas
```
