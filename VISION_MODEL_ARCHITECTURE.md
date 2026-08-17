# CareMind Vision Model Architecture

## Decision

CareMind uses two independent vision paths:

```text
Document pipeline
  -> DocumentUnderstandingAgent
  -> configured Groq/OpenAI-compatible document vision model when configured
  -> OCR/local fallback when unavailable, failed, or low-confidence
  -> structured JSON
  -> document_understanding
  -> structured evidence embeddings
  -> message_attachments when uploaded through chat
  -> DocumentRAGAgent

Radiology pipeline
  -> ImagingAgent
  -> dedicated medical vision model such as MedGemma only when configured
  -> linked report/evidence fallback or explicit limitation
  -> cited imaging answer
```

## Document Understanding

`DocumentUnderstandingAgent` runs only during upload. It does not answer user questions.

Primary use cases:

- Medical PDFs
- Lab and blood reports
- ECG reports as documents
- Discharge summaries
- Prescriptions
- Referral letters
- Mobile phone photos of reports
- Scanned clinical documents

Responsibilities:

- Call the configured document vision model when `CAREMIND_DOCUMENT_VLM_BASE_URL` is configured. Groq can be used through its OpenAI-compatible endpoint.
- Fall back to conservative local extraction and OCR-derived text when the model is unavailable, fails, or returns confidence below `CAREMIND_DOCUMENT_VLM_MIN_CONFIDENCE`.
- Extract structured JSON: document type, summary, patient entities, lab values, diagnoses, medications, measurements, sections, warnings, and confidence.
- Store raw JSON in `document_understanding`.
- Create a conversation-scoped `message_attachments` row and activate conversation memory when upload includes `session_id`.
- Generate embeddings from structured evidence.

Normal question answering does not call the document vision model again. It uses stored structured evidence and, when `attachment_ids` are present, exact message-attachment lookup before vector retrieval.

## Radiology

Chest X-ray, CT, MRI, and other radiology interpretation should remain separate from document ingestion.

Future radiology support belongs behind `ImagingAgent` with settings such as:

```env
CAREMIND_RADIOLOGY_VISION_BASE_URL=
CAREMIND_RADIOLOGY_VISION_MODEL=google/medgemma-4b-it
```

This keeps report parsing, OCR, and table/form extraction independent from pixel-level medical-image reasoning.

## Why Modular

A single vision model for documents and radiology would mix two different jobs:

- Documents need layout, OCR-like reading, table understanding, forms, and JSON extraction.
- Radiology needs modality-specific image interpretation, anatomy-aware uncertainty, and stronger safety controls.

The split improves:

- Latency: document vision runs once at upload; chat uses stored evidence.
- Cost: ordinary reports do not pay radiology-model cost.
- Maintainability: each model has one responsibility.
- Extensibility: the document vision provider can be upgraded without touching ImagingAgent, and MedGemma can be added without changing document ingestion.
- Production safety: answers can be audited back to stored JSON, source spans, model version, and confidence.
