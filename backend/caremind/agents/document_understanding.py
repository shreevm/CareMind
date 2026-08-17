import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..ocr import OCR_DERIVED_CONTENT_TYPE
from ..ocr_postprocessing import ChunkBuilder, OCRPostProcessingPipeline

logger = logging.getLogger(__name__)

DOCUMENT_UNDERSTANDING_SCHEMA_VERSION = "document-understanding/v1"


@dataclass
class DocumentUnderstandingResult:
    document_id: str
    workspace_id: str
    filename: str
    content_type: str
    document_type: str
    summary: str
    confidence: float
    model_name: str
    extraction_source: str
    raw_json: dict[str, Any]
    index_text: str
    warnings: list[str] = field(default_factory=list)
    semantic_chunks: list[dict[str, Any]] = field(default_factory=list)


class DocumentUnderstandingAgent:
    """Ingestion-time document parser. It never answers user questions."""

    def __init__(self, settings: Settings, llm: Any | None = None):
        self.settings = settings
        self.llm = llm
        self.ocr_pipeline = OCRPostProcessingPipeline(settings)
        self.chunk_builder = ChunkBuilder()

    def understand(
        self,
        *,
        document_id: str,
        workspace_id: str,
        filename: str,
        content_type: str,
        file_path: Path,
        extracted_text: str,
        source_image_id: str | None = None,
        modality: str = "document",
        ocr_confidence: float | None = None,
    ) -> DocumentUnderstandingResult:
        logger.info(
            "document_understanding.start document_id=%s filename=%s content_type=%s",
            document_id,
            filename,
            content_type,
        )
        payload = self._call_document_vlm(
            filename=filename,
            content_type=content_type,
            file_path=file_path,
            extracted_text=extracted_text,
            modality=modality,
        )
        fallback_reason = ""
        if payload and self._bounded_confidence(payload.get("confidence")) < self.settings.document_vlm_min_confidence:
            fallback_reason = "low_confidence"
            payload = None
        extraction_source = "document_vlm" if payload else self._fallback_source(source_image_id, content_type)
        if not payload:
            payload = self._structure_locally(
                text=extracted_text,
                filename=filename,
                content_type=content_type,
                source_image_id=source_image_id,
                modality=modality,
                ocr_confidence=ocr_confidence,
            )
            if fallback_reason:
                warnings = self._list(payload.get("warnings"))
                warnings.append(f"Vision extraction fallback reason: {fallback_reason}.")
                payload["warnings"] = warnings
        normalized = self._normalize_payload(
            payload=payload,
            document_id=document_id,
            workspace_id=workspace_id,
            filename=filename,
            content_type=content_type,
            source_image_id=source_image_id,
            modality=modality,
            extraction_source=extraction_source,
        )
        index_text = self.to_index_text(normalized)
        result = DocumentUnderstandingResult(
            document_id=document_id,
            workspace_id=workspace_id,
            filename=filename,
            content_type=content_type,
            document_type=str(normalized.get("document_type") or "clinical_document"),
            summary=str(normalized.get("summary") or self._summarize(extracted_text)),
            confidence=float(normalized.get("confidence") or 0.0),
            model_name=str(normalized.get("model_name") or self._model_name(extraction_source)),
            extraction_source=extraction_source,
            raw_json=normalized,
            index_text=index_text,
            warnings=list(normalized.get("warnings") or []),
            semantic_chunks=list(normalized.get("semantic_chunks") or []),
        )
        logger.info(
            "document_understanding.done document_id=%s source=%s document_type=%s confidence=%.2f index_chars=%s",
            document_id,
            result.extraction_source,
            result.document_type,
            result.confidence,
            len(result.index_text),
        )
        return result

    def to_index_text(self, payload: dict[str, Any]) -> str:
        sections = [
            "Structured clinical document evidence.",
            f"Document type: {payload.get('document_type', 'clinical_document')}.",
            f"Source filename: {payload.get('filename', 'unknown')}.",
        ]
        if payload.get("ingestion_method"):
            sections.append(f"Ingestion method: {payload['ingestion_method']}.")
        if payload.get("summary"):
            sections.append(f"Summary: {payload['summary']}")
        for key, label in [
            ("patient", "Patient"),
            ("provider", "Provider"),
            ("hospital", "Hospital"),
            ("diagnoses", "Diagnoses"),
            ("medications", "Medications"),
            ("tests", "Laboratory tests"),
            ("lab_values", "Laboratory values"),
            ("measurements", "Measurements"),
            ("clinical_entities", "Clinical entities"),
            ("dates", "Dates"),
            ("tables", "Tables"),
            ("sections", "Sections"),
            ("keywords", "Keywords"),
        ]:
            value = payload.get(key)
            if value:
                sections.append(f"{label}: {json.dumps(value, ensure_ascii=False)}")
        for chunk in payload.get("semantic_chunks") or []:
            if isinstance(chunk, dict) and chunk.get("title") and chunk.get("text"):
                sections.append(f"{chunk['title']}:\n{chunk['text']}")
        if payload.get("raw_text_excerpt"):
            sections.append(f"Source text excerpt:\n{payload['raw_text_excerpt']}")
        return "\n\n".join(sections).strip()

    def _call_document_vlm(
        self,
        *,
        filename: str,
        content_type: str,
        file_path: Path,
        extracted_text: str,
        modality: str,
    ) -> dict[str, Any] | None:
        if not self.settings.document_understanding_enabled or self.llm is None:
            return None
        structure_document = getattr(self.llm, "structure_document", None)
        if not callable(structure_document):
            return None
        try:
            payload = structure_document(
                text=extracted_text,
                filename=filename,
                content_type=content_type,
                file_path=file_path,
                modality=modality,
            )
        except Exception as exc:
            logger.warning("document_understanding.vlm_failed filename=%s error=%s", filename, exc.__class__.__name__)
            return None
        return payload if isinstance(payload, dict) else None

    def _normalize_payload(
        self,
        *,
        payload: dict[str, Any],
        document_id: str,
        workspace_id: str,
        filename: str,
        content_type: str,
        source_image_id: str | None,
        modality: str,
        extraction_source: str,
    ) -> dict[str, Any]:
        normalized = {
            "schema_version": DOCUMENT_UNDERSTANDING_SCHEMA_VERSION,
            "document_id": document_id,
            "workspace_id": workspace_id,
            "filename": filename,
            "content_type": content_type,
            "source_image_id": source_image_id,
            "modality": modality,
            "document_type": payload.get("document_type") or self._document_type("", filename, content_type),
            "ingestion_method": payload.get("ingestion_method") or self._ingestion_method(extraction_source, source_image_id, content_type),
            "summary": payload.get("summary") or payload.get("structured_summary") or "",
            "patient": self._dict(payload.get("patient")),
            "provider": self._dict(payload.get("provider")),
            "hospital": self._dict(payload.get("hospital")),
            "dates": self._list(payload.get("dates")),
            "diagnoses": self._list(payload.get("diagnoses")),
            "medications": self._list(payload.get("medications")),
            "tests": self._list(payload.get("tests") or payload.get("lab_values") or payload.get("laboratory_values")),
            "lab_values": self._list(payload.get("lab_values") or payload.get("tests") or payload.get("laboratory_values")),
            "measurements": self._list(payload.get("measurements")),
            "clinical_entities": self._list(payload.get("clinical_entities") or payload.get("entities")),
            "tables": self._list(payload.get("tables")),
            "sections": self._list(payload.get("sections")),
            "keywords": self._list(payload.get("keywords")),
            "confidence": self._bounded_confidence(payload.get("confidence")),
            "field_confidence": self._dict(payload.get("field_confidence")),
            "warnings": self._list(payload.get("warnings") or payload.get("quality_notes")),
            "model_name": payload.get("model_name") or self._model_name(extraction_source),
            "extraction_source": extraction_source,
            "raw_text": str(payload.get("raw_text") or payload.get("raw_text_excerpt") or "")[:12000],
            "raw_text_excerpt": str(payload.get("raw_text_excerpt") or payload.get("raw_text") or "")[:4000],
            "semantic_chunks": self._list(payload.get("semantic_chunks")),
        }
        normalized["dates"] = self._dates_payload(normalized["dates"])
        if not normalized["summary"]:
            normalized["summary"] = self._summary_from_payload(normalized)
        if not normalized["semantic_chunks"]:
            normalized["semantic_chunks"] = [
                {"title": chunk.title, "text": chunk.text, "metadata": chunk.metadata}
                for chunk in self.chunk_builder.build(normalized)
            ]
        return normalized

    def _structure_locally(
        self,
        *,
        text: str,
        filename: str,
        content_type: str,
        source_image_id: str | None,
        modality: str,
        ocr_confidence: float | None,
    ) -> dict[str, Any]:
        if source_image_id or content_type == OCR_DERIVED_CONTENT_TYPE:
            return self.ocr_pipeline.process(
                text=text,
                filename=filename,
                content_type=content_type,
                source_image_id=source_image_id,
                ocr_confidence=ocr_confidence,
                ingestion_method="ocr",
            )
        compact = re.sub(r"\s+", " ", text).strip()
        lab_values = self._lab_values(compact)
        medications = self._medications(compact)
        diagnoses = self._diagnoses(compact)
        measurements = self._measurements(compact)
        patient = self._patient_fields(compact)
        warnings: list[str] = []
        if len(compact) < self.settings.image_ocr_min_chars:
            warnings.append("Document text is short; extraction may be incomplete.")
        if source_image_id:
            warnings.append("Source is an uploaded image or scan; verify extracted values against the original.")
        confidence = 0.35
        confidence += 0.15 if compact else 0
        confidence += 0.1 if patient else 0
        confidence += 0.12 if lab_values else 0
        confidence += 0.08 if medications else 0
        confidence += 0.08 if diagnoses else 0
        confidence += 0.06 if measurements else 0
        return {
            "document_type": self._document_type(compact, filename, content_type),
            "ingestion_method": "text_extraction",
            "summary": self._summarize(compact),
            "patient": patient,
            "dates": self._dates(compact),
            "diagnoses": diagnoses,
            "medications": medications,
            "lab_values": lab_values,
            "measurements": measurements,
            "clinical_entities": self._clinical_entities(compact),
            "keywords": self._clinical_entities(compact),
            "sections": self._sections(text),
            "confidence": min(confidence, 0.82),
            "warnings": warnings,
            "raw_text": compact[:12000],
            "raw_text_excerpt": compact[:4000],
        }

    def _patient_fields(self, text: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        name = re.search(
            r"\b(?:patient|name)\s*[:\-]\s*([A-Z][A-Za-z .'-]{2,80}?)(?=\s+\b(?:age|sex|gender|date|dob|id|uhid|mrn|hemoglobin|wbc|rbc|platelet|glucose|medication|diagnosis)\b\s*[:\-]?|$)",
            text,
            re.I,
        )
        age = re.search(r"\bage\s*[:\-]?\s*(\d{1,3})\b", text, re.I)
        gender = re.search(r"\b(?:sex|gender)\s*[:\-]?\s*(male|female|m|f|other)\b", text, re.I)
        if name:
            fields["name"] = name.group(1).strip()
        if age:
            fields["age"] = age.group(1)
        if gender:
            fields["gender"] = gender.group(1)
        return fields

    def _lab_values(self, text: str) -> list[dict[str, str]]:
        units = r"mg/dl|g/dl|mmol/l|u/l|iu/l|x10\^?3/?ul|x10\^?9/l|k/ul|%|mmhg|bpm|ng/ml|pg/ml|meq/l"
        pattern = re.compile(
            rf"\b([A-Za-z][A-Za-z0-9 /()%-]{{1,42}}?)\s*[:\-]?\s*([<>]?\d+(?:\.\d+)?)\s*({units})?\b",
            re.I,
        )
        rows = []
        for name, value, unit in pattern.findall(text):
            clean_name = re.sub(r"\s+", " ", name).strip(" -:().")
            if len(clean_name) < 2 or clean_name.lower() in {"age", "page", "date"}:
                continue
            if unit or self._looks_like_lab_name(clean_name):
                rows.append({"name": clean_name, "value": value, "unit": unit, "source": "document_understanding"})
        return rows[:40]

    def _measurements(self, text: str) -> list[dict[str, str]]:
        rows = []
        for label, value, unit in re.findall(r"\b(bp|blood pressure|heart rate|hr|spo2|temperature|weight)\s*[:\-]?\s*([<>]?\d+(?:/\d+)?(?:\.\d+)?)\s*(mmhg|bpm|%|c|f|kg|lb)?", text, re.I):
            rows.append({"name": label, "value": value, "unit": unit})
        return rows[:30]

    def _medications(self, text: str) -> list[dict[str, str]]:
        rows = []
        pattern = re.compile(
            r"\b([A-Z][A-Za-z-]{2,40})\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)\b(?:\s+([A-Za-z0-9 /.-]{0,40}))?",
            re.I,
        )
        for name, dose, unit, frequency in pattern.findall(text):
            if name.lower() in {"hemoglobin", "glucose", "platelet", "patient"}:
                continue
            rows.append({"name": name, "dose": dose, "unit": unit, "frequency": frequency.strip()})
        return rows[:30]

    def _diagnoses(self, text: str) -> list[str]:
        diagnoses = []
        for marker in ["diagnosis", "diagnoses", "assessment", "impression", "clinical impression"]:
            match = re.search(rf"\b{marker}\b\s*[:\-]\s*(.+?)(?=\b(?:plan|medications|findings|history|date)\b\s*[:\-]|$)", text, re.I)
            if match:
                diagnoses.append(match.group(1).strip(" .")[:500])
        return diagnoses[:10]

    def _dates(self, text: str) -> list[str]:
        values = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", text)
        return list(dict.fromkeys(values))[:20]

    def _sections(self, text: str) -> list[dict[str, str]]:
        rows = []
        for heading, body in re.findall(r"(?m)^([A-Z][A-Za-z /-]{2,40}):\s*(.+)$", text):
            rows.append({"heading": heading.strip(), "text": body.strip()[:500]})
        return rows[:20]

    def _clinical_entities(self, text: str) -> list[str]:
        terms = []
        for term in ["hemoglobin", "wbc", "platelet", "glucose", "creatinine", "diabetes", "hypertension", "anemia", "ecg", "sinus rhythm"]:
            if re.search(rf"\b{re.escape(term)}\b", text, re.I):
                terms.append(term)
        return terms

    def _document_type(self, text: str, filename: str, content_type: str) -> str:
        lowered = f"{filename} {content_type} {text}".lower()
        if any(term in lowered for term in ["hemoglobin", "platelet", "wbc", "cbc", "cholesterol", "glucose", "creatinine", "lab"]):
            return "lab_report"
        if any(term in lowered for term in ["ecg", "ekg", "sinus rhythm", "heart rate", "qt interval"]):
            return "ecg_report"
        if any(term in lowered for term in ["discharge summary", "admission", "discharge diagnosis"]):
            return "discharge_summary"
        if any(term in lowered for term in ["rx", "tablet", "capsule", "prescription"]):
            return "prescription"
        if any(term in lowered for term in ["referral", "referred to", "consultation requested"]):
            return "referral_letter"
        if content_type == "application/pdf":
            return "medical_pdf"
        return "clinical_document"

    def _looks_like_lab_name(self, name: str) -> bool:
        lowered = name.lower()
        return any(term in lowered for term in ["hemoglobin", "wbc", "rbc", "platelet", "glucose", "creatinine", "cholesterol", "sodium", "potassium"])

    def _summarize(self, text: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        return compact[:500] if compact else "No confident document text was extracted."

    def _summary_from_payload(self, payload: dict[str, Any]) -> str:
        parts = [f"{payload.get('document_type', 'Clinical document')} parsed"]
        if payload.get("patient", {}).get("name"):
            parts.append(f"for {payload['patient']['name']}")
        counts = []
        for key, label in [("lab_values", "lab values"), ("medications", "medications"), ("diagnoses", "diagnoses")]:
            if payload.get(key):
                counts.append(f"{len(payload[key])} {label}")
        if counts:
            parts.append("with " + ", ".join(counts))
        return " ".join(parts) + "."

    def _bounded_confidence(self, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(parsed, 1.0))

    def _dict(self, value: Any) -> dict:
        return value if isinstance(value, dict) else {}

    def _list(self, value: Any) -> list:
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return value
        if value is None or value == "":
            return []
        return [value]

    def _model_name(self, extraction_source: str) -> str:
        if extraction_source == "document_vlm":
            return self.settings.document_vlm_model
        if extraction_source == "ocr_postprocess":
            return "caremind-ocr-postprocessing"
        return "caremind-local-document-understanding"

    def _fallback_source(self, source_image_id: str | None, content_type: str) -> str:
        if source_image_id or content_type == OCR_DERIVED_CONTENT_TYPE:
            return "ocr_postprocess"
        return "local_fallback"

    def _ingestion_method(self, extraction_source: str, source_image_id: str | None, content_type: str) -> str:
        if extraction_source == "document_vlm":
            return "vision_model"
        if source_image_id or content_type == OCR_DERIVED_CONTENT_TYPE:
            return "ocr"
        return "text_extraction"

    def _dates_payload(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {"all": value}
        if value:
            return {"all": [value]}
        return {"all": []}
