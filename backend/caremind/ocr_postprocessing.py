import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import Settings
from .schemas import RetrievedChunk


@dataclass
class StructuredChunk:
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class OCRCleanupService:
    """Normalizes noisy OCR while preserving clinically useful row boundaries."""

    _COMMON_REPLACEMENTS = {
        "\r\n": "\n",
        "\r": "\n",
        "\u00a0": " ",
        "|": " ",
        " l0 ": " 10 ",
        " O.": " 0.",
        "mgldl": "mg/dL",
        "gldl": "g/dL",
        "mmolll": "mmol/L",
        "x 10": "x10",
    }

    def clean(self, text: str) -> dict[str, Any]:
        cleaned = text or ""
        for old, new in self._COMMON_REPLACEMENTS.items():
            cleaned = cleaned.replace(old, new)
        cleaned = re.sub(r"[^\S\n]+", " ", cleaned)
        cleaned = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", cleaned)
        cleaned = re.sub(r"(?i)\b(mg)\s*/?\s*(dl)\b", "mg/dL", cleaned)
        cleaned = re.sub(r"(?i)\b(g)\s*/?\s*(dl)\b", "g/dL", cleaned)
        cleaned = re.sub(r"(?i)\b(mmol)\s*/?\s*(l)\b", "mmol/L", cleaned)
        lines = self._dedupe_lines(self._merge_broken_lines(cleaned.splitlines()))
        quality_notes = []
        if len("\n".join(lines).strip()) < 30:
            quality_notes.append("OCR text is short; source image may be low quality, handwritten, rotated, or non-textual.")
        if self._looks_rotated_or_columnar(lines):
            quality_notes.append("OCR line structure appears columnar or rotated; verify against the original document.")
        return {
            "clean_text": "\n".join(lines).strip(),
            "quality_notes": quality_notes,
        }

    def _merge_broken_lines(self, lines: list[str]) -> list[str]:
        merged: list[str] = []
        for raw_line in lines:
            line = raw_line.strip(" -:\t")
            if not line:
                continue
            if merged and self._should_merge(merged[-1], line):
                merged[-1] = f"{merged[-1]} {line}".strip()
            else:
                merged.append(line)
        return merged

    def _should_merge(self, previous: str, current: str) -> bool:
        if re.search(r"[:;]$", previous):
            return False
        if re.match(r"^(patient|name|age|sex|gender|date|doctor|physician|hospital|diagnosis|impression)\b", current, re.I):
            return False
        if re.search(r"\d\s*(mg/dl|g/dl|mmol/l|%|bpm|mmhg)$", previous, re.I):
            return False
        if len(previous) < 34 and re.match(r"^[a-z,)]", current):
            return True
        return previous.endswith("-")

    def _dedupe_lines(self, lines: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for line in lines:
            key = re.sub(r"\W+", "", line.lower())
            if key and key in seen:
                continue
            seen.add(key)
            output.append(line)
        return output

    def _looks_rotated_or_columnar(self, lines: list[str]) -> bool:
        if len(lines) < 8:
            return False
        very_short = sum(1 for line in lines if len(line) <= 3)
        return very_short / max(len(lines), 1) > 0.35


class DocumentClassifier:
    def classify(self, text: str, filename: str, content_type: str) -> str:
        lowered = f"{filename} {content_type} {text}".lower()
        if any(term in lowered for term in ["hemoglobin", "platelet", "wbc", "rbc", "glucose", "creatinine", "cholesterol", "lab"]):
            return "lab_report"
        if any(term in lowered for term in ["ecg", "ekg", "sinus rhythm", "qt interval", "pr interval"]):
            return "ecg_report"
        if any(term in lowered for term in ["discharge summary", "admission date", "discharge diagnosis"]):
            return "discharge_summary"
        if any(term in lowered for term in ["rx", "tablet", "capsule", "prescription", "sig:"]):
            return "prescription"
        if any(term in lowered for term in ["referral", "referred to", "consultation requested"]):
            return "referral_letter"
        if any(term in lowered for term in ["x-ray", "xray", "radiograph", "impression", "findings"]):
            return "imaging_report_or_caption"
        if content_type == "application/pdf":
            return "medical_pdf"
        return "clinical_document"


class TableExtractor:
    _UNIT_PATTERN = r"mg/dl|g/dl|mmol/l|u/l|iu/l|x10\^?3/?ul|x10\^?9/l|k/ul|%|mmhg|bpm|ng/ml|pg/ml|meq/l|fl|pg"

    def extract(self, text: str) -> dict[str, Any]:
        tests: list[dict[str, Any]] = []
        tables: list[dict[str, Any]] = []
        rows = []
        for line in text.splitlines():
            parsed = self._parse_result_line(line)
            if parsed:
                tests.append(parsed)
                rows.append(parsed)
        if rows:
            tables.append({"title": "Laboratory Results", "rows": rows[:80], "confidence": 0.72})
        return {"tests": tests[:80], "tables": tables}

    def _parse_result_line(self, line: str) -> dict[str, Any] | None:
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            return None
        pattern = re.compile(
            rf"^([A-Za-z][A-Za-z0-9 /()#%.-]{{1,44}}?)\s*[:\-]?\s*([<>]?\d+(?:\.\d+)?)\s*({self._UNIT_PATTERN})?"
            rf"(?:\s+(?:ref(?:erence)?\.?\s*range|range|normal)\s*[:\-]?\s*([<>]?\d+(?:\.\d+)?\s*(?:-|to)\s*[<>]?\d+(?:\.\d+)?))?",
            re.I,
        )
        match = pattern.search(line)
        if not match:
            return None
        name, value, unit, reference_range = match.groups()
        name = re.sub(r"\s+", " ", name).strip(" -:().")
        if len(name) < 2 or name.lower() in {"age", "page", "date", "uhid", "id"}:
            return None
        if not unit and not self._looks_like_test_name(name):
            return None
        return {
            "name": name,
            "value": value,
            "unit": self._normalize_unit(unit),
            "reference_range": (reference_range or "").strip(),
            "flag": self._flag_from_line(line),
            "confidence": 0.78 if unit else 0.62,
        }

    def _normalize_unit(self, unit: str | None) -> str:
        if not unit:
            return ""
        normalized = unit.strip()
        replacements = {"mg/dl": "mg/dL", "g/dl": "g/dL", "mmol/l": "mmol/L", "u/l": "U/L", "iu/l": "IU/L"}
        return replacements.get(normalized.lower(), normalized)

    def _flag_from_line(self, line: str) -> str:
        match = re.search(r"\b(high|low|critical|abnormal|positive|negative|reactive|non-reactive)\b", line, re.I)
        return match.group(1).lower() if match else ""

    def _looks_like_test_name(self, name: str) -> bool:
        return any(
            term in name.lower()
            for term in [
                "hemoglobin",
                "wbc",
                "rbc",
                "platelet",
                "glucose",
                "creatinine",
                "cholesterol",
                "sodium",
                "potassium",
                "bilirubin",
                "tsh",
                "hba1c",
            ]
        )


class EntityExtractor:
    _DATE_PATTERN = r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"

    def extract(self, text: str, tests: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "patient": self._patient(text),
            "provider": self._provider(text),
            "hospital": self._hospital(text),
            "dates": self._dates(text),
            "measurements": self._measurements(text),
            "diagnoses": self._diagnoses(text),
            "medications": self._medications(text),
            "sections": self._sections(text),
            "keywords": self._keywords(text, tests),
            "field_confidence": self._field_confidence(text, tests),
        }

    def _patient(self, text: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        patterns = {
            "name": r"\b(?:patient(?: name)?|name)\s*[:\-]\s*([A-Z][A-Za-z .'-]{2,80}?)(?=\s+\b(?:age|sex|gender|dob|date|uhid|mrn|id|hemoglobin|wbc|rbc|platelet|glucose|medication|diagnosis)\b\s*[:\-]?|$)",
            "age": r"\bage\s*[:\-]?\s*(\d{1,3})\b",
            "gender": r"\b(?:sex|gender)\s*[:\-]?\s*(male|female|m|f|other)\b",
            "dob": rf"\b(?:dob|date of birth)\s*[:\-]?\s*{self._DATE_PATTERN}",
            "id": r"\b(?:uhid|mrn|patient id|id)\s*[:\-]?\s*([A-Za-z0-9/-]{3,40})\b",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.I)
            if match:
                fields[key] = match.group(1).strip()
        return fields

    def _provider(self, text: str) -> dict[str, str]:
        provider: dict[str, str] = {}
        doctor = re.search(r"\b(?:dr\.?|doctor|physician|consultant)\s*[:\-]?\s*([A-Z][A-Za-z .'-]{2,80})", text, re.I)
        if doctor:
            provider["physician_name"] = doctor.group(1).strip()
        specialty = re.search(r"\b(?:department|specialty)\s*[:\-]\s*([A-Za-z /-]{3,80})", text, re.I)
        if specialty:
            provider["specialty"] = specialty.group(1).strip()
        return provider

    def _hospital(self, text: str) -> dict[str, str]:
        for line in text.splitlines()[:8]:
            if re.search(r"\b(hospital|clinic|medical center|diagnostics|laborator(?:y|ies))\b", line, re.I):
                return {"name": line.strip()}
        return {}

    def _dates(self, text: str) -> dict[str, Any]:
        values = list(dict.fromkeys(match.group(1) for match in re.finditer(self._DATE_PATTERN, text)))
        labeled: dict[str, str] = {}
        for label, date in re.findall(rf"\b(report date|collection date|sample date|admission date|discharge date|date)\s*[:\-]?\s*{self._DATE_PATTERN}", text, re.I):
            labeled[label.lower().replace(" ", "_")] = date
        return {"all": values[:20], **labeled}

    def _measurements(self, text: str) -> list[dict[str, Any]]:
        rows = []
        for label, value, unit in re.findall(r"\b(bp|blood pressure|heart rate|hr|spo2|temperature|weight|height)\s*[:\-]?\s*([<>]?\d+(?:/\d+)?(?:\.\d+)?)\s*(mmhg|bpm|%|c|f|kg|lb|cm)?", text, re.I):
            rows.append({"name": label, "value": value, "unit": unit, "confidence": 0.74})
        return rows[:40]

    def _diagnoses(self, text: str) -> list[dict[str, Any]]:
        rows = []
        for marker in ["diagnosis", "diagnoses", "assessment", "impression", "clinical impression"]:
            match = re.search(rf"\b{marker}\b\s*[:\-]\s*(.+?)(?=\b(?:plan|medications|findings|history|date|recommendations)\b\s*[:\-]|$)", text, re.I)
            if match:
                rows.append({"text": match.group(1).strip(" .")[:500], "source_section": marker, "confidence": 0.68})
        return rows[:12]

    def _medications(self, text: str) -> list[dict[str, Any]]:
        rows = []
        pattern = re.compile(
            r"\b([A-Z][A-Za-z-]{2,40})\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)\b(?:\s+([A-Za-z0-9 /.-]{0,40}))?",
            re.I,
        )
        for name, dose, unit, frequency in pattern.findall(text):
            if name.lower() in {"hemoglobin", "glucose", "platelet", "creatinine"}:
                continue
            rows.append({"name": name, "dose": dose, "unit": unit, "frequency": frequency.strip(), "confidence": 0.66})
        return rows[:30]

    def _sections(self, text: str) -> list[dict[str, Any]]:
        rows = []
        for heading, body in re.findall(r"(?m)^([A-Z][A-Za-z /-]{2,40}):\s*(.+)$", text):
            rows.append({"heading": heading.strip(), "text": body.strip()[:900], "confidence": 0.64})
        return rows[:30]

    def _keywords(self, text: str, tests: list[dict[str, Any]]) -> list[str]:
        terms = {item["name"].lower() for item in tests if item.get("name")}
        for term in ["hemoglobin", "wbc", "platelet", "glucose", "creatinine", "diabetes", "hypertension", "anemia", "ecg", "sinus rhythm"]:
            if re.search(rf"\b{re.escape(term)}\b", text, re.I):
                terms.add(term)
        return sorted(terms)[:40]

    def _field_confidence(self, text: str, tests: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "patient": 0.78 if re.search(r"\b(patient|name|age|sex|gender|uhid|mrn)\b", text, re.I) else 0.2,
            "doctor": 0.72 if re.search(r"\b(dr|doctor|physician|consultant)\b", text, re.I) else 0.15,
            "hospital": 0.72 if re.search(r"\b(hospital|clinic|laborator|diagnostics)\b", text, re.I) else 0.2,
            "test_names": 0.78 if tests else 0.18,
            "measurements": 0.76 if tests else 0.25,
            "dates": 0.75 if re.search(self._DATE_PATTERN, text) else 0.2,
        }


class SummaryGenerator:
    def generate(self, payload: dict[str, Any]) -> str:
        lines = [f"Document Type: {payload.get('document_type', 'Clinical document').replace('_', ' ').title()}"]
        patient = payload.get("patient") or {}
        if patient:
            patient_bits = [patient.get("name", ""), f"Age {patient.get('age')}" if patient.get("age") else "", patient.get("gender", "")]
            lines.append(f"Patient: {'; '.join(bit for bit in patient_bits if bit)}")
        tests = payload.get("tests") or []
        if tests:
            rendered = ", ".join(
                f"{item.get('name')} {item.get('value')} {item.get('unit', '')}".strip()
                for item in tests[:8]
                if item.get("name")
            )
            lines.append(f"Tests: {rendered}")
        abnormal = [item for item in tests if item.get("flag") in {"high", "low", "critical", "abnormal", "positive", "reactive"}]
        if abnormal:
            lines.append("Abnormal Findings: " + ", ".join(item.get("name", "") for item in abnormal[:8] if item.get("name")))
        diagnoses = payload.get("diagnoses") or []
        if diagnoses:
            lines.append("Clinical Impression: " + "; ".join(str(item.get("text") if isinstance(item, dict) else item) for item in diagnoses[:3]))
        if len(lines) == 1 and payload.get("raw_text"):
            lines.append(str(payload["raw_text"])[:400])
        return "\n".join(lines)


class ChunkBuilder:
    def build(self, payload: dict[str, Any]) -> list[StructuredChunk]:
        chunks: list[StructuredChunk] = []
        self._append(chunks, "Document Summary", payload.get("summary"), payload, "summary")
        self._append(chunks, "Patient Information", payload.get("patient"), payload, "patient")
        self._append(chunks, "Hospital Information", payload.get("hospital"), payload, "hospital")
        self._append(chunks, "Physician Information", payload.get("provider"), payload, "provider")
        self._append(chunks, "Laboratory Results", payload.get("tests") or payload.get("lab_values"), payload, "laboratory_results")
        self._append(chunks, "Measurements", payload.get("measurements"), payload, "measurements")
        self._append(chunks, "Medications", payload.get("medications"), payload, "medications")
        self._append(chunks, "Diagnoses", payload.get("diagnoses"), payload, "diagnoses")
        self._append(chunks, "Clinical Notes", payload.get("sections"), payload, "clinical_notes")
        self._append(chunks, "Impression", self._impression_text(payload), payload, "impression")
        self._append(chunks, "Recommendations", self._section_text(payload, "recommend"), payload, "recommendations")
        if not chunks and payload.get("raw_text"):
            self._append(chunks, "Source Text", str(payload["raw_text"])[:1800], payload, "raw_text")
        return chunks

    def to_retrieved_chunks(
        self,
        structured_chunks: list[StructuredChunk],
        *,
        document_id: str,
        document_name: str,
        base_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        chunks = []
        for index, chunk in enumerate(structured_chunks):
            metadata = {**(base_metadata or {}), **chunk.metadata}
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"{document_id}:{index}",
                    document_id=document_id,
                    document_name=document_name,
                    text=chunk.text,
                    metadata=metadata,
                )
            )
        return chunks

    def _append(self, chunks: list[StructuredChunk], title: str, value: Any, payload: dict[str, Any], section_type: str) -> None:
        if value in (None, "", [], {}):
            return
        body = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        text = f"{title}\nDocument type: {payload.get('document_type', 'clinical_document')}\n{body}".strip()
        chunks.append(
            StructuredChunk(
                title=title,
                text=text,
                metadata={
                    "chunk_type": section_type,
                    "document_type": payload.get("document_type"),
                    "ingestion_method": payload.get("ingestion_method"),
                    "confidence": payload.get("confidence"),
                },
            )
        )

    def _impression_text(self, payload: dict[str, Any]) -> str:
        diagnoses = payload.get("diagnoses") or []
        for item in diagnoses:
            text = item.get("text") if isinstance(item, dict) else str(item)
            if text:
                return text
        return self._section_text(payload, "impression")

    def _section_text(self, payload: dict[str, Any], needle: str) -> str:
        for section in payload.get("sections") or []:
            heading = str(section.get("heading", "") if isinstance(section, dict) else "")
            if needle in heading.lower():
                return str(section.get("text", ""))
        return ""


class OCRPostProcessingPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cleanup = OCRCleanupService()
        self.classifier = DocumentClassifier()
        self.entities = EntityExtractor()
        self.tables = TableExtractor()
        self.summary = SummaryGenerator()
        self.chunks = ChunkBuilder()

    def process(
        self,
        *,
        text: str,
        filename: str,
        content_type: str,
        source_image_id: str | None = None,
        ocr_confidence: float | None = None,
        ingestion_method: str = "ocr",
    ) -> dict[str, Any]:
        cleanup = self.cleanup.clean(text)
        clean_text = cleanup["clean_text"]
        table_payload = self.tables.extract(clean_text)
        entity_payload = self.entities.extract(clean_text, table_payload["tests"])
        document_type = self.classifier.classify(clean_text, filename, content_type)
        confidence = self._overall_confidence(
            clean_text=clean_text,
            ocr_confidence=ocr_confidence,
            field_confidence=entity_payload["field_confidence"],
            test_count=len(table_payload["tests"]),
        )
        payload: dict[str, Any] = {
            "document_type": document_type,
            "confidence": confidence,
            "ingestion_method": ingestion_method,
            "patient": entity_payload["patient"],
            "provider": entity_payload["provider"],
            "hospital": entity_payload["hospital"],
            "dates": entity_payload["dates"],
            "tests": table_payload["tests"],
            "lab_values": table_payload["tests"],
            "measurements": entity_payload["measurements"],
            "diagnoses": entity_payload["diagnoses"],
            "medications": entity_payload["medications"],
            "sections": entity_payload["sections"],
            "tables": table_payload["tables"],
            "keywords": entity_payload["keywords"],
            "field_confidence": entity_payload["field_confidence"],
            "warnings": cleanup["quality_notes"],
            "raw_text": clean_text,
            "raw_text_excerpt": clean_text[:4000],
            "source_image_id": source_image_id,
        }
        payload["summary"] = self.summary.generate(payload)
        payload["semantic_chunks"] = [
            {"title": chunk.title, "text": chunk.text, "metadata": chunk.metadata}
            for chunk in self.chunks.build(payload)
        ]
        return payload

    def _overall_confidence(
        self,
        *,
        clean_text: str,
        ocr_confidence: float | None,
        field_confidence: dict[str, float],
        test_count: int,
    ) -> float:
        base = ocr_confidence if ocr_confidence is not None else 0.45
        signal = 0.0
        signal += 0.1 if len(clean_text) >= self.settings.image_ocr_min_chars else -0.12
        signal += 0.12 if test_count else 0.0
        signal += sum(field_confidence.values()) / max(len(field_confidence), 1) * 0.18
        return round(max(0.05, min(0.92, base * 0.62 + signal)), 3)
