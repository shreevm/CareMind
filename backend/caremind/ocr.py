import json
import logging
import re
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)

OCR_DERIVED_CONTENT_TYPE = "application/x-ocr-json"


@dataclass
class OCRResult:
    text: str = ""
    engine: str = "none"
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)


class ImageOCRService:
    """Best-effort OCR for report screenshots/scans without making image upload brittle."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def extract(self, image_path: Path) -> OCRResult:
        if not self.settings.image_ocr_enabled:
            return OCRResult(warnings=["Image OCR is disabled."])
        engine = self.settings.image_ocr_engine.lower().strip()
        engines = ["pytesseract", "easyocr"] if engine == "auto" else [engine]
        if engine == "auto" and self.settings.nvidia_api_key:
            engines = ["nvidia", *engines]
        warnings: list[str] = []
        for candidate in engines:
            if candidate == "nvidia":
                result = self._extract_with_nvidia(image_path)
            elif candidate == "pytesseract":
                result = self._extract_with_pytesseract(image_path)
            elif candidate == "easyocr":
                result = self._extract_with_easyocr(image_path)
            else:
                result = OCRResult(warnings=[f"Unsupported OCR engine '{candidate}'."])
            if result.text.strip():
                return result
            warnings.extend(result.warnings)
        return OCRResult(warnings=warnings or ["No OCR engine produced text."])

    def _extract_with_nvidia(self, image_path: Path) -> OCRResult:
        if not self.settings.nvidia_api_key:
            return OCRResult(engine="nvidia", warnings=["NVIDIA OCR unavailable: NVIDIA_API_KEY is not configured."])
        try:
            import httpx
        except Exception as exc:
            return OCRResult(engine="nvidia", warnings=[f"NVIDIA OCR unavailable: httpx import failed ({exc.__class__.__name__})."])
        try:
            image_bytes = image_path.read_bytes()
            image_format = self._nvidia_image_format(image_path)
            payload = {
                "input": [
                    {
                        "type": "image_url",
                        "url": f"data:image/{image_format};base64,{base64.b64encode(image_bytes).decode('ascii')}",
                    }
                ],
                "merge_levels": ["word"],
            }
            headers = {
                "Authorization": f"Bearer {self.settings.nvidia_api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            with httpx.Client(timeout=self.settings.nvidia_ocr_timeout_seconds) as client:
                response = client.post(self.settings.nvidia_ocr_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return self._parse_nvidia_response(data)
        except Exception as exc:
            logger.warning("ocr.nvidia.failed path=%s error=%s", image_path, exc.__class__.__name__)
            return OCRResult(engine="nvidia", warnings=[f"NVIDIA OCR failed: {exc.__class__.__name__}"])

    def _parse_nvidia_response(self, data: dict[str, Any]) -> OCRResult:
        detections: list[dict[str, Any]] = []
        for item in data.get("data") or []:
            if isinstance(item, dict):
                detections.extend([row for row in item.get("text_detections") or [] if isinstance(row, dict)])
        pieces: list[str] = []
        confidences: list[float] = []
        for detection in detections:
            prediction = detection.get("text_prediction")
            if not isinstance(prediction, dict):
                continue
            text = str(prediction.get("text") or "").strip()
            if text:
                pieces.append(text)
            confidence = prediction.get("confidence")
            try:
                confidences.append(float(confidence))
            except (TypeError, ValueError):
                pass
        return OCRResult(
            text=self._clean_ocr_text(" ".join(pieces)),
            engine=self.settings.nvidia_ocr_model,
            confidence=(sum(confidences) / len(confidences)) if confidences else None,
        )

    def _nvidia_image_format(self, image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            return "jpeg"
        return "png"

    def _extract_with_pytesseract(self, image_path: Path) -> OCRResult:
        try:
            from PIL import Image, ImageEnhance, ImageFilter, ImageOps
            import pytesseract
        except Exception as exc:
            return OCRResult(engine="pytesseract", warnings=[f"pytesseract unavailable: {exc.__class__.__name__}"])
        try:
            with Image.open(image_path) as original:
                variants = self._preprocess_variants(original.convert("RGB"), ImageOps, ImageEnhance, ImageFilter)
                best_text = ""
                best_score = -1
                best_confidence: float | None = None
                for variant in variants:
                    text = pytesseract.image_to_string(variant, config="--psm 6")
                    cleaned = self._clean_ocr_text(text)
                    score = self._quality_score(cleaned)
                    if score > best_score:
                        best_text = cleaned
                        best_score = score
                        best_confidence = self._pytesseract_confidence(pytesseract, variant)
                return OCRResult(text=best_text, engine="pytesseract", confidence=best_confidence)
        except Exception as exc:
            logger.warning("ocr.pytesseract.failed path=%s error=%s", image_path, exc.__class__.__name__)
            return OCRResult(engine="pytesseract", warnings=[f"pytesseract failed: {exc.__class__.__name__}"])

    def _extract_with_easyocr(self, image_path: Path) -> OCRResult:
        try:
            import easyocr
        except Exception as exc:
            return OCRResult(engine="easyocr", warnings=[f"easyocr unavailable: {exc.__class__.__name__}"])
        try:
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            rows = reader.readtext(str(image_path), detail=1, paragraph=True)
            texts = []
            confidences = []
            for row in rows:
                if len(row) >= 2:
                    texts.append(str(row[1]))
                if len(row) >= 3:
                    confidences.append(float(row[2]))
            return OCRResult(
                text=self._clean_ocr_text("\n".join(texts)),
                engine="easyocr",
                confidence=(sum(confidences) / len(confidences)) if confidences else None,
            )
        except Exception as exc:
            logger.warning("ocr.easyocr.failed path=%s error=%s", image_path, exc.__class__.__name__)
            return OCRResult(engine="easyocr", warnings=[f"easyocr failed: {exc.__class__.__name__}"])

    def _preprocess_variants(self, image, ImageOps, ImageEnhance, ImageFilter):
        grayscale = ImageOps.grayscale(image)
        autocontrast = ImageOps.autocontrast(grayscale)
        sharpened = autocontrast.filter(ImageFilter.SHARPEN)
        high_contrast = ImageEnhance.Contrast(sharpened).enhance(1.8)
        threshold = high_contrast.point(lambda pixel: 255 if pixel > 165 else 0)
        return [grayscale, autocontrast, high_contrast, threshold]

    def _pytesseract_confidence(self, pytesseract, image) -> float | None:
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 6")
            values = [float(value) for value in data.get("conf", []) if str(value).replace(".", "", 1).lstrip("-").isdigit() and float(value) >= 0]
            return (sum(values) / len(values)) / 100 if values else None
        except Exception:
            return None

    def _clean_ocr_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = []
        seen = set()
        for line in (item.strip(" |") for item in text.splitlines()):
            if not line:
                continue
            key = re.sub(r"\W+", "", line.lower())
            if key and key in seen:
                continue
            seen.add(key)
            lines.append(line)
        return "\n".join(lines).strip()

    def _quality_score(self, text: str) -> int:
        alnum = sum(1 for char in text if char.isalnum())
        medical_terms = len(re.findall(r"\b(patient|age|sex|gender|mg|dl|mmol|hemoglobin|wbc|rbc|platelet|impression|finding|diagnosis)\b", text, re.I))
        return alnum + 30 * medical_terms


class ClinicalTextStructurer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def structure(
        self,
        *,
        text: str,
        filename: str,
        modality: str,
        source: str,
        llm: Any | None = None,
    ) -> dict[str, Any]:
        model_payload = self._structure_with_model(text=text, filename=filename, modality=modality, llm=llm)
        fallback = self._structure_locally(text=text, filename=filename, modality=modality, source=source)
        if model_payload:
            return {**fallback, **model_payload, "raw_text": text}
        return fallback

    def to_index_text(self, payload: dict[str, Any]) -> str:
        sections = [
            "OCR-derived clinical document.",
            f"Source image: {payload.get('source_filename', 'unknown')}.",
            f"Modality: {payload.get('modality', 'clinical_image')}.",
            f"Document type: {payload.get('document_type', 'unknown')}.",
        ]
        if payload.get("patient"):
            sections.append(f"Patient: {json.dumps(payload['patient'], ensure_ascii=False)}.")
        if payload.get("measurements"):
            sections.append(f"Measurements: {json.dumps(payload['measurements'], ensure_ascii=False)}.")
        if payload.get("findings"):
            sections.append(f"Findings: {json.dumps(payload['findings'], ensure_ascii=False)}.")
        if payload.get("impression"):
            sections.append(f"Impression: {payload['impression']}.")
        if payload.get("quality_notes"):
            sections.append(f"OCR quality notes: {'; '.join(payload['quality_notes'])}.")
        if payload.get("raw_text"):
            sections.append(f"Raw OCR text:\n{payload['raw_text']}")
        return "\n\n".join(sections).strip()

    def _structure_with_model(self, *, text: str, filename: str, modality: str, llm: Any | None) -> dict[str, Any] | None:
        if not self.settings.image_ocr_structuring_enabled or llm is None:
            return None
        try:
            structured = llm.structure_ocr_text(text=text, filename=filename, modality=modality)
        except Exception as exc:
            logger.warning("ocr.structuring.model_failed filename=%s error=%s", filename, exc.__class__.__name__)
            return None
        return structured if isinstance(structured, dict) else None

    def _structure_locally(self, *, text: str, filename: str, modality: str, source: str) -> dict[str, Any]:
        compact = re.sub(r"\s+", " ", text).strip()
        patient = self._patient_fields(compact)
        measurements = self._measurements(compact)
        findings = self._findings(compact, measurements)
        quality_notes = []
        if len(compact) < self.settings.image_ocr_min_chars:
            quality_notes.append("OCR text is short; the source image may be noisy, handwritten, or non-textual.")
        if source != "user_report_text":
            quality_notes.append("OCR text may contain recognition errors; verify against the original image.")
        return {
            "source": source,
            "source_filename": filename,
            "modality": modality,
            "document_type": self._document_type(compact, filename),
            "patient": patient,
            "measurements": measurements,
            "findings": findings,
            "impression": self._impression(compact),
            "quality_notes": quality_notes,
            "raw_text": text,
        }

    def _patient_fields(self, text: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        name = re.search(r"\b(?:patient|name)\s*[:\-]\s*([A-Z][A-Za-z .'-]{2,80})", text, re.I)
        age = re.search(r"\b(?:age)\s*[:\-]?\s*(\d{1,3})\b", text, re.I)
        gender = re.search(r"\b(?:sex|gender)\s*[:\-]?\s*(male|female|m|f|other)\b", text, re.I)
        if name:
            fields["name"] = name.group(1).strip()
        if age:
            fields["age"] = age.group(1)
        if gender:
            fields["gender"] = gender.group(1)
        return fields

    def _measurements(self, text: str) -> list[dict[str, str]]:
        pattern = re.compile(
            r"\b([A-Za-z][A-Za-z0-9 /()%-]{1,40})\s*[:\-]?\s*([<>]?\d+(?:\.\d+)?)\s*"
            r"(mg/dl|g/dl|mmol/l|u/l|iu/l|x10\^?3/?ul|k/ul|%|mmhg|bpm|ng/ml|pg/ml)?\b",
            re.I,
        )
        rows = []
        for label, value, unit in pattern.findall(text):
            clean_label = re.sub(r"\s+", " ", label).strip(" -:")
            if len(clean_label) < 2 or clean_label.lower() in {"age", "page"}:
                continue
            rows.append({"name": clean_label, "value": value, "unit": unit})
        return rows[:30]

    def _findings(self, text: str, measurements: list[dict[str, str]]) -> list[str]:
        findings = []
        for marker in ["impression", "findings", "diagnosis", "conclusion", "remarks"]:
            match = re.search(rf"\b{marker}\b\s*[:\-]\s*(.+?)(?=\b(?:impression|findings|diagnosis|conclusion|remarks)\b\s*[:\-]|$)", text, re.I)
            if match:
                findings.append(match.group(1).strip()[:800])
        if measurements and not findings:
            findings.append(f"{len(measurements)} measurement values were extracted from OCR text.")
        return findings[:8]

    def _impression(self, text: str) -> str:
        match = re.search(r"\b(?:impression|conclusion)\b\s*[:\-]\s*(.+)$", text, re.I)
        return match.group(1).strip()[:900] if match else ""

    def _document_type(self, text: str, filename: str) -> str:
        lowered = f"{filename} {text}".lower()
        if any(term in lowered for term in ["hemoglobin", "platelet", "wbc", "cholesterol", "glucose", "lab"]):
            return "lab_report"
        if any(term in lowered for term in ["x-ray", "xray", "radiograph", "impression", "findings"]):
            return "imaging_report_or_caption"
        if any(term in lowered for term in ["rx", "tablet", "capsule", "prescription"]):
            return "prescription"
        return "ocr_text"
