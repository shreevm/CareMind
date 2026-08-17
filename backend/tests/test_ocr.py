from pathlib import Path

from backend.caremind.config import Settings
from backend.caremind.ocr import ImageOCRService, OCRResult


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "data": [
                {
                    "text_detections": [
                        {"text_prediction": {"text": "Glucose", "confidence": 0.91}},
                        {"text_prediction": {"text": "95", "confidence": 0.89}},
                        {"text_prediction": {"text": "mg/dL", "confidence": 0.87}},
                    ]
                }
            ],
            "usage": {"images_size_mb": 0.01},
        }


def test_nvidia_ocr_parses_text_and_confidence(tmp_path: Path, monkeypatch) -> None:
    calls = {}

    class FakeClient:
        def __init__(self, timeout):
            calls["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def post(self, url, headers, json):
            calls["url"] = url
            calls["authorization"] = headers.get("Authorization")
            calls["payload"] = json
            return _FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)
    path = tmp_path / "report.png"
    path.write_bytes(b"synthetic-image-bytes")
    settings = Settings(
        _env_file=None,
        NVIDIA_API_KEY="test-key",
        CAREMIND_IMAGE_OCR_ENGINE="nvidia",
        NVIDIA_OCR_ENDPOINT="https://example.test/ocr",
        NVIDIA_OCR_TIMEOUT_SECONDS=12,
    )

    result = ImageOCRService(settings).extract(path)

    assert result.engine == "nvidia/nemotron-ocr-v2"
    assert result.text == "Glucose 95 mg/dL"
    assert round(result.confidence or 0, 2) == 0.89
    assert calls["url"] == "https://example.test/ocr"
    assert calls["timeout"] == 12
    assert calls["authorization"] == "Bearer test-key"
    assert calls["payload"]["input"][0]["url"].startswith("data:image/png;base64,")
    assert calls["payload"]["merge_levels"] == ["word"]


def test_auto_ocr_can_fallback_after_nvidia_failure(tmp_path: Path) -> None:
    class FallbackOCR(ImageOCRService):
        def _extract_with_nvidia(self, image_path: Path) -> OCRResult:
            return OCRResult(engine="nvidia", warnings=["NVIDIA OCR failed: HTTPStatusError"])

        def _extract_with_pytesseract(self, image_path: Path) -> OCRResult:
            return OCRResult(text="Fallback text", engine="pytesseract", confidence=0.8)

    path = tmp_path / "report.png"
    path.write_bytes(b"synthetic-image-bytes")
    settings = Settings(
        _env_file=None,
        NVIDIA_API_KEY="test-key",
        CAREMIND_IMAGE_OCR_ENGINE="auto",
    )

    result = FallbackOCR(settings).extract(path)

    assert result.engine == "pytesseract"
    assert result.text == "Fallback text"
