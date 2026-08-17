from __future__ import annotations

from io import BytesIO
import math
from typing import Any

from .config import Settings
from .schemas import TranscriptMetadata


class SpeechTranscriptionError(RuntimeError):
    pass


class SpeechToTextService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def transcribe(self, audio: bytes, *, started_at=None, ended_at=None) -> TranscriptMetadata:
        if not self.settings.elevenlabs_api_key:
            raise SpeechTranscriptionError("ELEVENLABS_API_KEY is not configured.")
        try:
            from elevenlabs.client import ElevenLabs
        except Exception as exc:
            raise SpeechTranscriptionError("The elevenlabs Python package is not installed.") from exc

        client = ElevenLabs(api_key=self.settings.elevenlabs_api_key)
        result = client.speech_to_text.convert(
            file=BytesIO(audio),
            model_id="scribe_v2",
            tag_audio_events=True,
            language_code=None,
            diarize=False,
            timestamps_granularity="word",
        )
        words = self._get(result, "words") or []
        timestamps = [self._timestamp(word) for word in words]
        detected_language = self._get(result, "language_code")
        language_confidence = self._safe_float(self._get(result, "language_probability"))
        confidence = self._transcript_confidence(words)
        text = str(self._get(result, "text") or "")
        return TranscriptMetadata(
            text=text,
            confidence=confidence,
            language=detected_language,
            detected_language=detected_language,
            language_confidence=language_confidence,
            timestamps=timestamps,
            modality="voice",
            input_modality="voice",
            started_at=started_at,
            ended_at=ended_at,
        )

    def _timestamp(self, word: Any) -> dict[str, Any]:
        return {
            "text": self._get(word, "text") or "",
            "start": self._safe_float(self._get(word, "start")),
            "end": self._safe_float(self._get(word, "end")),
            "type": self._get(word, "type") or "",
            "speaker_id": self._get(word, "speaker_id"),
            "logprob": self._safe_float(self._get(word, "logprob")),
        }

    def _transcript_confidence(self, words: list[Any]) -> float | None:
        probabilities = []
        for word in words:
            if self._get(word, "type") not in {None, "word"}:
                continue
            logprob = self._safe_float(self._get(word, "logprob"))
            if logprob is None:
                continue
            probabilities.append(max(0.0, min(1.0, math.exp(logprob))))
        if not probabilities:
            return None
        return round(sum(probabilities) / len(probabilities), 4)

    def _safe_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _get(self, value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)
