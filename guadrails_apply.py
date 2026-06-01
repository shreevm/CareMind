from transformers import pipeline

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _classifier


def apply_guardrails(answer: str, _context=None) -> str:
    result = _get_classifier()(answer, candidate_labels=["accurate", "hallucinated"])
    if result["labels"][0] == "accurate" and result["scores"][0] > 0.7:
        return answer
    return answer + "\n\n[!] This response may be inaccurate. Please consult a clinician."
