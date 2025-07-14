

from transformers import pipeline

def apply_guardrails(answer, context):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(answer, candidate_labels=["accurate", "hallucinated"])
    print(f"Guardrail evaluation: {result}")
    if result["labels"][0] == "accurate" and result["scores"][0] > 0.7:
        return answer
    return answer + " This response may be inaccurate. Please consult a clinician."
