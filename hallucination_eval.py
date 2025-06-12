# hallucination_eval.py placeholder

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load NLI model for model-based guardrail
nli_model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

def model_based_guardrail(premise, hypothesis):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    logits = nli_model(**inputs).logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction == 0  # 0 = entailment

def rule_based_guardrail(answer, valid_icd_codes):
    if "Type 1 diabetes" in answer and "E10" not in valid_icd_codes:
        return False
    return True

def run_guardrails(answer, context, valid_icd_codes):
    # Apply model-based
    if not model_based_guardrail(context, answer):
        return "[⚠️ Guardrail] Answer rejected: does not match retrieved context."

    # Apply rule-based
    if not rule_based_guardrail(answer, valid_icd_codes):
        return "[⚠️ Guardrail] ICD mismatch detected for condition mentioned."

    return answer  # passed both checks
