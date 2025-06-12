# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# nli_model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
# nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
# nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

# def model_based_guardrail(premise, hypothesis):
#     inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
#     logits = nli_model(**inputs).logits
#     prediction = torch.argmax(logits, dim=1).item()
#     return prediction == 0  # entailment



# def apply_guardrails(context, answer, valid_icd_codes):
#     if not model_based_guardrail(context, answer):
#         return "[ Guardrail] Rejected: does not align with retrieved context."
#     if not rule_based_guardrail(answer, valid_icd_codes):
#         return "[Guardrail] ICD mismatch for Type 1 diabetes."
#     return answer


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

nli_model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

def run_nli_guardrail(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction == 0  # entailment