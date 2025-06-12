
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

def format_prompt(context, query):

    return f"""
You are a professional clinical AI assistant.

Your task is to answer **ONLY using the given clinical context** from electronic health records (EHR).
You are not allowed to fabricate information, guess, or use external knowledge.

### Rules:
1. If the context does **not** support an answer, say: "I don't know based on the provided data."
2. Maintain clinical accuracy. Do **not** generalize beyond what's stated.
3. Be concise. Use clinical terms (e.g., hypotension, AFib, INR).
4. Do not speculate. Always ground your response in context.
5. Think step-by-step before finalizing the answer.
---

### Clinical Context:
{context}

### Question:
{query}

### Answer:
"""

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9)
    return tokenizer.decode(output[0], skip_special_tokens=True).split("Answer:")[-1].strip()