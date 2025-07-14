
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# model_name = "meta-llama/Llama-2-7b-chat-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# def format_prompt(context, query):
#     return f"""You are a helpful medical assistant. Answer only based on the following context. If you are unsure, say 'I don't know.'

# Context:
# {context}

# Query: {query}
# Answer:"""

# def generate_answer(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     output = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9)
#     return tokenizer.decode(output[0], skip_special_tokens=True).split("Answer:")[-1].strip()



from transformers import pipeline

def answer_question(question, context_list):
    context = "\n".join(context_list[:3])  # Top 3 retrieved patient summaries

    prompt = f"""You are a clinical assistant. Based on the patient's medical summary below, answer the question.
    
    --- PATIENT SUMMARY ---
    {context}
    
    --- QUESTION ---
    {question}
    
    --- ANSWER ---
    """

    # Use a generative LLM
    generator = pipeline("text-generation", model="google/flan-t5-xl", tokenizer="google/flan-t5-xl")
    result = generator(prompt, max_length=512, do_sample=True)
    return result[0]["generated_text"].split("--- ANSWER ---")[-1].strip()
