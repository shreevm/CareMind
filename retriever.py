import pinecone
import torch
from transformers import AutoTokenizer, AutoModel

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")
index = pinecone.Index("mimic-index")

biobert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = biobert(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().tolist()

def retrieve_context(query):
    query_vec = get_embedding(query)
    result = index.query(query_vec, top_k=3, include_metadata=True)
    contexts = [m["metadata"]["text"] for m in result["matches"]]
    return "\n\n".join(contexts)
