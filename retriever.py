
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

def retrieve(query, top_k=5, index_name="caremind-index"):
    pc = Pinecone(
       api_key="pcsk_3SN2pp_7EaQhfSqyargU43MWYcqnTPN7eLMQF1NzNZgayBkBDvVofohuuGECew7KJecxzK"
   )
    index = pc.Index(index_name)
    model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    query_vec = model.encode(query).tolist()
    results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    docs = [match["metadata"].get("summary", "") for match in results["matches"]]
    return docs  


