import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

_INDEX_NAME = "caremind-index"

# Module-level init — loaded once at startup
_embed_model = SentenceTransformer(
    'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb',
    token=os.getenv("HF_TOKEN"),
)

_pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
try:
    _index = _pc.Index(_INDEX_NAME)
    print(f"[ok] Pinecone index '{_INDEX_NAME}' connected")
except Exception as e:
    _index = None
    print(f"[!] Pinecone index '{_INDEX_NAME}' unavailable: {e}")


def retrieve(query: str, top_k: int = 5) -> list[str]:
    if _index is None:
        return []

    query_vec = _embed_model.encode(query).tolist()
    results = _index.query(vector=query_vec, top_k=top_k, include_metadata=True)

    docs = []
    for match in results.matches:
        meta = match["metadata"]
        summary = meta.get("summary", "")
        if not summary:
            parts = []
            if meta.get("subject_id"):
                parts.append(f"Subject {meta['subject_id']}")
            if meta.get("Diagnosis"):
                parts.append(meta["Diagnosis"])
            if meta.get("icu_stay"):
                parts.append(meta["icu_stay"])
            if meta.get("procedures"):
                parts.append(meta["procedures"])
            summary = "\n".join(parts)
        if summary:
            docs.append(summary)

    return docs
