from dataloader import load_all_mimic_datasets,prepare_patient_documents
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
import uuid
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()


def build_vectorstore(index_name="caremind-index"):
    data = load_all_mimic_datasets()
    docs = prepare_patient_documents(data)
    batch_size = 32

    print(f"[✓] Loaded {len(docs)} documents from MIMIC-IV dataset.")
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
    )
    
    index = pc.Index(index_name)
    model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    for i in tqdm(range(0, len(docs), batch_size)):
        batch = docs[i:i + batch_size]
        texts = [doc["text"] for doc in batch]
        embeddings = model.encode(texts).tolist()
        ids = [doc["subject_id"] for doc in batch]

        metadata = []
        for doc in batch:
            subject_id = doc["subject_id"]
            text = doc["text"]

            # Extract ICU stay and medications from the text
            icu_stay = next((line for line in text.splitlines() if line.startswith("ICU stay")), "")
            Diagnosis = next((line for line in text.splitlines() if line.startswith("Medications:")), "")
            procedures = next((line for line in text.splitlines() if line.startswith("Procedures:")), "")
            # Optional: trim or clean if needed
            metadata.append({
                "subject_id": subject_id,
                "icu_stay": icu_stay,
                "Diagnosis": Diagnosis[:1000],
                "procedures": procedures[:1000]
              
            })

        vectors = list(zip(ids, embeddings, metadata))
        index.upsert(vectors)


    print(" Vectorstore built and uploaded to Pinecone.")

if __name__ == "__main__":
    build_vectorstore()


