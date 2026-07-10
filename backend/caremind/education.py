from .embeddings import EmbeddingClient
from .schemas import RetrievedChunk
from .vectorstore import cosine_similarity


EDUCATION_CORPUS = [
    {
        "id": "education:pneumonia",
        "title": "General Education - Pneumonia",
        "text": (
            "Pneumonia is an infection or inflammation of the lung tissue. Common symptoms include cough, "
            "fever, shortness of breath, fatigue, and chest discomfort. Evaluation may include vital signs, "
            "physical examination, pulse oximetry, chest imaging, and selected laboratory tests. Treatment "
            "depends on severity, risk factors, likely organism, allergies, and local clinical guidance."
        ),
    },
    {
        "id": "education:anemia",
        "title": "General Education - Anemia",
        "text": (
            "Anemia means the blood has a lower than expected capacity to carry oxygen, often reflected by low "
            "hemoglobin. Symptoms can include fatigue, weakness, shortness of breath, dizziness, or palpitations. "
            "Common evaluation includes complete blood count, iron studies, ferritin, vitamin B12, folate, kidney "
            "function, and assessment for blood loss when clinically appropriate."
        ),
    },
    {
        "id": "education:chest-pain",
        "title": "General Education - Chest Pain",
        "text": (
            "Chest pain can come from cardiac, lung, gastrointestinal, muscle, anxiety-related, or other causes. "
            "Potentially serious causes are assessed using symptoms, risk factors, ECG, cardiac biomarkers such as "
            "troponin, vital signs, and imaging when needed. Severe, persistent, or concerning chest pain requires "
            "urgent clinical assessment."
        ),
    },
    {
        "id": "education:diabetes",
        "title": "General Education - Diabetes",
        "text": (
            "Diabetes is a condition involving elevated blood glucose due to impaired insulin production, insulin "
            "action, or both. Monitoring can include glucose logs, hemoglobin A1c, kidney function, eye exams, foot "
            "checks, blood pressure, and cholesterol. Management is individualized and can include nutrition, activity, "
            "medications, and complication screening."
        ),
    },
    {
        "id": "education:hypertension",
        "title": "General Education - Hypertension",
        "text": (
            "Hypertension means blood pressure is persistently elevated. It is often monitored with repeated office "
            "or home readings. General management may include lifestyle changes, sodium reduction, physical activity, "
            "weight management, limiting alcohol, and medications selected by a clinician based on patient context."
        ),
    },
]


class MedicalEducationStore:
    def __init__(self, embeddings: EmbeddingClient):
        self.embeddings = embeddings
        self._vectors: list[tuple[dict, list[float]]] | None = None

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if self._vectors is None:
            vectors = self.embeddings.embed_texts([item["text"] for item in EDUCATION_CORPUS])
            self._vectors = list(zip(EDUCATION_CORPUS, vectors))
        query_vector = self.embeddings.embed_query(query)
        scored = []
        for item, vector in self._vectors:
            scored.append((cosine_similarity(query_vector, vector), item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        chunks = []
        for score, item in scored[:top_k]:
            chunks.append(
                RetrievedChunk(
                    chunk_id=item["id"],
                    document_id="general-medical-education",
                    document_name=item["title"],
                    text=item["text"],
                    score=score,
                )
            )
        return chunks
