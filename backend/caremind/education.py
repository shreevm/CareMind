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
        "id": "education:tuberculosis",
        "title": "General Education - Tuberculosis",
        "text": (
            "Tuberculosis is an infection caused by Mycobacterium tuberculosis. Pulmonary tuberculosis can cause "
            "a persistent cough, coughing up blood or sputum, chest pain, fever, night sweats, chills, loss of "
            "appetite, weight loss, fatigue, and weakness. Symptoms can develop gradually, and some people with "
            "latent tuberculosis infection have no symptoms. Evaluation may include clinical assessment, chest "
            "imaging, sputum testing, molecular tests, culture, and tuberculosis infection testing. Care decisions "
            "and public health steps should be handled by qualified clinicians and local health authorities."
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
        "id": "education:iron-deficiency-nutrition",
        "title": "General Education - Iron Deficiency Nutrition",
        "text": (
            "For iron deficiency nutrition education, iron-rich foods include lean red meat, poultry, fish, eggs, "
            "beans, lentils, chickpeas, tofu, spinach and other dark leafy greens, nuts, seeds, dried fruits, and "
            "iron-fortified cereals or grains. Vitamin C foods such as citrus, berries, tomatoes, bell peppers, "
            "and broccoli can improve absorption of non-heme iron from plant foods. Tea, coffee, calcium "
            "supplements, and high-calcium foods can reduce iron absorption when taken at the same time, so timing "
            "may matter. Dietary advice should be individualized by a clinician, especially for children, pregnancy, "
            "heavy menstrual bleeding, gastrointestinal symptoms, chronic disease, or when iron supplements are being "
            "considered."
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
        "id": "education:medical-report-follow-up",
        "title": "General Education - Medical Report Follow-Up",
        "text": (
            "After receiving a medical test report, general follow-up commonly includes reviewing the report with "
            "the clinician who ordered the test so findings can be interpreted together with symptoms, medical "
            "history, medications, and prior results. Patients can prepare by noting symptoms such as palpitations, "
            "dizziness, fainting, chest discomfort, shortness of breath, or symptom timing. Urgent medical attention "
            "is appropriate for severe chest pain, fainting, major breathing difficulty, blue lips, severe weakness, "
            "or rapidly worsening symptoms. This is general education and does not replace individualized clinician "
            "instructions."
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
    {
        "id": "education:nursing-postoperative-monitoring",
        "title": "Nursing Education - Postoperative Monitoring",
        "text": (
            "Postoperative nursing monitoring commonly includes scheduled vital signs, pain and sedation assessment, "
            "airway and breathing checks, wound or dressing inspection, intake and output tracking, mobility and fall-risk "
            "assessment, medication reconciliation, and escalation for fever, hypotension, oxygen desaturation, uncontrolled "
            "pain, bleeding, confusion, or other concerning changes. Protocols vary by procedure, anesthesia, unit policy, "
            "and clinician orders."
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
