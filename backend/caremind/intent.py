import re

from .education import EDUCATION_CORPUS


DOCUMENT_SCOPE_TERMS = [
    "report",
    "document",
    "uploaded",
    "based on this",
    "based on the report",
    "according to",
    "in this report",
    "in the file",
    "my report",
    "my results",
    "my result",
    "my labs",
    "these results",
    "this result",
    "this x-ray",
    "this xray",
    "x-ray",
    "xray",
    "scan",
    "image",
    "the patient",
    "this patient",
    "patient earlier",
    "same patient",
]

GENERAL_TOPIC_PHRASES = [
    "not about the patient",
    "not about patient",
    "in general",
    "generally",
    "general question",
    "general am asking",
    "educational",
]

GENERAL_MEDICAL_QUESTION_STARTS = (
    "what is ",
    "what are ",
    "what does ",
    "explain ",
    "tell me about ",
    "how does ",
    "how do ",
    "how can ",
    "why does ",
    "why do ",
    "which ",
)

GENERAL_MEDICAL_SIGNALS = [
    "abnormal",
    "blood",
    "blood pressure",
    "breath",
    "breathing",
    "cbc",
    "cholesterol",
    "condition",
    "cough",
    "diagnosis",
    "diet",
    "disease",
    "deficiency",
    "food",
    "foods",
    "fever",
    "glucose",
    "heart",
    "heart rate",
    "hemoglobin",
    "infection",
    "inflammation",
    "iron",
    "lab",
    "labs",
    "lung",
    "medication",
    "medicine",
    "monitoring",
    "normal",
    "nutrition",
    "otc",
    "pain",
    "protocol",
    "resting heart rate",
    "rhythm",
    "supplement",
    "supplements",
    "surgery",
    "symptom",
    "symptoms",
    "tablet",
    "tablets",
    "test",
    "treatment",
]

GENERIC_SUBJECT_TERMS = ["person", "people", "someone", "he/she", "he or she", "they"]
CONTEXTUAL_PERSON_TERMS = [" him", " his ", " her ", " she ", " he ", " earlier", "same patient", "this patient"]

GENERAL_MEDICAL_PATTERNS = [
    re.compile(r"\b(?:symptoms?|signs?|causes?|treatments?|prevention|risk factors?|diagnosis|tests?)\s+(?:of|for|from|with|having)\b"),
    re.compile(r"\b(?:what|how|why)\s+(?:causes?|leads?\s+to|happens?\s+with)\b"),
    re.compile(r"\b(?:otc|over the counter|supplements?|tablets?|capsules?|medicines?)\b"),
    re.compile(r"\bdeficiency\s+(?:in|of|for|with)\s+[a-z][a-z\s-]{2,}\b"),
    re.compile(r"\b[a-z][a-z\s-]{2,}\s+(?:deficiency|excess|level|levels|content|count|reading|result|range)\b"),
]

LAB_OR_HEALTH_TERMS = [
    "blood",
    "cbc",
    "cholesterol",
    "content",
    "count",
    "glucose",
    "hemoglobin",
    "iron",
    "level",
    "levels",
    "mineral",
    "oxygen",
    "platelet",
    "potassium",
    "pressure",
    "rbc",
    "sodium",
    "thyroid",
    "vitamin",
    "wbc",
]

MEDICAL_WORD_PATTERN = re.compile(
    r"\b[a-z]+(?:itis|osis|emia|aemia|oma|pathy|uria|algia|ectomy|scopy|plasty|trophy|paresis)\b"
)


def has_document_scope(message: str) -> bool:
    lowered = message.lower()
    if any(term in lowered for term in DOCUMENT_SCOPE_TERMS):
        return True
    return bool(re.search(r"\b(mr|mrs|ms|miss|patient)\.?\s+[a-z][a-z]+(?:\s+[a-z][a-z]+)?", lowered))


def explicitly_general_topic(message: str) -> bool:
    lowered = message.lower()
    return any(phrase in lowered for phrase in GENERAL_TOPIC_PHRASES)


def has_contextual_patient_reference(message: str) -> bool:
    padded = f" {message.lower()} "
    return any(term in padded for term in CONTEXTUAL_PERSON_TERMS)


def asks_general_medical_not_document(message: str) -> bool:
    lowered = message.lower().strip()
    if explicitly_general_topic(lowered):
        return True
    if has_document_scope(lowered) or has_contextual_patient_reference(lowered):
        return False
    if _has_general_medical_pattern(lowered):
        return True
    if _contains_education_topic(lowered):
        return True
    if any(term in lowered for term in GENERAL_MEDICAL_SIGNALS):
        return True
    if MEDICAL_WORD_PATTERN.search(lowered):
        return True
    if lowered.startswith(GENERAL_MEDICAL_QUESTION_STARTS):
        return any(term in lowered for term in GENERIC_SUBJECT_TERMS)
    return False


def _has_general_medical_pattern(message: str) -> bool:
    normalized = _normalize_common_typos(message)
    if re.search(r"\b(?:low|less|reduced|decreased|high|elevated|increased|abnormal)\b", normalized):
        return any(term in normalized for term in LAB_OR_HEALTH_TERMS)
    return any(pattern.search(normalized) for pattern in GENERAL_MEDICAL_PATTERNS)


def _normalize_common_typos(message: str) -> str:
    replacements = {
        "sympotms": "symptoms",
        "symptms": "symptoms",
        "symtoms": "symptoms",
        "hvaing": "having",
        "hving": "having",
        "deficency": "deficiency",
        "defiecny": "deficiency",
        "deficieny": "deficiency",
        "difiecny": "deficiency",
        "rion": "iron",
    }
    normalized = message
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def _contains_education_topic(message: str) -> bool:
    return any(topic and topic in message for topic in _education_topics())


def _education_topics() -> set[str]:
    topics: set[str] = set()
    for item in EDUCATION_CORPUS:
        item_id = str(item.get("id", "")).removeprefix("education:").replace("-", " ").strip()
        title = str(item.get("title", "")).lower()
        title = re.sub(r"^(general|nursing)\s+education\s+-\s+", "", title).strip()
        for value in [item_id, title]:
            if value:
                topics.add(value)
                topics.update(part for part in value.split() if len(part) >= 5)
    return topics
