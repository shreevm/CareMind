import re
from typing import Any

from .schemas import Citation, RetrievedChunk


_CITATION_RE = re.compile(r"\[((?:\s*\d+\s*,?\s*)+)\]")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:/[a-z0-9]+)?", re.IGNORECASE)
_NEGATION_RE = re.compile(r"\b(no|not|without|absent|negative|never|none|denies|denied)\b", re.IGNORECASE)
_DISCLAIMER_MARKERS = (
    "caremind is for medical education",
    "not a diagnosis or treatment plan",
    "consult a qualified",
)
_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s*)?(?:patient(?: profile)?|study|key (?:clinical )?findings|findings|"
    r"impression|limitations|uncertainty|evidence-grounded synthesis|summary)$",
    re.IGNORECASE,
)
_RAW_EVIDENCE_MARKERS = (
    "<evidence_set>",
    "<evidence id=",
    '"raw_text":',
    '"measurements":',
    '"findings":',
    "document id:",
    "chunk id:",
    "retrieval score:",
    "evidence audit terms:",
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "based", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "report", "reported", "says", "section", "that", "the",
    "this", "to", "was", "were", "with", "patient", "document", "evidence", "available", "according",
}
_QUESTION_STOPWORDS = _STOPWORDS | {
    "about", "does", "explain", "findings", "give", "key", "main", "me", "mention", "my", "please",
    "show", "summarize", "tell", "what", "which", "who", "why",
}


def extract_answer_claims(answer: str) -> list[dict[str, Any]]:
    """Split an answer into auditable claims and preserve bracketed citation ids."""
    claims: list[dict[str, Any]] = []
    for line in answer.splitlines():
        cleaned_line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not cleaned_line or cleaned_line.endswith(":") or _HEADING_RE.fullmatch(cleaned_line):
            continue
        if any(marker in cleaned_line.lower() for marker in _DISCLAIMER_MARKERS):
            continue
        for sentence in split_claim_sentences(cleaned_line):
            sentence = sentence.strip()
            plain = _CITATION_RE.sub("", sentence).strip()
            if len(_TOKEN_RE.findall(plain)) < 3:
                continue
            citation_ids: list[int] = []
            for match in _CITATION_RE.finditer(sentence):
                citation_ids.extend(int(value) for value in re.findall(r"\d+", match.group(1)))
            claims.append(
                {
                    "claim_index": len(claims) + 1,
                    "text": plain,
                    "citation_ids": list(dict.fromkeys(citation_ids)),
                }
            )
    return claims


def split_claim_sentences(text: str) -> list[str]:
    """Split prose without treating common clinical titles as sentence endings."""
    protected = text
    placeholders = {
        "Mr.": "Mr<prd>", "Mrs.": "Mrs<prd>", "Ms.": "Ms<prd>",
        "Dr.": "Dr<prd>", "Prof.": "Prof<prd>", "Sr.": "Sr<prd>",
        "Jr.": "Jr<prd>", "e.g.": "e<prd>g<prd>", "i.e.": "i<prd>e<prd>",
    }
    for abbreviation, placeholder in placeholders.items():
        protected = re.sub(re.escape(abbreviation), placeholder, protected, flags=re.IGNORECASE)
    return [item.replace("<prd>", ".") for item in re.split(r"(?<=[.!?])\s+", protected)]


def looks_like_raw_evidence_dump(answer: str) -> bool:
    """Detect serialized retrieval payloads and repeated extraction scaffolding."""
    lowered = answer.lower()
    marker_count = sum(marker in lowered for marker in _RAW_EVIDENCE_MARKERS)
    json_keys = re.findall(r'"(?:patient|measurements|findings|raw_text|confidence|value)"\s*:', answer, re.I)
    repeated_labels = sum(lowered.count(label) for label in ("raw ocr text:", "measurements:", "findings:"))
    return marker_count >= 2 or len(json_keys) >= 3 or repeated_labels >= 4


def assess_evidence_sufficiency(question: str, chunks: list[RetrievedChunk]) -> dict[str, Any]:
    """Conservative lexical screen used for harness diagnostics, not calibrated semantics."""
    if not chunks:
        return {"decision": "insufficient", "score": 0.0, "reason": "no_evidence", "method": "heuristic"}
    question_terms = _content_tokens(question, stopwords=_QUESTION_STOPWORDS)
    if not question_terms:
        return {"decision": "sufficient", "score": 1.0, "reason": "broad_question_with_evidence", "method": "heuristic"}
    evidence_terms = _content_tokens(" ".join(chunk.text for chunk in chunks))
    overlap = question_terms & evidence_terms
    score = len(overlap) / max(len(question_terms), 1)
    if not overlap:
        decision = "insufficient"
        reason = "no_distinctive_term_overlap"
    elif score < 0.34 and len(overlap) < 2:
        decision = "uncertain"
        reason = "limited_term_overlap"
    else:
        decision = "sufficient"
        reason = "question_terms_covered"
    return {
        "decision": decision,
        "score": round(score, 4),
        "reason": reason,
        "matched_terms": sorted(overlap)[:12],
        "question_term_count": len(question_terms),
        "method": "heuristic",
    }


def validate_answer_citations(
    answer: str,
    citations: list[Citation],
    evidence_chunks: list[RetrievedChunk] | None = None,
) -> dict[str, Any]:
    """Validate citation structure and estimate claim/passage support conservatively."""
    claims = extract_answer_claims(answer)
    evidence_chunks = evidence_chunks or []
    passages = {
        index: (evidence_chunks[index - 1].text if index <= len(evidence_chunks) else citation.quote)
        for index, citation in enumerate(citations, start=1)
    }
    valid_ids = set(passages)
    invalid_ids: set[int] = set()
    uncited_claims: list[int] = []
    supported_claims: list[int] = []
    unsupported_claims: list[int] = []
    contradicted_claims: list[int] = []
    claim_results: list[dict[str, Any]] = []

    for claim in claims:
        claim_index = claim["claim_index"]
        ids = claim["citation_ids"]
        missing = [citation_id for citation_id in ids if citation_id not in valid_ids]
        invalid_ids.update(missing)
        if not ids:
            uncited_claims.append(claim_index)
            decision = "uncited"
            score = 0.0
            reason = "no_citation_id"
        elif missing:
            decision = "invalid"
            score = 0.0
            reason = "citation_id_not_available"
        else:
            cited_text = " ".join(passages[citation_id] for citation_id in ids)
            decision, score, reason = _support_decision(claim["text"], cited_text)
            if decision == "supported":
                supported_claims.append(claim_index)
            elif decision == "contradicted":
                contradicted_claims.append(claim_index)
            else:
                unsupported_claims.append(claim_index)
        claim_results.append(
            {
                "claim_index": claim_index,
                "citation_ids": ids,
                "decision": decision,
                "support_score": round(score, 4),
                "reason": reason,
            }
        )

    total = len(claims)
    cited_count = total - len(uncited_claims)
    coverage = cited_count / total if total else 1.0
    structural_passed = not invalid_ids and not uncited_claims
    semantic_passed = not unsupported_claims and not contradicted_claims
    return {
        "passed": structural_passed and semantic_passed,
        "structural_passed": structural_passed,
        "semantic_passed": semantic_passed,
        "claim_count": total,
        "cited_claim_count": cited_count,
        "citation_coverage": round(coverage, 4),
        "invalid_citation_ids": sorted(invalid_ids),
        "uncited_claim_indexes": uncited_claims,
        "supported_claim_indexes": supported_claims,
        "unsupported_claim_indexes": unsupported_claims,
        "contradicted_claim_indexes": contradicted_claims,
        "claims": claim_results,
        "semantic_method": "lexical_heuristic_shadow",
    }


def _support_decision(claim: str, evidence: str) -> tuple[str, float, str]:
    claim_terms = _content_tokens(claim)
    evidence_terms = _content_tokens(evidence)
    overlap = claim_terms & evidence_terms
    score = len(overlap) / max(len(claim_terms), 1)
    claim_numbers = set(_NUMBER_RE.findall(claim))
    evidence_numbers = set(_NUMBER_RE.findall(evidence))
    shared_terms = len(overlap) >= 2 or score >= 0.4
    if claim_numbers and not claim_numbers.issubset(evidence_numbers):
        return ("contradicted" if shared_terms else "unsupported", score, "numeric_value_not_supported")
    claim_negated = bool(_NEGATION_RE.search(claim))
    evidence_negated = bool(_NEGATION_RE.search(evidence))
    if shared_terms and claim_negated != evidence_negated:
        return "contradicted", score, "negation_mismatch"
    if score >= 0.5 or (len(overlap) >= 3 and score >= 0.35):
        return "supported", score, "claim_terms_supported"
    return "unsupported", score, "insufficient_claim_passage_overlap"


def _content_tokens(text: str, *, stopwords: set[str] | None = None) -> set[str]:
    excluded = stopwords or _STOPWORDS
    return {
        token.lower()
        for token in _TOKEN_RE.findall(_CITATION_RE.sub("", text))
        if len(token) >= 2 and token.lower() not in excluded
    }
