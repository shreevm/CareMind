from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.caremind.citation_validation import (
    assess_evidence_sufficiency,
    extract_answer_claims,
    looks_like_raw_evidence_dump,
    validate_answer_citations,
)
from backend.caremind.schemas import Citation, RetrievedChunk


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="doc-1:0",
        document_id="doc-1",
        document_name="report.txt",
        text=text,
        score=0.9,
    )


def _citation(text: str) -> Citation:
    return Citation(
        chunk_id="doc-1:0",
        document_id="doc-1",
        document_name="report.txt",
        quote=text,
        score=0.9,
    )


def test_claim_splitting_preserves_titles_and_ignores_plain_headings() -> None:
    claims = extract_answer_claims(
        "Patient summary\nDr. Rivera reviewed the report with Mr. Jones [1].\nKey clinical findings\nHemoglobin was 10 g/dL [1]."
    )

    assert [item["text"] for item in claims] == [
        "Dr. Rivera reviewed the report with Mr. Jones .",
        "Hemoglobin was 10 g/dL .",
    ]


def test_raw_serialized_evidence_dump_is_detected() -> None:
    answer = '{"patient": {"name": "Synthetic"}, "measurements": [{"value": "?"}], "findings": ["low"]}'
    assert looks_like_raw_evidence_dump(answer) is True


def test_extract_answer_claims_preserves_ids_and_skips_disclaimer() -> None:
    claims = extract_answer_claims(
        "Hemoglobin is 11.2 g/dL [1].\n"
        "CareMind is for medical education and is not a diagnosis or treatment plan."
    )

    assert claims == [
        {
            "claim_index": 1,
            "text": "Hemoglobin is 11.2 g/dL .",
            "citation_ids": [1],
        }
    ]


def test_citation_validation_accepts_supported_claim() -> None:
    evidence = "Laboratory results show hemoglobin 11.2 g/dL and mild anemia."

    result = validate_answer_citations(
        "The hemoglobin result is 11.2 g/dL [1].",
        [_citation(evidence)],
        [_chunk(evidence)],
    )

    assert result["structural_passed"] is True
    assert result["semantic_passed"] is True
    assert result["citation_coverage"] == 1.0


def test_citation_validation_finds_invalid_and_uncited_claims() -> None:
    result = validate_answer_citations(
        "Hemoglobin is low [2]. Follow-up was recommended.",
        [_citation("Hemoglobin is low.")],
    )

    assert result["structural_passed"] is False
    assert result["invalid_citation_ids"] == [2]
    assert result["uncited_claim_indexes"] == [2]
    assert result["citation_coverage"] == 0.5


def test_citation_validation_flags_numeric_mismatch() -> None:
    evidence = "Hemoglobin was 13.8 g/dL."

    result = validate_answer_citations(
        "Hemoglobin was 11.2 g/dL [1].",
        [_citation(evidence)],
        [_chunk(evidence)],
    )

    assert result["semantic_passed"] is False
    assert result["contradicted_claim_indexes"] == [1]
    assert result["claims"][0]["reason"] == "numeric_value_not_supported"


def test_evidence_sufficiency_reports_missing_and_matching_evidence() -> None:
    missing = assess_evidence_sufficiency("What is the hemoglobin?", [])
    matching = assess_evidence_sufficiency(
        "What is the hemoglobin value?",
        [_chunk("The hemoglobin value is 11.2 g/dL.")],
    )

    assert missing["decision"] == "insufficient"
    assert matching["decision"] == "sufficient"
    assert matching["method"] == "heuristic"
