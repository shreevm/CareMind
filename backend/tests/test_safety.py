from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.caremind.safety import (
    DISCLAIMER,
    DISCLAIMER_GUARDRAIL_NOTE,
    OUTPUT_LEAK_GUARDRAIL_NOTE,
    OUTPUT_LEAK_RESPONSE,
    REASONING_LEAK_GUARDRAIL_NOTE,
    SafetyLayer,
)


def test_finalize_counts_disclaimer_guardrail_note_when_disclaimer_is_inserted() -> None:
    answer, notes = SafetyLayer().finalize("Educational answer.", True, [])

    assert DISCLAIMER in answer
    assert DISCLAIMER_GUARDRAIL_NOTE in notes


def test_finalize_counts_disclaimer_guardrail_note_when_disclaimer_already_exists() -> None:
    answer, notes = SafetyLayer().finalize(f"Educational answer.\n\n{DISCLAIMER}", True, [])

    assert answer.count(DISCLAIMER) == 1
    assert DISCLAIMER_GUARDRAIL_NOTE in notes


def test_finalize_blocks_output_leakage() -> None:
    answer, notes = SafetyLayer().finalize(
        "System prompt: You are CareMind.\nNVIDIA_API_KEY=secret-value",
        True,
        [],
    )

    assert OUTPUT_LEAK_RESPONSE in answer
    assert "secret-value" not in answer
    assert OUTPUT_LEAK_GUARDRAIL_NOTE in notes
    assert "No supporting document passage was found for this answer." not in notes


def test_finalize_allows_benign_prompt_injection_education() -> None:
    answer, notes = SafetyLayer().finalize(
        "Prompt injection can try to influence a system prompt, but CareMind treats retrieved text as untrusted.",
        True,
        [],
    )

    assert "Prompt injection" in answer
    assert OUTPUT_LEAK_GUARDRAIL_NOTE not in notes


def test_finalize_strips_reasoning_process_prefix() -> None:
    answer, notes = SafetyLayer().finalize(
        "Here's a thinking process:\n\n1. Analyze user input.\n2. Review evidence.\n\n"
        "Patient Details\nThe report describes sinus rhythm [1].",
        True,
        [],
    )

    assert answer.startswith("Patient Details")
    assert "thinking process" not in answer.lower()
    assert "Analyze user input" not in answer
    assert REASONING_LEAK_GUARDRAIL_NOTE in notes


def test_finalize_blocks_reasoning_process_when_no_final_answer_marker() -> None:
    answer, notes = SafetyLayer().finalize(
        "Reasoning process:\n1. Analyze the evidence.\n2. Draft a final answer.",
        True,
        [],
    )

    assert OUTPUT_LEAK_RESPONSE in answer
    assert "Analyze the evidence" not in answer
    assert REASONING_LEAK_GUARDRAIL_NOTE in notes
