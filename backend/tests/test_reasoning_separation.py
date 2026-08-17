from contextlib import contextmanager

from backend.caremind.config import Settings
import pytest

from backend.caremind.llm import GenerationContentUnavailable, LLMClient


def test_default_nvidia_model_and_env_override(monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_CHAT_MODEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.nvidia_chat_model == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert settings.nvidia_reasoning_enabled is True

    monkeypatch.setenv("NVIDIA_CHAT_MODEL", "meta/llama-3.1-8b-instruct")
    overridden = Settings(_env_file=None)

    assert overridden.nvidia_chat_model == "meta/llama-3.1-8b-instruct"


def test_generation_options_do_not_send_reasoning_as_provider_parameter() -> None:
    client = LLMClient(
        Settings(
            _env_file=None,
            nvidia_reasoning_enabled=True,
            llm_temperature=0.2,
            llm_top_p=0.8,
            llm_max_tokens=256,
        )
    )

    options = client._generation_options()

    assert options == {
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": 256,
    }
    assert "reasoning" not in options
    assert "reasoning_content" not in options
    assert "reasoning_enabled" not in options


def test_nonstreaming_reasoning_content_is_not_concatenated_or_traced(monkeypatch) -> None:
    trace_outputs: list[dict] = []

    @contextmanager
    def fake_trace_block(*args, **kwargs):
        class FakeRun:
            def end(self, outputs=None):
                trace_outputs.append(outputs or {})

        yield FakeRun()

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "private chain of thought should not leak",
                            "content": "Final answer only.",
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            assert "reasoning_content" not in kwargs["json"]
            return FakeResponse()

    monkeypatch.setattr("backend.caremind.llm.trace_block", fake_trace_block)
    monkeypatch.setattr("backend.caremind.llm.httpx.Client", FakeClient)

    client = LLMClient(Settings(_env_file=None, nvidia_api_key="test-key"))
    answer = client._answer_with_chat_endpoint(
        question="What is shown?",
        chunks=[],
        history=[],
        base_url="https://example.test/v1",
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        api_key="test-key",
        system_prompt="system",
    )

    assert answer == "Final answer only."
    assert "private chain" not in answer
    assert trace_outputs
    assert trace_outputs[-1]["reasoning_status"] == "complete"
    assert trace_outputs[-1]["reasoning_token_count"] > 0
    assert "private chain" not in str(trace_outputs[-1])


def test_streaming_reasoning_content_is_consumed_without_frontend_delta(monkeypatch) -> None:
    trace_outputs: list[dict] = []

    @contextmanager
    def fake_trace_block(*args, **kwargs):
        class FakeRun:
            def end(self, outputs=None):
                trace_outputs.append(outputs or {})

        yield FakeRun()

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter(
                [
                    'data: {"choices":[{"delta":{"reasoning_content":"hidden planning text"}}]}',
                    'data: {"choices":[{"delta":{"content":"Final "}}]}',
                    'data: {"choices":[{"delta":{"content":"answer."}}]}',
                    "data: [DONE]",
                ]
            )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            assert kwargs["json"]["stream"] is True
            assert "reasoning_content" not in kwargs["json"]
            return FakeStream()

    monkeypatch.setattr("backend.caremind.llm.trace_block", fake_trace_block)
    monkeypatch.setattr("backend.caremind.llm.httpx.Client", FakeClient)

    client = LLMClient(Settings(_env_file=None, nvidia_api_key="test-key"))
    pieces = list(
        client._answer_with_chat_endpoint_stream(
            question="What is shown?",
            chunks=[],
            history=[],
            base_url="https://example.test/v1",
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            api_key="test-key",
            system_prompt="system",
        )
    )

    assert "".join(pieces) == "Final answer."
    assert "hidden planning text" not in "".join(pieces)
    assert trace_outputs[-1]["reasoning_status"] == "complete"
    assert trace_outputs[-1]["reasoning_token_count"] > 0
    assert "hidden planning text" not in str(trace_outputs[-1])


def test_streaming_reasoning_inside_content_is_not_yielded(monkeypatch) -> None:
    trace_outputs: list[dict] = []

    @contextmanager
    def fake_trace_block(*args, **kwargs):
        class FakeRun:
            def end(self, outputs=None):
                trace_outputs.append(outputs or {})

        yield FakeRun()

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter(
                [
                    'data: {"choices":[{"delta":{"content":"Here is a thinking process:\\n1. Analyze user input.\\n"}}]}',
                    'data: {"choices":[{"delta":{"content":"2. Review evidence.\\n\\nPatient Details\\nThe report describes sinus rhythm [1]."}}]}',
                    "data: [DONE]",
                ]
            )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return FakeStream()

    monkeypatch.setattr("backend.caremind.llm.trace_block", fake_trace_block)
    monkeypatch.setattr("backend.caremind.llm.httpx.Client", FakeClient)

    client = LLMClient(Settings(_env_file=None, nvidia_api_key="test-key"))
    pieces = list(
        client._answer_with_chat_endpoint_stream(
            question="What is shown?",
            chunks=[],
            history=[],
            base_url="https://example.test/v1",
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            api_key="test-key",
            system_prompt="system",
        )
    )

    answer = "".join(pieces)
    assert answer.startswith("Patient Details")
    assert "thinking process" not in answer.lower()
    assert "Analyze user input" not in answer
    assert trace_outputs[-1]["reasoning_status"] == "complete"


def test_streaming_reasoning_only_content_raises_generation_failure(monkeypatch) -> None:
    trace_outputs: list[dict] = []

    @contextmanager
    def fake_trace_block(*args, **kwargs):
        class FakeRun:
            def end(self, outputs=None):
                trace_outputs.append(outputs or {})

        yield FakeRun()

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter(
                [
                    'data: {"choices":[{"delta":{"content":"Here is a thinking process:\\n1. Analyze user input.\\n2. Review evidence."}}]}',
                    "data: [DONE]",
                ]
            )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return FakeStream()

    monkeypatch.setattr("backend.caremind.llm.trace_block", fake_trace_block)
    monkeypatch.setattr("backend.caremind.llm.httpx.Client", FakeClient)

    client = LLMClient(Settings(_env_file=None, nvidia_api_key="test-key"))
    with pytest.raises(GenerationContentUnavailable):
        list(
            client._answer_with_chat_endpoint_stream(
                question="What is shown?",
                chunks=[],
                history=[],
                base_url="https://example.test/v1",
                model="nvidia/nemotron-3.5-lightning-30b-a3b",
                api_key="test-key",
                system_prompt="system",
            )
        )

    assert trace_outputs[-1]["final_content_received"] is False
    assert "Analyze user input" not in str(trace_outputs[-1])


def test_streaming_final_content_is_emitted_before_model_completion(monkeypatch) -> None:
    consumed: list[str] = []

    @contextmanager
    def fake_trace_block(*args, **kwargs):
        class FakeRun:
            def end(self, outputs=None):
                pass

        yield FakeRun()

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            for line in [
                'data: {"choices":[{"delta":{"reasoning_content":"internal analysis"}}]}',
                'data: {"choices":[{"delta":{"content":"The report "}}]}',
                'data: {"choices":[{"delta":{"content":"documents "}}]}',
                'data: {"choices":[{"delta":{"content":"sinus rhythm ["}}]}',
                'data: {"choices":[{"delta":{"content":"1"}}]}',
                'data: {"choices":[{"delta":{"content":"]."}}]}',
                'data: {"choices":[{"delta":{"content":" This is second sentence."}}]}',
                "data: [DONE]",
            ]:
                consumed.append(line)
                yield line

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return FakeStream()

    monkeypatch.setattr("backend.caremind.llm.trace_block", fake_trace_block)
    monkeypatch.setattr("backend.caremind.llm.httpx.Client", FakeClient)

    client = LLMClient(Settings(_env_file=None, nvidia_api_key="test-key"))
    stream = client._answer_with_chat_endpoint_stream(
        question="What is shown?",
        chunks=[],
        history=[],
        base_url="https://example.test/v1",
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        api_key="test-key",
        system_prompt="system",
    )

    first_visible = next(stream)

    assert first_visible == "The report documents sinus rhythm [1]."
    assert "This is second sentence" not in first_visible
    assert not any(line == "data: [DONE]" for line in consumed)
    assert "".join([first_visible, *list(stream)]) == (
        "The report documents sinus rhythm [1]. This is second sentence."
    )


def test_trace_text_and_chunk_outputs_never_return_raw_content() -> None:
    from backend.caremind.observability import chunk_outputs, trace_text
    from backend.caremind.schemas import RetrievedChunk

    settings = Settings(_env_file=None, langsmith_redact_inputs=False)
    text = "Patient Name: Jane Example. Secret reasoning should not appear."

    traced = trace_text(settings, text)
    chunks = chunk_outputs(
        settings,
        [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                document_name="report.pdf",
                text=text,
            )
        ],
    )

    assert isinstance(traced, dict)
    assert text not in str(traced)
    assert text not in str(chunks)
    assert "preview" not in chunks[0]
