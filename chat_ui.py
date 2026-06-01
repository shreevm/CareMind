import os
import uuid
import json
import threading

import gradio as gr
from retriever import retrieve
from guadrails_apply import apply_guardrails
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TextIteratorStreamer

# ── Redis with in-memory fallback ────────────────────────────────────────────
try:
    import redis as _redis_lib
    _redis = _redis_lib.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD") or None,
        db=0,
        decode_responses=True,
    )
    _redis.ping()
    USE_REDIS = True
    print("[ok] Redis connected")
except Exception:
    USE_REDIS = False
    _mem_store: dict = {}
    print("[!] Redis unavailable - using in-memory session store")

SESSION_TTL = 3600  # seconds


def _session_key(sid: str) -> str:
    return f"caremind:session:{sid}"


def load_history(sid: str) -> list:
    if USE_REDIS:
        raw = _redis.get(_session_key(sid))
        return json.loads(raw) if raw else []
    return _mem_store.get(sid, [])


def save_history(sid: str, history: list) -> None:
    if USE_REDIS:
        _redis.setex(_session_key(sid), SESSION_TTL, json.dumps(history))
    else:
        _mem_store[sid] = history


# ── Local LLM ────────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("CAREMIND_MODEL", "google/flan-t5-small")
print(f"[..] Loading local model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print(f"[ok] Model loaded: {MODEL_NAME}")


def _build_prompt(context: str, history: list, question: str) -> str:
    turns = ""
    for msg in history[-4:]:  # last 2 full turns
        prefix = "User" if msg["role"] == "user" else "CareMind"
        turns += f"{prefix}: {msg['content']}\n"
    return (
        "You are CareMind, a clinical decision-support assistant. "
        "Answer using only the patient records provided. "
        "If the answer is not in the records, say 'I don't know.'\n\n"
        f"--- PATIENT RECORDS ---\n{context}\n\n"
        f"--- CONVERSATION HISTORY ---\n{turns}"
        f"User: {question}\nCareMind:"
    )


def stream_response(message: str, session_id: str):
    """Generator that yields partial tokens as the model produces them."""
    docs = retrieve(message)
    context = "\n\n".join(docs[:3]) if docs else "No relevant patient records found."

    history = load_history(session_id)
    prompt = _build_prompt(context, history, message)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    thread = threading.Thread(
        target=model.generate,
        kwargs={**inputs, "streamer": streamer, "max_new_tokens": 256},
    )
    thread.start()

    partial = ""
    for chunk in streamer:
        partial += chunk
        yield partial

    final = apply_guardrails(partial)
    if final != partial:
        yield final

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": final})
    save_history(session_id, history)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
EXAMPLES = [
    "What treatment did patient 10018533 receive?",
    "Which patients received vancomycin?",
    "What medications were prescribed for pneumonia?",
    "A patient has hypertension and needs surgery. What is the plan?",
    "How should diabetes be managed post-operatively?",
]


def respond(message: str, history: list, session_id: str):
    if not message.strip():
        yield "", history
        return
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    for partial in stream_response(message, session_id):
        history[-1]["content"] = partial
        yield "", history


with gr.Blocks(title="CareMind - Clinical QA") as demo:
    session_state = gr.State(value=lambda: str(uuid.uuid4()))

    gr.Markdown(
        "## CareMind - Clinical QA\n"
        "Ask questions about MIMIC-IV patient records or general clinical guidelines."
    )

    chatbot = gr.Chatbot(
        height=500,
        label="CareMind",
        placeholder="CareMind is ready. Ask a clinical question.",
        show_copy_button=True,
        type="messages",
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="e.g. What treatment did patient 10018533 receive?",
            show_label=False,
            scale=8,
            autofocus=True,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("Clear Chat")

    gr.Examples(
        examples=EXAMPLES,
        inputs=msg_box,
        label="Example Questions",
    )

    # Wire events
    msg_box.submit(
        fn=respond,
        inputs=[msg_box, chatbot, session_state],
        outputs=[msg_box, chatbot],
    )
    send_btn.click(
        fn=respond,
        inputs=[msg_box, chatbot, session_state],
        outputs=[msg_box, chatbot],
    )
    clear_btn.click(
        fn=lambda: ([], str(uuid.uuid4())),
        outputs=[chatbot, session_state],
    )


if __name__ == "__main__":
    demo.launch(share=True)
