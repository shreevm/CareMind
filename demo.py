import gradio as gr
from retriever import retrieve
from llm_qa import answer_question
from guadrails_apply import apply_guardrails

def ask_question(question):
    retrieved_ids = retrieve(question)
    context_docs = retrieved_ids  # In real setup, fetch text via metadata
    raw_answer = answer_question(question, context_docs)
    final_answer = apply_guardrails(raw_answer, context_docs)
    return final_answer


demo = gr.Interface(fn=ask_question,
                    inputs="text",
                    outputs="text",
                    title="CareMind: Clinical QA System",
                    description="Ask patient-related questions based on MIMIC-IV clinical notes.")

if __name__ == "__main__":
    demo.launch()
