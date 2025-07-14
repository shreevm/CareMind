# import gradio as gr
# from agentic_qa import run_clinical_agentic_query, clinical_qa

# # Enhanced version: includes evidence display and question suggestions

# def query_with_evidence(user_question):
#     result = clinical_qa(user_question)
#     answer = result['result']
#     evidence = "\n---\n".join(
#         f"📄 Subject {doc.metadata.get('subject_id')}\n{doc.page_content[:400]}" for doc in result['source_documents']
#     )
#     return answer, evidence

# suggested_questions = [
#     "What treatment did patient 10018533 receive?",
#     "Which patients received vancomycin?",
#     "What medications are prescribed for pneumonia?",
#     "A patient has hypertension and needs surgery. What is the plan?",
#     "How should diabetes be managed post-operatively?"
# ]

# # Define Gradio UI
# with gr.Blocks() as demo:
#     gr.Markdown("# CareMind: Clinical Agentic QA System")
#     gr.Markdown("""
#         Ask clinical questions using patient IDs or general diagnoses. 
#         The system retrieves data from structured MIMIC-IV records and suggests treatments based on external knowledge.
#     """)

#     with gr.Row():
#         for question in suggested_questions:
#             gr.Button(question).click(fn=query_with_evidence, inputs=gr.State(question), outputs=[gr.Textbox.update(), gr.Textbox.update()])

#     user_input = gr.Textbox(label="Enter your clinical question", placeholder="e.g. What treatment is given to patient 10018533?")
#     submit_btn = gr.Button("Submit")
#     answer_output = gr.Textbox(label="Answer")
#     evidence_output = gr.Textbox(label="Supporting Clinical Evidence")

#     submit_btn.click(fn=query_with_evidence, inputs=user_input, outputs=[answer_output, evidence_output])

# if __name__ == "__main__":
#     demo.launch()
