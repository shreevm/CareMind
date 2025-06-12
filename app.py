import os
import gradio as gr
from langchain.agents import Tool, initialize_agent
from langchain.llms import HuggingFacePipeline
from langchain.tools import SerpAPIWrapper
from llm_chain import generate_answer, format_prompt
from retriever import retrieve_context
from guardrails import run_nli_guardrail
from transformers import pipeline

qa_pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", device=0)
llm = HuggingFacePipeline(pipeline=qa_pipe)

# TOOL 1: MIMIC RAG

def mimic_qa_tool(input):
    context = retrieve_context(input)
    prompt = format_prompt(context, input)
    answer = generate_answer(prompt)
    if run_nli_guardrail(context, answer):
        return answer
    return "[Guardrail Triggered] Answer not supported by context."

# TOOL 2: Web Search
search_tool = SerpAPIWrapper()

tools = [
    Tool(name="MIMIC-QA", func=mimic_qa_tool, description="Use for clinical questions from EHR"),
    Tool(name="WebSearch", func=search_tool.run, description="Use for questions not found in MIMIC")
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

def multi_tool_qa(query):
    return agent.run(query)

iface = gr.Interface(fn=multi_tool_qa, inputs="text", outputs="text",
                     title="CareMind Clinical Agent",
                     description="Answers clinical questions via MIMIC-IV or external web tools.")

if __name__ == '__main__':
    iface.launch()