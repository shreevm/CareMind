import os
import gradio as gr
from langchain.agents import Tool, initialize_agent
from langchain.llms import HuggingFacePipeline
from langchain.tools import SerpAPIWrapper
from llm_chain import generate_answer, format_prompt
from retriever import retrieve_context
from guardrails import run_nli_guardrail
from transformers import pipeline

# qa_pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", device=0)
# llm = HuggingFacePipeline(pipeline=qa_pipe)

# # TOOL 1: MIMIC RAG

# def mimic_qa_tool(input):
#     context = retrieve_context(input)
#     prompt = format_prompt(context, input)
#     answer = generate_answer(prompt)
#     if run_nli_guardrail(context, answer):
#         return answer
#     return "[Guardrail Triggered] Answer not supported by context."

# # TOOL 2: Web Search
# search_tool = SerpAPIWrapper()

# tools = [
#     Tool(name="MIMIC-QA", func=mimic_qa_tool, description="Use for clinical questions from EHR"),
#     Tool(name="WebSearch", func=search_tool.run, description="Use for questions not found in MIMIC")
# ]

# agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# def multi_tool_qa(query):
#     return agent.run(query)

# iface = gr.Interface(fn=multi_tool_qa, inputs="text", outputs="text",
#                      title="CareMind Clinical Agent",
#                      description="Answers clinical questions via MIMIC-IV or external web tools.")

# if __name__ == '__main__':
#     iface.launch()
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import OpenAI
from transformers import pipeline
from retriever import retrieve
from guadrails_apply import apply_guardrails

# 1. Define retriever tool
def rag_retriever_tool(query: str) -> str:
    docs = retrieve(query)
    return "\n".join(docs[:3])

retriever_tool = Tool(
    name="Retriever",
    func=rag_retriever_tool,
    description="Retrieves clinical documents for a given query"
)

# 2. Define QA tool
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def qa_tool(query: str) -> str:
    context = rag_retriever_tool(query)
    result = qa_pipeline(question=query, context=context)
    return result["answer"]

qa_langchain_tool = Tool(
    name="QA Tool",
    func=qa_tool,
    description="Answers clinical questions using retrieved context"
)

# 3. Define Guardrail tool
def guardrail_tool(answer: str) -> str:
    return apply_guardrails(answer, [])

guardrail_langchain_tool = Tool(
    name="Guardrail",
    func=guardrail_tool,
    description="Filters hallucinated answers"
)

# 4. Combine into Agent
tools = [retriever_tool, qa_langchain_tool, guardrail_langchain_tool]

# Replace with your actual LLM, or use OpenAI/GPT if available
llm = HuggingFacePipeline(pipeline=qa_pipeline)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# 5. Run agent with a question
response = agent.run("What medications were prescribed for pneumonia?")
print(response)
 