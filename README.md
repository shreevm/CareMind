# README.md 
# CareMind Clinical QA System

A Retrieval-Augmented Generation (RAG) system designed to **answer clinical questions** based on MIMIC-IV data, using **LangChain**, **LLaMA 2**, **BioBERT embeddings**, and **Pinecone** for retrieval. Guardrails are enforced using **Natural Language Inference (NLI)** to reduce hallucinations.


This project builds an intelligent clinical QA system powered by LLaMA and LangChain. It uses:

- MIMIC-IV data via Pinecone + BioBERT retriever
- Advanced prompt-based LLaMA2 for generation
- NLI-based guardrails to reduce hallucinations
- External search fallback (via SerpAPI)

---

## Features: 

- **Context-Aware Retrieval**: Uses BioBERT + Pinecone to pull semantically relevant patient notes from MIMIC-IV.
- **LLaMA 2 QA Model**: Generates grounded, clinical answers using a structured prompt.
- **Model-Based Guardrails (NLI)**:
  - Applies a pre-trained RoBERTa NLI model to detect hallucinated or unfaithful outputs.
- **LangChain Framework**: Modular, production-ready RAG pipeline.
- **Gradio Interface**: Simple UI for clinicians, students, or researchers to interact with.

---

### System Architecture
```
[Clinical Query]
   ↓
[Pinecone + BioBERT Retriever]
   ↓
[Prompt Template]
   ↓
[LLaMA 2 Generation]
   ↓
[NLI Guardrails (RoBERTa-MedNLI)]
   ↓
[Fallback: SerpAPI for Web Answers]
   ↓
[Gradio UI]
```

## How to Run (Locally or Hugging Face Spaces)
1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```
2. Set your `.env` variables:
```
PINECONE_API_KEY=your_key
SERPAPI_API_KEY=your_key
```
3. Launch the app:
```bash
python caremind_agent.py
```


## Deployment

To deploy on Hugging Face Spaces:
1. Create a new Space (Gradio + GPU)
2. Upload all files
3. Set `PINECONE_API_KEY`,`SERP_API_KEY` in Secrets
4. Click **Run**

You're now ready to deploy this to Hugging Face Spaces! Just include `requirements.txt`, `caremind_agent.py`, and the rest of the files in your repo. Spaces will auto-launch your Gradio interface.