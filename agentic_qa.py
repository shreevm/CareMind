from langchain.agents import Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from pinecone import Pinecone
import os
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

from dotenv import load_dotenv
import os

load_dotenv()

#  Init Pinecone 
pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
    )
#  Get Pinecone index using old method
index = pc.Index("caremind-index")

# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
model = AutoModel.from_pretrained("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

# Use LangChain Pinecone wrapper correctly
vectorstore = Pinecone(index, model, text_key="text")
retriever = vectorstore.as_retriever()
# Set up retrieval QA chain for clinical records
llm = OpenAI(temperature=0.3)
clinical_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def search_clinical_data(question: str) -> str:
    result = clinical_qa(question)
    return f"Answer: {result['result']}\n\nEvidence:\n" + "\n---\n".join(
        f"{doc.metadata.get('subject_id')}: {doc.page_content[:400]}" for doc in result['source_documents']
    )

#dummy function for medical guidelines
def medical_guideline_lookup(question: str) -> str:
    if "hypertension" in question.lower():
        return ("Hypertension treatment includes ACE inhibitors (e.g. lisinopril), beta blockers, lifestyle changes. "
                "Pre-surgery: monitor BP, reduce sodium. Post-surgery: anticoagulants, pain control.")
    elif "diabetes" in question.lower():
        return ("For diabetic patients: maintain HbA1c < 7%, insulin management, dietary regulation, monitor for complications.")
    else:
        return "Please refer to standard treatment protocols or consult UpToDate."

# Define tools
tools = [
    Tool(name="ClinicalRecordSearch", func=search_clinical_data,
         description="Look up structured patient data from MIMIC-IV"),
    Tool(name="MedicalGuidelines", func=medical_guideline_lookup,
         description="Suggest treatments or plans based on diagnosis")
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)


def run_clinical_agentic_query(question):
    return agent.run(question)

# Example usage
if __name__ == "__main__":
    q = "A patient has high blood pressure and needs heart surgery. What treatment plan is recommended?"
    print(run_clinical_agentic_query(q))
