import os
import gdown
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from huggingface_hub import hf_hub_download

# ── Load API key ───────────────────────────────────────────────────
# Locally reads from .chatenv
# On Streamlit Cloud reads from st.secrets
load_dotenv(".chatenv")
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]



# ── Download FAISS index from Hugging Face if not present ──────────
HF_REPO_ID = "BipinSunny/india-reports-faiss-index"

if not os.path.exists("faiss_index/index.faiss"):
    print("Downloading FAISS index from Hugging Face...")
    os.makedirs("faiss_index", exist_ok=True)

    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="index.faiss",
        repo_type="dataset",
        local_dir="faiss_index"
    )
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="index.pkl",
        repo_type="dataset",
        local_dir="faiss_index"
    )
    print("Download complete.")

# ── Load FAISS index ───────────────────────────────────────────────
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.load_local(
    "faiss_index",
    embedder,
    allow_dangerous_deserialization=True
)

# ── Retriever and LLM ──────────────────────────────────────────────
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# ── Prompt template ────────────────────────────────────────────────
prompt_template = """
You are an expert analyst of Indian government education and health reports
spanning 2020 to 2025.

Use the context provided below to answer the question as completely as possible.
The context may contain tables, numbers, and data from multiple report years.

Guidelines:
- Extract specific numbers, percentages, and figures when available
- If data spans multiple years in the context, compare them
- If the context contains partial information, share what you found
  and mention it may be incomplete
- Only say "I could not find this information" if the context contains
  absolutely nothing relevant to the question
- Always mention the report year your answer comes from
- For budget or table data, extract the numbers even if formatting looks messy

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ── Format docs helper ─────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Build chain ────────────────────────────────────────────────────
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ── Ask function ───────────────────────────────────────────────────
def ask(question: str) -> dict:
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)

    seen = set()
    citations = []
    for doc in source_docs:
        source = doc.metadata["source"].replace("data\\", "").replace("data/", "")
        page = doc.metadata.get("page_label", doc.metadata["page"] + 1)
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            citations.append(f"{source}, page {page}")

    return {
        "answer": answer,
        "citations": citations
    }

# ── Test block ─────────────────────────────────────────────────────
if __name__ == "__main__":
    test_question = "What was the gross enrollment ratio in higher education?"
    print(f"Question: {test_question}\n")
    result = ask(test_question)
    print(f"Answer:\n{result['answer']}\n")
    print("Sources:")
    for citation in result["citations"]:
        print(f"  - {citation}")