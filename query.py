import os
import gdown
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv(".chatenv")

# ── Load API key ───────────────────────────────────────────────────
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# ── Download FAISS index from Hugging Face if not present ──────────
from huggingface_hub import hf_hub_download

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

# ── FIX 3 — Increase max_tokens to prevent NaN faithfulness ───────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=2000      # ← Fix 3 added here
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

# ── FIX 2 — Domain detection function ─────────────────────────────
def detect_domain(question: str) -> str:
    health_keywords = [
        "hospital", "health", "disease", "medical", "doctor",
        "vaccine", "mortality", "aiims", "ayushman", "medicine",
        "nutrition", "maternal", "infant", "epidemic", "outbreak",
        "pharmaceutical", "surgery", "clinical", "patient",
        "population research", "prc", "family welfare", "mohfw",
        "immunization", "sanitation", "nhm", "nrhm", "pmjay",
        "nursing", "pharmacy", "dental", "cancer", "tb", "hiv",
        "malaria", "diabetes", "mental health", "disability"
    ]
    education_keywords = [
        "school", "education", "student", "teacher", "university",
        "college", "enrollment", "dropout", "literacy", "curriculum",
        "scholarship", "ger", "higher education", "ugc", "aicte",
        "naac", "nep", "samagra", "stars", "diksha", "swayam",
        "ignou", "iit", "iim", "nit", "grievance", "cpgrams"
    ]

    question_lower = question.lower()

    health_score = sum(1 for w in health_keywords if w in question_lower)
    education_score = sum(1 for w in education_keywords if w in question_lower)

    # If scores are equal or both zero — search both domains
    if health_score == education_score:
        return "both"
    elif health_score > education_score:
        return "health"
    else:
        return "education"
# ── Format docs helper ─────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Default retriever (used by eval.py) ───────────────────────────
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# ── FIX 2 — Domain-filtered ask function ──────────────────────────
def ask(question: str) -> dict:
    domain = detect_domain(question)
    print(f"Detected domain: {domain}")

    # Build retriever based on domain
    if domain == "both":
        # No filter — search everything
        active_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 8}
        )
    else:
        # Filter by detected domain
        active_retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 8,
                "filter": {"domain": domain}
            }
        )

    # Build chain with active retriever
    filtered_chain = (
        {
            "context": active_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = filtered_chain.invoke(question)
    source_docs = active_retriever.invoke(question)

    seen = set()
    citations = []
    for doc in source_docs:
        source = doc.metadata["source"].replace("data\\", "").replace("data/", "")
        page = doc.metadata["page"] + 1
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
        

# Add this temporarily to query.py test block
if __name__ == "__main__":
    # Check metadata of first few chunks
    docs = vectorstore.similarity_search("education", k=3)
    for doc in docs:
        print(doc.metadata)
        

if __name__ == "__main__":
    test_questions = [
        "How many Population Research Centres exist in India as of 2023-24?",
        "How many public grievances were registered under CPGRAMS in the Education Ministry in 2023-24?",
        "What was the gross enrollment ratio in higher education?",
    ]

    for test_question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {test_question}")
        result = ask(test_question)
        print(f"Answer: {result['answer'][:150]}...")
        print(f"First citation: {result['citations'][0] if result['citations'] else 'None'}")