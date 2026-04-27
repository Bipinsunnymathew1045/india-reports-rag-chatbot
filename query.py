from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(".chatenv")

# ── Load FAISS index ───────────────────────────────────────────────
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.load_local(
    "faiss_index",
    embedder,
    allow_dangerous_deserialization=True
)

# ── Set up retriever and LLM ───────────────────────────────────────
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

# ── Helper to format retrieved chunks into one string ─────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Build the chain using pipe operator ───────────────────────────
# This is the modern LangChain way — called LCEL (LangChain Expression Language)
# Read it left to right:
# 1. retriever fetches chunks, question passes through
# 2. prompt fills {context} and {question}
# 3. llm generates the answer
# 4. StrOutputParser extracts the text from the LLM response object
chain = (
    {
        "context": retriever | format_docs,  # retrieve chunks → format as string
        "question": RunnablePassthrough()    # pass question through unchanged
    }
    | prompt          # fill the template
    | llm             # generate answer
    | StrOutputParser()  # extract text from response
)

# ── The ask function ───────────────────────────────────────────────
def ask(question: str) -> dict:
    """
    Takes a question string.
    Returns answer + citations.
    """
    # Get answer from chain
    answer = chain.invoke(question)

    # Get source documents separately for citations
    source_docs = retriever.invoke(question)

    # Build deduplicated citation list
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
    print(f"Sources:")
    for citation in result["citations"]:
        print(f"  - {citation}")