import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(".chatenv")

embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the saved index — no embedding cost, instant
vectorstore = FAISS.load_local(
    "faiss_index",
    embedder,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully")
print(f"Total vectors in index: {vectorstore.index.ntotal}")

# Test 3 different queries
queries = [
    "What was the school dropout rate?",
    "How many hospitals were built under Ayushman Bharat?",
    "What is the budget allocation for higher education?",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    results = vectorstore.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Page:   {doc.metadata['page']}")
        print(f"Text:   {doc.page_content[:200]}")