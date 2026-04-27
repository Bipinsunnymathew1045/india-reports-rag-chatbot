import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(".chatenv")

# ── Step 1: Load all PDFs ──────────────────────────────────────────
PDF_FOLDER = "data"

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDFs: {pdf_files}")

all_documents = []
for pdf_file in pdf_files:
    path = os.path.join(PDF_FOLDER, pdf_file)
    print(f"Loading {pdf_file}...")
    loader = PyPDFLoader(path)
    pages = loader.load()
    all_documents.extend(pages)
    print(f"  → {len(pages)} pages loaded")

print(f"\nTotal pages across all PDFs: {len(all_documents)}")

# ── Step 2: Chunk the documents ────────────────────────────────────
print("\nChunking documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # max characters per chunk
    chunk_overlap=50,      # overlap between chunks so context isn't lost
    length_function=len,
)

chunks = splitter.split_documents(all_documents)
print(f"Total chunks created: {len(chunks)}")
print(f"\nExample chunk:")
print(f"Content: {chunks[0].page_content[:200]}")
print(f"Metadata: {chunks[0].metadata}")

# ── Step 3: Embed and store in FAISS ──────────────────────────────
print("\nEmbedding chunks and building FAISS index...")
print("This will take a few minutes for 10 PDFs...")

embedder = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.from_documents(chunks, embedder)

# ── Step 4: Save the index so you never pay to embed again ────────
vectorstore.save_local("faiss_index")
print("\nFAISS index saved to faiss_index/")
print("You will never need to re-embed these documents again.")