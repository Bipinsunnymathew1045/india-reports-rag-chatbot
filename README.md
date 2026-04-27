# India Education & Health Reports Chatbot

A RAG (Retrieval Augmented Generation) chatbot that answers questions
across 10 official Government of India annual reports spanning 2020-2025.

## What it does
- Answers natural language questions about Indian education and health policy
- Retrieves answers from 3,492 pages across 10 government PDFs
- Cites exact page numbers and report years for every answer
- Compares data across multiple years automatically

## Tech stack
- LangChain — RAG pipeline and LLM orchestration
- OpenAI — text-embedding-3-small (embeddings) + gpt-4o-mini (answers)
- FAISS — vector similarity search across 21,330 chunks
- Streamlit — chat interface

## Architecture
1. ingest.py — loads PDFs, chunks text, embeds and saves FAISS index
2. query.py — loads index, retrieves relevant chunks, generates answers
3. app.py — Streamlit chat UI with citation display

## Setup
1. Clone the repo
2. pip install -r requirements.txt
3. Add your OpenAI API key to .env
4. Add PDF files to data/ folder
5. Run python ingest.py
6. Run streamlit run app.py

## Sample questions
- What was the GER in higher education in 2022?
- How many Ayushman Bharat cards were created?
- What is the National Education Policy about?
- How many AIIMS hospitals exist in India?