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

## Evaluation Pipeline (Project 2)

Automated RAG evaluation using RAGAS framework across 7 test questions
covering both education and health domains.

### Metrics tracked
- **Faithfulness** — are all claims traceable to retrieved chunks?
- **Answer Relevancy** — does the answer address the question asked?
- **Context Precision** — were retrieved chunks actually relevant?
- **Context Recall** — did the answer use all available information?

### Key findings
| Domain    | Overall Score |
|-----------|--------------|
| Health    | 0.832        |
| Education | 0.560        |
| Overall   | 0.723        |

Health domain scores higher due to structured numerical data in reports.
Education domain identified as priority for improvement.

### Failure modes identified
1. Temporal drift — old chunks retrieved for current year questions
2. Shallow grounding — list questions produce unverifiable summaries
3. Ghost citations — retrieved but unused chunks listed as sources
4. Depth truncation — LLM stops early despite rich context available
5. Data gaps — well-formed questions where answer isn't in documents

### Stack
RAGAS · SQLite · Pandas · Streamlit · LangChain · OpenAI
- How many Ayushman Bharat cards were created?
- What is the National Education Policy about?
- How many AIIMS hospitals exist in India?
