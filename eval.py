import os
import ragas
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    AnswerRelevancy,
)
from langchain_openai import ChatOpenAI,OpenAIEmbeddings as LCOpenAIEmbeddings
from query import ask, retriever


ragas.llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=2000)

load_dotenv(".chatenv")

# ── Embeddings for answer_relevancy metric ─────────────────────────
embeddings_model = LCOpenAIEmbeddings(model="text-embedding-3-small")
answer_relevancy_metric = AnswerRelevancy(embeddings=embeddings_model)

# ── Golden dataset ─────────────────────────────────────────────────
golden_dataset = [
    {
        "question": "How many Population Research Centres exist in India as of 2023-24 and how many research projects did they complete?",
        "ground_truth": "As of 2023-24 there are 18 Population Research Centres in India. They completed 95 research studies and monitored 355 districts under NHM."
    },
    {
        "question": "How many public grievances were registered under CPGRAMS in the Education Ministry in 2023-24 and what percentage were resolved?",
        "ground_truth": "33,772 grievances were received between April 2023 and March 2024. 32,822 were resolved — approximately 97% resolution rate."
    },
    {
        "question": "STARS was implemented by the government in the year 2020, how effective was the initiative?",
        "ground_truth": "STARS was approved in October 2020 and became effective February 23 2021. It covers 6 states. Governance index improved by 10 points in each state. All 6 states received Vidya Sameeksha Kendra. It supports NAS and PARAKH assessment centres."
    },
]

# ── Run questions through chatbot ──────────────────────────────────
print("Running questions through RAG chatbot...")

questions = []
answers = []
contexts = []
ground_truths = []

for item in golden_dataset:
    question = item["question"]
    print(f"\nQ: {question[:60]}...")

    result = ask(question)
    answer = result["answer"]
    docs = retriever.invoke(question)
    context = [doc.page_content for doc in docs]

    questions.append(question)
    answers.append(answer)
    contexts.append(context)
    ground_truths.append(item["ground_truth"])

    print(f"A: {answer[:100]}...")

# ── Build RAGAS dataset ────────────────────────────────────────────
print("\nBuilding evaluation dataset...")

ragas_data = Dataset.from_dict({
    "question":     questions,
    "answer":       answers,
    "contexts":     contexts,
    "ground_truth": ground_truths,
})

# ── Evaluate ───────────────────────────────────────────────────────
print("\nRunning RAGAS evaluation...")
print("Takes ~1-2 minutes and costs ~$0.05...")

results = evaluate(
    dataset=ragas_data,
    metrics=[
        faithfulness,
        answer_relevancy_metric,
        context_precision,
        context_recall,
    ]
)

# ── Display results ────────────────────────────────────────────────
print("\n" + "="*60)
print("RAGAS EVALUATION RESULTS")
print("="*60)

df = results.to_pandas()
print("\nColumn names:", df.columns.tolist())
print("\nRaw results:")
print(df.to_string())

print("\n" + "="*60)
print("AVERAGE SCORES:")
for col in df.columns:
    if df[col].dtype in ['float64', 'float32']:
        print(f"  {col}: {df[col].mean():.3f}")

# ── Save to database ───────────────────────────────────────────────
from store import init_db, save_run

init_db()
run_id = save_run(df, questions, answers, ground_truths)
print(f"\nResults saved to database. Run ID: {run_id}")