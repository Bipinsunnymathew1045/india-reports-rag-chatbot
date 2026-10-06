import os
import ragas
import sqlite3
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    AnswerRelevancy,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LCOpenAIEmbeddings
from query import ask, retriever

load_dotenv(".chatenv")

ragas.llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=2000)
embeddings_model = LCOpenAIEmbeddings(model="text-embedding-3-small")
answer_relevancy_metric = AnswerRelevancy(embeddings=embeddings_model)

# ── Golden dataset — 7 questions (3 education + 4 health) ──────────
golden_dataset = [
    # Education questions
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
    # Health questions
    {
        "question": "How many Ayushman Bharat Health and Wellness Centres were operational as of 2023-24?",
        "ground_truth": "As of 2023-24, 1,72,148 Ayushman Arogya Mandir Health and Wellness Centres were operational as reported by States/UTs on the HWC Portal."
    },
    {
        "question": "How many AIIMS hospitals exist in India and what is their total bed capacity as of 2023-24?",
        "ground_truth": "As of 2023-24 there are 18 AIIMS hospitals in India with a total bed capacity of 7,045 beds."
    },
    {
        "question": "How many beneficiaries received treatment under Ayushman Bharat PM-JAY scheme as of December 2024?",
        "ground_truth": "As of December 31, 2024, a total of 36.36 crore hospital admissions were recorded under AB PM-JAY."
    },
    {
        "question": "What was the total budget allocation and expenditure for the National Health Mission in 2023-24?",
        "ground_truth": "In 2023-24 the total NHM budget allocation was Rs. 7,159.77 crores. Expenditure up to December 2024 was Rs. 3,959.51 crores."
    },
]

# ── Run questions through chatbot ──────────────────────────────────
print("Running questions through RAG chatbot...")
print(f"Total questions: {len(golden_dataset)}\n")

questions = []
answers = []
contexts = []
ground_truths = []

for i, item in enumerate(golden_dataset):
    question = item["question"]
    domain = "health" if i >= 3 else "education"
    print(f"Q{i+1} [{domain}]: {question[:60]}...")

    result = ask(question)
    answer = result["answer"]
    docs = retriever.invoke(question)
    context = [doc.page_content for doc in docs]

    questions.append(question)
    answers.append(answer)
    contexts.append(context)
    ground_truths.append(item["ground_truth"])

    print(f"A: {answer[:100]}...")
    print()

# ── Build RAGAS dataset ────────────────────────────────────────────
print("Building evaluation dataset...")

ragas_data = Dataset.from_dict({
    "question":     questions,
    "answer":       answers,
    "contexts":     contexts,
    "ground_truth": ground_truths,
})

# ── Evaluate ───────────────────────────────────────────────────────
print("Running RAGAS evaluation...")
print("Takes ~3-4 minutes for 7 questions...\n")

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
print("RAGAS EVALUATION RESULTS — 7 QUESTIONS")
print("="*60)

df = results.to_pandas()

print("\nScores per question:")
for i, row in df.iterrows():
    domain = "health" if i >= 3 else "education"
    print(f"\nQ{i+1} [{domain}]: {questions[i][:55]}...")
    print(f"  Faithfulness:      {row['faithfulness']:.3f}" if str(row['faithfulness']) != 'nan' else "  Faithfulness:      NaN")
    print(f"  Answer Relevancy:  {row['answer_relevancy']:.3f}")
    print(f"  Context Precision: {row['context_precision']:.3f}")
    print(f"  Context Recall:    {row['context_recall']:.3f}")

print("\n" + "="*60)
print("AVERAGE SCORES:")
print(f"  Faithfulness:      {df['faithfulness'].mean():.3f}")
print(f"  Answer Relevancy:  {df['answer_relevancy'].mean():.3f}")
print(f"  Context Precision: {df['context_precision'].mean():.3f}")
print(f"  Context Recall:    {df['context_recall'].mean():.3f}")
print(f"  Overall:           {df[['faithfulness','answer_relevancy','context_precision','context_recall']].mean().mean():.3f}")

# ── Education vs Health breakdown ──────────────────────────────────
edu_df = df.iloc[:3]
health_df = df.iloc[3:]

print("\nEDUCATION questions average:")
print(f"  Overall: {edu_df[['faithfulness','answer_relevancy','context_precision','context_recall']].mean().mean():.3f}")

print("\nHEALTH questions average:")
print(f"  Overall: {health_df[['faithfulness','answer_relevancy','context_precision','context_recall']].mean().mean():.3f}")

# ── Save to database ───────────────────────────────────────────────
from store import init_db, save_run

init_db()
run_id = save_run(df, questions, answers, ground_truths)
print(f"\nResults saved to database. Run ID: {run_id}")