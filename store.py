import sqlite3
import json
from datetime import datetime

DB_PATH = "eval_results.db"

def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp TEXT NOT NULL,
            total_questions INTEGER,
            avg_faithfulness REAL,
            avg_answer_relevancy REAL,
            avg_context_precision REAL,
            avg_context_recall REAL,
            avg_overall REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eval_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            question TEXT,
            answer TEXT,
            ground_truth TEXT,
            faithfulness REAL,
            answer_relevancy REAL,
            context_precision REAL,
            context_recall REAL,
            verdict TEXT,
            FOREIGN KEY (run_id) REFERENCES eval_runs(id)
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialised.")

def save_run(df, questions, answers, ground_truths):
    """Save a complete evaluation run to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Calculate averages — handle NaN values
    avg_faith = df["faithfulness"].mean()
    avg_rel   = df["answer_relevancy"].mean()
    avg_prec  = df["context_precision"].mean()
    avg_rec   = df["context_recall"].mean()
    avg_overall = df[["faithfulness", "answer_relevancy",
                       "context_precision", "context_recall"]].mean().mean()

    # Insert run summary
    cursor.execute("""
        INSERT INTO eval_runs (
            run_timestamp, total_questions,
            avg_faithfulness, avg_answer_relevancy,
            avg_context_precision, avg_context_recall, avg_overall
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        len(questions),
        float(avg_faith),
        float(avg_rel),
        float(avg_prec),
        float(avg_rec),
        float(avg_overall),
    ))

    run_id = cursor.lastrowid

    # Insert individual question scores
    for i, row in df.iterrows():
        cursor.execute("""
            INSERT INTO eval_questions (
                run_id, question, answer, ground_truth,
                faithfulness, answer_relevancy,
                context_precision, context_recall, verdict
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            questions[i],
            answers[i],
            ground_truths[i],
            float(row["faithfulness"]) if not str(row["faithfulness"]) == "nan" else None,
            float(row["answer_relevancy"]),
            float(row["context_precision"]),
            float(row["context_recall"]),
            "auto"
        ))

    conn.commit()
    conn.close()
    print(f"Run {run_id} saved to database.")
    return run_id

def load_all_runs():
    """Load all evaluation runs for dashboard."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM eval_runs ORDER BY run_timestamp
    """)
    runs = cursor.fetchall()
    conn.close()
    return runs

def load_run_questions(run_id):
    """Load all questions for a specific run."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM eval_questions WHERE run_id = ?
    """, (run_id,))
    questions = cursor.fetchall()
    conn.close()
    return questions