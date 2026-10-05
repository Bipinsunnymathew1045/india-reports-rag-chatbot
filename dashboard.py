import streamlit as st
import sqlite3
import pandas as pd
from store import load_all_runs, load_run_questions, DB_PATH

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RAG Evaluation Dashboard")
st.caption("Tracking chatbot quality across evaluation runs")

# ── Check database exists ──────────────────────────────────────────
try:
    conn = sqlite3.connect(DB_PATH)
    runs_df = pd.read_sql("SELECT * FROM eval_runs", conn)
    conn.close()
except Exception:
    st.error("No evaluation data found. Run eval.py first.")
    st.stop()

if runs_df.empty:
    st.warning("No evaluation runs found. Run eval.py to generate data.")
    st.stop()

# ── Format timestamps ──────────────────────────────────────────────
runs_df["run_timestamp"] = pd.to_datetime(runs_df["run_timestamp"])
runs_df["run_label"] = runs_df["run_timestamp"].dt.strftime("%b %d %H:%M")

# ── Summary metrics — latest run ───────────────────────────────────
st.subheader("Latest Run Summary")
latest = runs_df.iloc[-1]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Overall",            f"{latest['avg_overall']:.3f}")
col2.metric("Faithfulness",       f"{latest['avg_faithfulness']:.3f}")
col3.metric("Answer Relevancy",   f"{latest['avg_answer_relevancy']:.3f}")
col4.metric("Context Precision",  f"{latest['avg_context_precision']:.3f}")
col5.metric("Context Recall",     f"{latest['avg_context_recall']:.3f}")

# ── Trend chart ────────────────────────────────────────────────────
st.divider()
st.subheader("Score Trends Across Runs")

if len(runs_df) > 1:
    chart_df = runs_df.set_index("run_label")[[
        "avg_faithfulness",
        "avg_answer_relevancy",
        "avg_context_precision",
        "avg_context_recall"
    ]].rename(columns={
        "avg_faithfulness":      "Faithfulness",
        "avg_answer_relevancy":  "Answer Relevancy",
        "avg_context_precision": "Context Precision",
        "avg_context_recall":    "Context Recall"
    })
    st.line_chart(chart_df)
else:
    st.info("Run eval.py at least twice to see trend charts.")

# ── All runs table ─────────────────────────────────────────────────
st.divider()
st.subheader("All Evaluation Runs")

display_df = runs_df[[
    "id", "run_label", "total_questions",
    "avg_overall", "avg_faithfulness",
    "avg_answer_relevancy", "avg_context_precision",
    "avg_context_recall"
]].rename(columns={
    "id":                    "Run",
    "run_label":             "Timestamp",
    "total_questions":       "Questions",
    "avg_overall":           "Overall",
    "avg_faithfulness":      "Faithfulness",
    "avg_answer_relevancy":  "Relevancy",
    "avg_context_precision": "Precision",
    "avg_context_recall":    "Recall"
})

st.dataframe(
    display_df.style.format({
        "Overall":      "{:.3f}",
        "Faithfulness": "{:.3f}",
        "Relevancy":    "{:.3f}",
        "Precision":    "{:.3f}",
        "Recall":       "{:.3f}"
    }),
    use_container_width=True
)

# ── Per question breakdown ─────────────────────────────────────────
st.divider()
st.subheader("Question-Level Breakdown")

selected_run = st.selectbox(
    "Select run to inspect:",
    options=runs_df["id"].tolist(),
    format_func=lambda x: f"Run {x} — {runs_df[runs_df['id']==x]['run_label'].values[0]}"
)

if selected_run:
    conn = sqlite3.connect(DB_PATH)
    q_df = pd.read_sql(
        "SELECT * FROM eval_questions WHERE run_id = ?",
        conn, params=(selected_run,)
    )
    conn.close()

    for _, row in q_df.iterrows():
        with st.expander(f"Q: {row['question'][:80]}..."):
            st.write("**Answer:**", row["answer"])
            st.write("**Ground Truth:**", row["ground_truth"])
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Faithfulness",      f"{row['faithfulness']:.3f}" if row['faithfulness'] else "N/A")
            col2.metric("Answer Relevancy",  f"{row['answer_relevancy']:.3f}")
            col3.metric("Context Precision", f"{row['context_precision']:.3f}")
            col4.metric("Context Recall",    f"{row['context_recall']:.3f}")