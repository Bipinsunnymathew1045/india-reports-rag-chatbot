import streamlit as st
from query import ask

# ── Page configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="India Gov Reports Chatbot",
    page_icon="📚",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────────────
st.title("📚 India Education & Health Reports Chatbot")
st.caption("Powered by RAG — answers grounded in official Government "
           "of India reports 2020–2025 · Page numbers refer to PDF "
           "page positions and may vary slightly from printed pages")


# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Reports loaded")
    st.write("**Education:** 2020-21 · 2021-22 · 2022-23 · 2023-24 · 2024-25")
    st.write("**Health:** 2020-21 · 2021-22 · 2022-23 · 2023-24 · 2024-25")
    st.divider()

    st.header("💡 Try asking")
    questions = [
        "What was the GER in higher education in 2022?",
        "How many Ayushman Bharat cards were created?",
        "What is the budget for school education?",
        "How did dropout rates change over the years?",
        "What is the National Education Policy about?",
        "How many AIIMS hospitals exist in India?",
    ]
    for q in questions:
        # Clicking a suggestion sends it as a question
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    st.caption("Built with LangChain · OpenAI · FAISS · Streamlit")

# ── Chat history ───────────────────────────────────────────────────
# session_state persists data across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📄 Sources"):
                for citation in message["citations"]:
                    st.write(f"• {citation}")

# ── Handle sidebar button clicks ───────────────────────────────────
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

    # Show user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Get and show answer
    with st.chat_message("assistant"):
        with st.spinner("Searching through 3,492 pages..."):
            result = ask(question)
        st.write(result["answer"])
        with st.expander("📄 Sources"):
            for citation in result["citations"]:
                st.write(f"• {citation}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": result["citations"]
    })
    st.rerun()

# ── Chat input ─────────────────────────────────────────────────────
if question := st.chat_input("Ask anything about India's education or health reports..."):

    # Show user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Get and show answer
    with st.chat_message("assistant"):
        with st.spinner("Searching through 3,492 pages..."):
            result = ask(question)
        st.write(result["answer"])
        with st.expander("📄 Sources"):
            for citation in result["citations"]:
                st.write(f"• {citation}")
         
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": result["citations"]
    })