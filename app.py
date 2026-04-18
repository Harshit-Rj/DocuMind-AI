import os
import streamlit as st
from dotenv import load_dotenv
from backend.rag_chain import ask

load_dotenv()


def _validate_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "Missing OPENAI_API_KEY in .env.\n"
            "Add a valid OpenAI key and restart the app."
        )
        st.stop()


_validate_openai_api_key()

st.set_page_config(page_title="RAG Chatbot")

st.title("📄 RAG Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.text_input("Ask something")

filter_doc = st.text_input("Filter by document (optional)")

if st.button("Ask"):
    answer, sources = ask(query, filter_doc)

    st.session_state.chat.append((query, answer, sources))

for q, a, s in st.session_state.chat:
    st.write(f"**Q:** {q}")
    st.write(f"**A:** {a}")
    st.write(f"📚 Sources: {s}")
    st.write("---")