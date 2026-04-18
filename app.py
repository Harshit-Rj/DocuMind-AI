import os
import streamlit as st
from dotenv import load_dotenv
from backend.rag_chain import ask
from styles.custom_styles import apply_custom_styling
from config import REQUIRED_FOLDERS

load_dotenv()

def _ensure_folders_exist():
    try:
        for folder in REQUIRED_FOLDERS:
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
    except OSError as e:
        st.error(f"❌ Failed to create folders: {e}")
        st.stop()

_ensure_folders_exist()

# Page configuration
st.set_page_config(
    page_title="DocuMind - AI Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
apply_custom_styling(st)

# Header
st.title("📄 DocuMind AI Chat")
st.markdown("Ask questions about your documents using AI-powered search")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    filter_doc = st.text_input(
        "Filter by document",
        placeholder="Optional: filter responses by document name",
        help="Leave empty to search all documents"
    )
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat = []
        st.success("Chat history cleared!")
        st.rerun()
    
    st.divider()
    
    st.markdown("### About")
    st.markdown("""
    **DocuMind** is a RAG chatbot that uses AI to answer questions about your documents.
    
    - 📚 Upload PDF and TXT files
    - 🔍 Semantic search through documents
    - 💬 Natural conversation with AI
    """)

# Initialize session state
if "chat" not in st.session_state:
    st.session_state.chat = []

# Chat display container
st.markdown("---")
chat_container = st.container()

with chat_container:
    if st.session_state.chat:
        for i, (query, answer, sources) in enumerate(st.session_state.chat):
            # User message
            with st.chat_message("user", avatar="👤"):
                st.write(query)
            
            # Assistant message
            with st.chat_message("assistant", avatar="🤖"):
                st.write(answer)
                
                # Sources display
                if sources:
                    with st.expander("📚 View Sources"):
                        for source in sources:
                            st.write(f"• {source}")
    else:
        st.info("💡 Start a conversation by asking a question about your documents!")

st.markdown("---")

# Input area at the bottom
col1, col2 = st.columns([5, 1])

with col1:
    query = st.text_input(
        "Message",
        placeholder="Ask a question about your documents...",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send", use_container_width=True, type="primary")

# Handle message submission
if send_button or (query and st.session_state.get("just_pressed")):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("🤔 Thinking..."):
                answer, sources = ask(query, filter_doc)
            st.session_state.chat.append((query, answer, sources))
            st.rerun()
        except RuntimeError as e:
            st.error(f"❌ Error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")