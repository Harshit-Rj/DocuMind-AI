"""Custom styling for DocuMind Streamlit app"""

# Custom CSS for the application
CUSTOM_CSS = """
    <style>
        [data-testid="stChatMessageContent"] p {
            margin: 0;
        }
        .chat-container {
            max-width: 100%;
        }
        .sources-container {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 0.9em;
        }
    </style>
"""

def apply_custom_styling(st):
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
