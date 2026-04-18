from langchain_openai import ChatOpenAI
from backend.retriever import get_retriever

# Simple chat history storage
chat_history = []

def ask(query, filter_source=None):
    try:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
    except ValueError as e:
        raise RuntimeError(f"Invalid query: {e}")

    try:
        retriever = get_retriever(filter_source)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to get retriever: {e}")
    
    try:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            raise ValueError("No relevant documents found")
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve documents: {e}")

    try:
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
    except Exception as e:
        raise RuntimeError(f"Failed to process documents: {e}")

    try:
        llm = ChatOpenAI(model="gpt-4o-mini")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM: {e}")

    try:
        # Format chat history
        history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history[-5:]])  # Last 5 exchanges

        prompt = f"""
    You are a helpful assistant.
    Answer ONLY from the provided context.

    Context:
    {context}

    Chat History:
    {history_text}

    Question:
    {query}

    Also mention sources at the end.
    """

        response = llm.invoke(prompt)
        
        if not response or not response.content:
            raise ValueError("Empty response from LLM")
    except Exception as e:
        raise RuntimeError(f"Failed to generate response: {e}")

    try:
        # Add to history
        chat_history.append((query, response.content))
        return response.content, sources
    except Exception as e:
        raise RuntimeError(f"Failed to save chat history: {e}")