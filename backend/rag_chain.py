from langchain_openai import ChatOpenAI
from backend.retriever import get_retriever

# Simple chat history storage
chat_history = []

def ask(query, filter_source=None):
    retriever = get_retriever(filter_source)
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata["source"] for doc in docs]))

    llm = ChatOpenAI(model="gpt-4o-mini")

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

    # Add to history
    chat_history.append((query, response.content))

    return response.content, sources