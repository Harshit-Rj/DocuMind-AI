from backend.retriever import get_retriever

# ✅ NEW IMPORTS (replace OpenAI)
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from config import (
    GENERATION_MODEL_NAME,
    GENERATION_DEVICE,
    GENERATION_MAX_LENGTH,
    GENERATION_NUM_BEAMS,
    CHAT_HISTORY_LOOKBACK,
    TOKENIZER_MAX_LENGTH,
    TOKENIZER_TRUNCATION,
)

# Simple chat history storage
chat_history = []

# ✅ Initialize HuggingFace T5 model properly
try:
    device = GENERATION_DEVICE
    tokenizer = T5Tokenizer.from_pretrained(GENERATION_MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(GENERATION_MODEL_NAME).to(device)
except Exception as e:
    raise RuntimeError(f"Failed to load HuggingFace model: {e}")


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
        docs = retriever.invoke(query)
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
        # Format chat history
        history_text = "\n".join([
            f"Q: {q}\nA: {a}" for q, a in chat_history[-CHAT_HISTORY_LOOKBACK:]
        ])

        prompt = f"""
You are a helpful assistant.
Answer ONLY from the provided context.
If answer is not in context, say "I don't know".

Context:
{context}

Chat History:
{history_text}

Question:
{query}

Also mention sources at the end.
"""
    except Exception as e:
        raise RuntimeError(f"Failed to build prompt: {e}")

    try:
        # ✅ T5 inference (proper sequence-to-sequence generation)
        input_ids = tokenizer(prompt, return_tensors="pt", max_length=TOKENIZER_MAX_LENGTH, truncation=TOKENIZER_TRUNCATION).input_ids.to(device)
        output_ids = model.generate(input_ids, max_length=GENERATION_MAX_LENGTH, num_beams=GENERATION_NUM_BEAMS, early_stopping=True)
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        
        if not answer:
            answer = "I don't know."

    except Exception as e:
        raise RuntimeError(f"Failed to generate response: {e}")

    try:
        # Save history
        chat_history.append((query, answer))
        return answer, sources
    except Exception as e:
        raise RuntimeError(f"Failed to save chat history: {e}")