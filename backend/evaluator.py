from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    EMBEDDING_MODEL_NAME,
    EVALUATION_GOOD_THRESHOLD,
    EVALUATION_AVERAGE_THRESHOLD,
)

def cosine_score(answer, ground_truth):
    try:
        if not answer or not ground_truth:
            raise ValueError("Answer and ground_truth cannot be empty")
    except ValueError as e:
        raise RuntimeError(f"Invalid input: {e}")
    
    try:
        emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embeddings: {e}")
    
    try:
        v1 = emb.embed_query(answer)
        v2 = emb.embed_query(ground_truth)
    except Exception as e:
        raise RuntimeError(f"Failed to create embeddings: {e}")

    try:
        score = cosine_similarity([v1], [v2])[0][0]
        return float(score)
    except Exception as e:
        raise RuntimeError(f"Failed to compute cosine similarity: {e}")


def simple_eval(answer, ground_truth):
    try:
        score = cosine_score(answer, ground_truth)
    except RuntimeError as e:
        raise RuntimeError(f"Evaluation failed: {e}")

    try:
        if score > EVALUATION_GOOD_THRESHOLD:
            return "Good"
        elif score > EVALUATION_AVERAGE_THRESHOLD:
            return "Average"
        else:
            return "Poor"
    except Exception as e:
        raise RuntimeError(f"Failed to evaluate score: {e}")