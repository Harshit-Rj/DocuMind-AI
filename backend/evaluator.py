from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_openai import OpenAIEmbeddings

def cosine_score(answer, ground_truth):
    try:
        if not answer or not ground_truth:
            raise ValueError("Answer and ground_truth cannot be empty")
    except ValueError as e:
        raise RuntimeError(f"Invalid input: {e}")
    
    try:
        emb = OpenAIEmbeddings()
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
        if score > 0.8:
            return "Good"
        elif score > 0.6:
            return "Average"
        else:
            return "Poor"
    except Exception as e:
        raise RuntimeError(f"Failed to evaluate score: {e}")