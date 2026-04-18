from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_openai import OpenAIEmbeddings

def cosine_score(answer, ground_truth):
    emb = OpenAIEmbeddings()
    v1 = emb.embed_query(answer)
    v2 = emb.embed_query(ground_truth)

    return cosine_similarity([v1], [v2])[0][0]


def simple_eval(answer, ground_truth):
    score = cosine_score(answer, ground_truth)

    if score > 0.8:
        return "Good"
    elif score > 0.6:
        return "Average"
    else:
        return "Poor"