import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    VECTOR_STORE_PATH,
    VECTOR_STORE_INDEX_FILE,
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_SEARCH_TYPE,
    RETRIEVAL_K,
    RETRIEVAL_FETCH_K,
    FAISS_ALLOW_DANGEROUS_DESERIALIZATION,
)

DB_PATH = VECTOR_STORE_PATH
INDEX_FILE = VECTOR_STORE_INDEX_FILE


def _ensure_vector_store_exists():
    index_path = os.path.join(DB_PATH, INDEX_FILE)
    if not os.path.exists(index_path):
        raise RuntimeError(
            "Vector store not found. Create it by running:\n"
            "    python backend/ingest.py\n"
            "after placing your documents in the documents/ directory.\n"
            "Expected file: vector_store/index.faiss"
        )


def get_retriever(filter_source=None):
    try:
        _ensure_vector_store_exists()
    except RuntimeError as e:
        raise RuntimeError(f"Vector store check failed: {e}")

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embeddings: {e}")
    
    try:
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=FAISS_ALLOW_DANGEROUS_DESERIALIZATION)
    except FileNotFoundError as e:
        raise RuntimeError(f"Vector store files not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load vector store: {e}")

    try:
        search_kwargs = {
            "k": RETRIEVAL_K,
            "fetch_k": RETRIEVAL_FETCH_K
        }

        if filter_source:
            search_kwargs["filter"] = {"source": filter_source}

        retriever = db.as_retriever(
            search_type=RETRIEVAL_SEARCH_TYPE,
            search_kwargs=search_kwargs
        )
        
        return retriever
    except Exception as e:
        raise RuntimeError(f"Failed to create retriever: {e}")