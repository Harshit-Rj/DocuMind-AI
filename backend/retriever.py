import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

DB_PATH = "vector_store/"
INDEX_FILE = "index.faiss"


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
    _ensure_vector_store_exists()

    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    search_kwargs = {
        "k": 5,
        "fetch_k": 10
    }

    if filter_source:
        search_kwargs["filter"] = {"source": filter_source}

    retriever = db.as_retriever(
        search_type="mmr",   # 🔥 better than similarity
        search_kwargs=search_kwargs
    )

    return retriever