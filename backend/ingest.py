import os
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

_ensure_folders_exist()

DATA_PATH = "documents/"
DB_PATH = "vector_store/"


def _ensure_folders_exist():
    folders = [DATA_PATH.rstrip("/"), DB_PATH.rstrip("/")]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)


def _validate_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Add a valid key to .env and rerun:\n"
            "    OPENAI_API_KEY=sk-...\n"
            "Do not commit your API key to source control."
        )

def get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_documents():
    docs = []

    for file in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue

        loaded_docs = loader.load()

        # Add metadata
        for d in loaded_docs:
            d.metadata["source"] = file
            d.metadata["file_hash"] = get_file_hash(path)

        docs.extend(loaded_docs)

    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,     # tuned
        chunk_overlap=150   # tuned
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    _validate_openai_api_key()
    embeddings = OpenAIEmbeddings()

    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH, exist_ok=True)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks)

    print("✅ Improved Vector DB ready")