import os
import hashlib
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ✅ REPLACED OpenAI WITH HUGGINGFACE
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    DATA_PATH,
    VECTOR_STORE_PATH,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    REQUIRED_FOLDERS,
)

load_dotenv()

DB_PATH = VECTOR_STORE_PATH


def _ensure_folders_exist():
    folders = [folder.rstrip("/") for folder in REQUIRED_FOLDERS]
    for folder in folders:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError as e:
                raise RuntimeError(f"Failed to create folder '{folder}': {e}")


def get_file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {path}")
    except IOError as e:
        raise RuntimeError(f"Failed to read file '{path}': {e}")

def load_documents():
    docs = []

    if not os.path.exists(DATA_PATH):
        raise RuntimeError(f"Documents folder not found: {DATA_PATH}")
    
    files = os.listdir(DATA_PATH)
    if not files:
        raise RuntimeError(f"No files found in {DATA_PATH}")

    for file in files:
        path = os.path.join(DATA_PATH, file)

        if not os.path.isfile(path):
            continue

        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.endswith(".txt"):
                loader = TextLoader(path)
            else:
                continue

            loaded_docs = loader.load()
            
            if not loaded_docs:
                print(f"⚠️ Warning: No content extracted from {file}")
                continue

            # Add metadata
            for d in loaded_docs:
                d.metadata["source"] = file
                d.metadata["file_hash"] = get_file_hash(path)

            docs.extend(loaded_docs)
            print(f"✅ Loaded {len(loaded_docs)} pages from {file}")

        except Exception as e:
            raise RuntimeError(f"Failed to load document '{file}': {e}")

    if not docs:
        raise RuntimeError(f"No documents could be loaded from {DATA_PATH}")
    
    return docs


def split_documents(documents):
    try:
        if not documents:
            raise ValueError("No documents provided to split")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No chunks were created from documents")
        
        print(f"✅ Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        raise RuntimeError(f"Failed to split documents: {e}")


def create_vector_store(chunks):
    if not chunks:
        raise ValueError("No chunks provided to create vector store")
    
    try:
        print("🔄 Creating embeddings using SentenceTransformers...")
        # ✅ FREE LOCAL MODEL
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )

    except Exception as e:
        raise RuntimeError(f"Failed to initialize embeddings: {e}")

    try:
        if not os.path.exists(DB_PATH):
            os.makedirs(DB_PATH, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create vector store directory '{DB_PATH}': {e}")

    try:
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(DB_PATH)
        print(f"✅ Vector store created successfully with {len(chunks)} chunks")
    except Exception as e:
        raise RuntimeError(f"Failed to create or save FAISS index: {e}")


if __name__ == "__main__":
    try:
        print("🚀 Starting document ingestion process...\n")
        
        _ensure_folders_exist()

        print("📂 Loading documents...")
        docs = load_documents()
        print(f"✅ Loaded {len(docs)} pages\n")

        print("✂️  Splitting documents into chunks...")
        chunks = split_documents(docs)
        print(f"✅ Created {len(chunks)} chunks\n")

        print("🔧 Creating vector store...")
        create_vector_store(chunks)

        print("\n✅ Vector DB ready (FREE MODE)!")

    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        exit(1)
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)