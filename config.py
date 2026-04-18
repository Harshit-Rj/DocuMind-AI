"""
Configuration file for DocuMind RAG application.
Centralized settings for model names, paths, and chunk sizes.
"""

import os

# ============================================================================
# PATHS
# ============================================================================
DATA_PATH = "documents/"
VECTOR_STORE_PATH = "vector_store/"
VECTOR_STORE_INDEX_FILE = "index.faiss"

# Folders to ensure exist
REQUIRED_FOLDERS = [DATA_PATH, VECTOR_STORE_PATH]

# ============================================================================
# EMBEDDING MODEL
# ============================================================================
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ============================================================================
# GENERATION MODEL
# ============================================================================
GENERATION_MODEL_NAME = "google/flan-t5-base"
GENERATION_DEVICE = "cuda" if os.environ.get("USE_CUDA", "false").lower() == "true" else "cpu"

# ============================================================================
# CHUNKING PARAMETERS
# ============================================================================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ============================================================================
# RETRIEVAL PARAMETERS
# ============================================================================
RETRIEVAL_SEARCH_TYPE = "mmr"  # Options: "similarity", "mmr"
RETRIEVAL_K = 5  # Number of relevant documents to retrieve
RETRIEVAL_FETCH_K = 10  # Number of documents to fetch before MMR filtering

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================
GENERATION_MAX_LENGTH = 300
GENERATION_NUM_BEAMS = 4

# ============================================================================
# CHAT HISTORY
# ============================================================================
CHAT_HISTORY_LOOKBACK = 5  # Number of previous messages to include in context

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================
EVALUATION_GOOD_THRESHOLD = 0.8
EVALUATION_AVERAGE_THRESHOLD = 0.6

# ============================================================================
# TOKENIZER PARAMETERS
# ============================================================================
TOKENIZER_MAX_LENGTH = 512
TOKENIZER_TRUNCATION = True

# ============================================================================
# FAISS PARAMETERS
# ============================================================================
FAISS_ALLOW_DANGEROUS_DESERIALIZATION = True
