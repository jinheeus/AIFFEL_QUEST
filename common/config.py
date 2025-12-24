import os
from dotenv import load_dotenv

# Load .env from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
load_dotenv(os.path.join(project_root, ".env"))


class Config:
    # HyperCLOVA X (Main RAG Agent)
    CLOVASTUDIO_API_KEY = os.getenv("CLOVASTUDIO_API_KEY")

    # RAG Models (HCX)
    HCX_MODEL_LIGHT = os.getenv(
        "HCX_MODEL_LIGHT", "HCX-DASH-002"
    )  # Fast (Extraction, Routing)
    HCX_MODEL_STANDARD = os.getenv(
        "HCX_MODEL_STANDARD", "HCX-003"
    )  # Balanced (Writer, Router)
    HCX_MODEL_REASONING = os.getenv(
        "HCX_MODEL_REASONING", "HCX-007"
    )  # Deep Thinking (Analyst, SOP)

    # Alias for backward compatibility (points to STANDARD by default)
    HCX_MODEL_HEAVY = HCX_MODEL_STANDARD

    # Legacy Support (will be deprecated in favor of specific light/heavy usage)
    LLM_MODEL = HCX_MODEL_LIGHT

    # Evaluation Models (Toggle: 'gemini' or 'openai')
    EVAL_PROVIDER = os.getenv("EVAL_PROVIDER", "openai")  # or 'gemini'

    # Gemini Config
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL_LIGHT = "gemini-2.5-flash-lite"
    GEMINI_MODEL_HEAVY = "gemini-2.5-pro"

    # OpenAI Config
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_LIGHT = "gpt-4o-mini"
    OPENAI_MODEL_HEAVY = "gpt-4o"

    # Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

    # Milvus
    MILVUS_URI = os.getenv("MILVUS_URI", "./milvus_demo.db")
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
    MILVUS_COLLECTION_NAME_V0 = "data_v1"
    MILVUS_COLLECTION_NAME_V1 = "data_v2"
    MILVUS_COLLECTION_NAME_MARKDOWN = "markdown_rag_parent_child_v1"

    # Data
    DATA_PATH = os.getenv("DATA_PATH", "data_v10.json")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Feature Flags (Lego Switches)
    ENABLE_REDIS = os.getenv("ENABLE_REDIS", "true").lower() == "true"

    # RAG Versioning
    ACTIVE_RAG_DIR = os.getenv("ACTIVE_RAG_DIR", "agentic_rag_v2") # or "agentic_rag_v1"
