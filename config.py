import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # HyperCLOVA X (Main RAG Agent)
    CLOVANSTUDIO_API_KEY = os.getenv("CLOVANSTUDIO_API_KEY")

    # RAG Models (HCX)
    HCX_MODEL_LIGHT = os.getenv(
        "HCX_MODEL_LIGHT", "HCX-DASH-002"
    )  # Fast (Extraction, Routing)
    HCX_MODEL_HEAVY = os.getenv(
        "HCX_MODEL_HEAVY", "HCX-003"
    )  # Smart (Reasoning, Judgment)

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

    # Neo4j
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    # Data
    DATA_PATH = os.getenv("DATA_PATH", "data_v10.json")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Feature Flags (Lego Switches)
    ENABLE_NEO4J = os.getenv("ENABLE_NEO4J", "false").lower() == "true"
    ENABLE_SOP = os.getenv("ENABLE_SOP", "true").lower() == "true"
    ENABLE_ADVERSARIAL = os.getenv("ENABLE_ADVERSARIAL", "true").lower() == "true"
    ENABLE_REDIS = os.getenv("ENABLE_REDIS", "true").lower() == "true"
