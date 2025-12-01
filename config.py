import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # HyperCLOVA X
    CLOVANSTUDIO_API_KEY = os.getenv("CLOVANSTUDIO_API_KEY")
    
    # Models
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
    LLM_MODEL = os.getenv("LLM_MODEL", "HCX-DASH-002")

    # Milvus
    MILVUS_URI = os.getenv("MILVUS_URI", "./milvus_demo.db")
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
    MILVUS_COLLECTION_NAME_V0 = "audit_reports_v0"
    MILVUS_COLLECTION_NAME_V1 = "audit_reports_v1"
    
    # Neo4j
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # Data
    DATA_PATH = os.getenv("DATA_PATH", "data_v10.json")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
