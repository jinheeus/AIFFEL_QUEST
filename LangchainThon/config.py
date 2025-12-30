# 공통 설정/토글

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")
    docs_dir: str = os.getenv("DOCS_DIR", "./data/pdfs")

    # Retriever knobs (P1)
    use_mmr: bool = False
    score_threshold: Optional[float] = None   # e.g., 0.25
    top_k: int = 4

    # 🔹 LangSmith dataset (여기 한 줄 추가)
    langsmith_dataset_id: str = os.getenv("LANGSMITH_DATASET_ID", "")
    
settings = Settings()