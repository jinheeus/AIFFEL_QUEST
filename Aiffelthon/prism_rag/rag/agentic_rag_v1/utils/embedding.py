import os
import sys

# Ensure project root is in path to import common
current_dir = os.path.dirname(os.path.abspath(__file__))
v1_root = os.path.dirname(current_dir)
rag_root = os.path.dirname(v1_root)
project_root = os.path.dirname(rag_root)
if project_root not in sys.path:
    sys.path.append(project_root)

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_naver import ClovaXEmbeddings
from common.config import Config

load_dotenv()

embeddings_openai = OpenAIEmbeddings(
    model="text-embedding-3-small", openai_api_key=Config.OPENAI_API_KEY
)

embeddings_hcx = ClovaXEmbeddings(
    model=Config.EMBEDDING_MODEL, api_key=Config.CLOVASTUDIO_API_KEY
)
