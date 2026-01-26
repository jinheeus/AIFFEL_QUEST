import os
import logging
import warnings
from dotenv import load_dotenv
from langchain_milvus import Milvus
from utils.embedding import embeddings_hcx

logging.getLogger("pymilvus").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pymilvus")

load_dotenv()

uri = os.getenv("MILVUS_URI")
token = os.getenv("MILVUS_TOKEN")

vector_db = Milvus(
    embedding_function=embeddings_hcx,
    connection_args={"uri": uri, "token": token},
    collection_name="data",
    auto_id=True
)

documents_db = Milvus(
    embedding_function=embeddings_hcx,
    connection_args={"uri": uri, "token": token},
    collection_name="documents",
    auto_id=True
)