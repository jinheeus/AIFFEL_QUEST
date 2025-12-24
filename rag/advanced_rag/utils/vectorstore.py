import os
import asyncio
import pickle
import logging
import warnings
from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from kiwipiepy import Kiwi
from utils.embedding import embeddings_hcx

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

logging.getLogger("pymilvus").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pymilvus")

load_dotenv()

kiwi = Kiwi()

def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text) if token.tag.startswith('N')]

def get_documents_from_milvus(vector_db):
    collection = vector_db.col
    docs = []
    batch_size = 1000
    total_limit = 16384
    offset = 0
    
    print(f"[ INFO ] LOADING DOCUMENTS FROM MILVUS IN BATCHES OF {batch_size}...")

    while offset < total_limit:
        current_limit = min(batch_size, total_limit - offset)
        if current_limit <= 0:
            break
            
        try:
            res = collection.query(
                expr="", 
                output_fields=["text", "idx"], 
                offset=offset,
                limit=current_limit
            )
            
            if not res:
                break
                
            for item in res:
                text = item.get('text')
                idx = item.get('idx')
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={"idx": idx} 
                    ))
            
            offset += len(res)
            
            if len(res) < current_limit:
                break
                
        except Exception as e:
            print(f"[ ERROR ] FETCHING BATCH AT OFFSET {offset}: {e}")
            break
            
    print(f"[ SUCCESS ] LOADED {len(docs)} DOCUMENTS.")
    return docs

uri = os.getenv("MILVUS_URI")
token = os.getenv("MILVUS_TOKEN")

db_configs = ["title", "standards", "outline", "problems", "opinion", "criteria", "action", "documents"]
vector_dbs = {}
retrievers = {}

for name in db_configs:
    db = Milvus(
        embedding_function=embeddings_hcx,
        connection_args={"uri": uri, "token": token},
        collection_name=name,
        auto_id=True
    )
    vector_dbs[name] = db
    if name != "documents":
        retrievers[name] = db.as_retriever(search_kwargs={"k": 20})

documents_db = vector_dbs["documents"]

CACHE_FILE = "bm25_retriever.pkl"

if os.path.exists(CACHE_FILE):
    print(f"[ INFO ] LOADED CACHED BM25 RETRIEVER FROM {CACHE_FILE}")
    with open(CACHE_FILE, "rb") as f:
        global_bm25_retriever = pickle.load(f)
else:
    print("[ INFO ] INITIALIZING BM25 (FIRST RUN TAKES TIME)...")
    raw_docs_all = get_documents_from_milvus(documents_db)
    
    global_bm25_retriever = BM25Retriever.from_documents(
        raw_docs_all, 
        preprocess_func=kiwi_tokenize
    )
    global_bm25_retriever.k = 20
    
    print(f"[ INFO ] SAVING BM25 RETRIEVER TO {CACHE_FILE}...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(global_bm25_retriever, f)
    print("[ INFO ] BM25 SAVED.")