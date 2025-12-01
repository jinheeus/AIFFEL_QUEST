import json
import os
import sys
# Add root directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_naver import ClovaXEmbeddings
from pymilvus import MilvusClient, DataType
from tqdm import tqdm
from config import Config

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def prepare_documents(data):
    # Chunking Strategy: 500 ~ 800 tokens (approx 2000-3000 chars for Korean?)
    # User said "tokens", but RecursiveCharacterTextSplitter works on characters.
    # Assuming 1 token ~= 2-3 chars for Korean/English mix, 800 tokens ~ 2000 chars.
    # Let's stick to Config.CHUNK_SIZE which we set to 800.
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    documents = []
    print(f"Processing {len(data)} records for V0 Ingestion...")
    
    for loop_idx, item in enumerate(tqdm(data)):
        contents = item.get('contents', '')
        
        # Extract metadata from contents_summary
        title = ""
        problem = ""
        
        summary_raw = item.get('contents_summary')
        if isinstance(summary_raw, str):
            try:
                summary_json = json.loads(summary_raw)
                title = summary_json.get('title', '')
                problem = summary_json.get('problems', '')
            except:
                pass
        elif isinstance(summary_raw, dict):
            title = summary_raw.get('title', '')
            problem = summary_raw.get('problems', '')
            
        # Fallback to root 'problem' if not found in summary (optional, but good for safety)
        if not problem:
            problem = item.get('problem', '')

        merged_text = f"Problem: {problem}\n\nContents: {contents}"
        
        # Create chunks
        chunks = text_splitter.create_documents([merged_text])
        
        for chunk in chunks:
            # V0 Spec: id, vector, text, title. NO Category Metadata.
            doc_record = {
                "text": chunk.page_content,
                "title": title,
                "idx": str(item.get('idx', loop_idx)) # Use idx as doc_code (fallback to loop index)
            }
            documents.append(doc_record)
            
    return documents

def ingest_to_milvus(documents):
    print(f"Initializing Milvus Client for V0 (Collection: {Config.MILVUS_COLLECTION_NAME_V0})...")
    client = MilvusClient(
        uri=Config.MILVUS_URI,
        token=Config.MILVUS_TOKEN
    )
    
    collection_name = Config.MILVUS_COLLECTION_NAME_V0
    
    if client.has_collection(collection_name):
        print(f"Dropping existing collection: {collection_name}")
        client.drop_collection(collection_name)
        
    print(f"Creating collection: {collection_name}")
    # Schema Definition for V0
    # id: Int64 (PK)
    # vector: FloatVector (Dim: 1024)
    # text: VarChar
    # title: VarChar
    
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024) # bge-m3 dim
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="idx", datatype=DataType.VARCHAR, max_length=128) # Added doc_code
    
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector", 
        index_type="AUTOINDEX", 
        metric_type="IP" # Inner Product recommended for normalized vectors (bge-m3)
    )
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    
    print("Generating embeddings and inserting...")
    # HyperCLOVA X Embeddings
    embeddings_model = ClovaXEmbeddings(model=Config.EMBEDDING_MODEL) 
    
    batch_size = 50
    total_docs = len(documents)
    
    for i in tqdm(range(0, total_docs, batch_size)):
        batch_docs = documents[i : i + batch_size]
        batch_texts = [doc['text'] for doc in batch_docs]
        
        try:
            # Generate embeddings
            vectors = embeddings_model.embed_documents(batch_texts)
            
            # Prepare data for insertion
            insert_data = []
            for doc, vector in zip(batch_docs, vectors):
                record = {
                    "vector": vector,
                    "text": doc['text'],
                    "title": doc['title'],
                    "idx": str(doc['idx']) # Ensure string
                    # id is auto-generated
                }
                insert_data.append(record)
                
            client.insert(
                collection_name=collection_name,
                data=insert_data
            )
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            
    print(f"Successfully inserted {total_docs} chunks into {collection_name}")

if __name__ == "__main__":
    try:
        data = load_data(Config.DATA_PATH)
        docs = prepare_documents(data)
        ingest_to_milvus(docs)
    except Exception as e:
        print(f"Error: {e}")
