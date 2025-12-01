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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    documents = []
    print(f"Processing {len(data)} records for V1 Ingestion...")
    
    for loop_idx, item in enumerate(tqdm(data)):
        contents = item.get('contents', '')
        
        # Metadata Extraction
        title = ""
        problem = ""
        cat_L1 = "기타" # Default
        cat_L2 = "기타" # Default
        
        summary_raw = item.get('contents_summary')
        if isinstance(summary_raw, str):
            try:
                summary_json = json.loads(summary_raw)
                title = summary_json.get('title', '')
                problem = summary_json.get('problems', '')
                
                # Extract Category
                raw_cat = summary_json.get('cat', '')
                if raw_cat:
                    # Handle multi-values: "특정사안, 공직기강" -> "특정사안"
                    cat_L1 = raw_cat.split(',')[0].strip()
                
                # User spec says "cat_L2 from contents_summary.sub_cat"
                raw_sub_cat = summary_json.get('sub_cat', '')
                if raw_sub_cat:
                    cat_L2 = raw_sub_cat.split(',')[0].strip()
                    
            except:
                pass
        elif isinstance(summary_raw, dict):
             title = summary_raw.get('title', '')
             problem = summary_raw.get('problems', '')
             
             raw_cat = summary_raw.get('cat', '')
             if raw_cat:
                 cat_L1 = raw_cat.split(',')[0].strip()
             raw_sub_cat = summary_raw.get('sub_cat', '')
             if raw_sub_cat:
                 cat_L2 = raw_sub_cat.split(',')[0].strip()

        # Fallback to root 'problem' if not found in summary
        if not problem:
            problem = item.get('problem', '')

        merged_text = f"Problem: {problem}\n\nContents: {contents}"
        
        chunks = text_splitter.create_documents([merged_text])
        
        for chunk in chunks:
            # V1 Spec: id, vector, text, title, cat_L1, cat_L2
            doc_record = {
                "text": chunk.page_content,
                "title": title,
                "cat_L1": cat_L1,
                "cat_L2": cat_L2,
                "idx": str(item.get('idx', loop_idx)) # Use item['idx'] (fallback to loop_idx)
            }
            documents.append(doc_record)
            
    return documents

def ingest_to_milvus(documents):
    print(f"Initializing Milvus Client for V1 (Collection: {Config.MILVUS_COLLECTION_NAME_V1})...")
    client = MilvusClient(
        uri=Config.MILVUS_URI,
        token=Config.MILVUS_TOKEN
    )
    
    collection_name = Config.MILVUS_COLLECTION_NAME_V1
    
    if client.has_collection(collection_name):
        print(f"Dropping existing collection: {collection_name}")
        client.drop_collection(collection_name)
        
    print(f"Creating collection: {collection_name}")
    
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="idx", datatype=DataType.VARCHAR, max_length=128) # Renamed doc_code to idx
    
    # Metadata fields for filtering
    schema.add_field(field_name="cat_L1", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="cat_L2", datatype=DataType.VARCHAR, max_length=256)
    
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="IP")
    
    # Create index for scalar fields if needed for faster filtering (Milvus supports scalar indexing)
    # For now, autoindex on vector is sufficient for hybrid search in newer Milvus versions
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    
    print("Generating embeddings and inserting...")
    embeddings_model = ClovaXEmbeddings(model=Config.EMBEDDING_MODEL)
    
    batch_size = 50
    total_docs = len(documents)
    
    for i in tqdm(range(0, total_docs, batch_size)):
        batch_docs = documents[i : i + batch_size]
        batch_texts = [doc['text'] for doc in batch_docs]
        
        try:
            vectors = embeddings_model.embed_documents(batch_texts)
            
            insert_data = []
            for doc, vector in zip(batch_docs, vectors):
                record = {
                    "vector": vector,
                    "text": doc['text'],
                    "title": doc['title'],
                    "cat_L1": doc['cat_L1'],
                    "cat_L2": doc['cat_L2'],
                    "idx": str(doc['idx']) # Ensure string
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
