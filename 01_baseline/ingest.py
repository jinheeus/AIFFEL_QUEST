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

    with open(filepath, "r", encoding="utf-8") as f:
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
        separators=["\n\n", " ", ""],
    )

    documents = []
    print(f"Processing {len(data)} records for V0 Ingestion...")

    for loop_idx, item in enumerate(tqdm(data)):
        problem = item.get("problem", "")
        contents = item.get("contents", "")
        action = item.get("action", "")
        site = item.get("site", "")
        category = item.get("category", "")
        date = item.get("date", "")

        merged_text = (
            f"[Content]\n{contents}\n\n[Problems]\n{problem}\n\n[Action]\n{action}"
        )

        # Create chunks
        chunks = text_splitter.create_documents([merged_text])

        for chunk in chunks:
            # V0 Spec: id, vector, text. NO Category Metadata.
            doc_record = {
                "text": chunk.page_content,
                "idx": str(
                    item.get("idx", loop_idx)
                ),  # Use idx as doc_code (fallback to loop index)
                "site": site,
                "category": category,
                "date": date,
            }
            documents.append(doc_record)

    return documents


def ingest_to_milvus(documents):
    print(
        f"Initializing Milvus Client for V0 (Collection: {Config.MILVUS_COLLECTION_NAME_V0})..."
    )
    client = MilvusClient(uri=Config.MILVUS_URI, token=Config.MILVUS_TOKEN)

    collection_name = Config.MILVUS_COLLECTION_NAME_V0

    if client.has_collection(collection_name):
        print(f"Dropping existing collection: {collection_name}")
        client.drop_collection(collection_name)

    print(f"Creating collection: {collection_name}")
    # Schema Definition for V0
    # id: Int64 (PK)
    # vector: FloatVector (Dim: 1024)
    # text: VarChar

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(
        field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True
    )
    schema.add_field(
        field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024
    )  # bge-m3 dim
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(
        field_name="idx", datatype=DataType.VARCHAR, max_length=128
    )  # Added doc_code
    schema.add_field(field_name="site", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="date", datatype=DataType.VARCHAR, max_length=32)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="IP",  # Inner Product recommended for normalized vectors (bge-m3)
    )

    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )

    print("Generating embeddings and inserting...")
    # HyperCLOVA X Embeddings
    embeddings_model = ClovaXEmbeddings(model=Config.EMBEDDING_MODEL)

    batch_size = 50
    total_docs = len(documents)

    for i in tqdm(range(0, total_docs, batch_size)):
        batch_docs = documents[i : i + batch_size]
        batch_texts = [doc["text"] for doc in batch_docs]

        try:
            # Generate embeddings
            vectors = embeddings_model.embed_documents(batch_texts)

            # Prepare data for insertion
            insert_data = []
            for doc, vector in zip(batch_docs, vectors):
                record = {
                    "vector": vector,
                    "text": doc["text"],
                    "idx": str(doc["idx"]),  # Ensure string
                    "site": doc["site"],
                    "category": doc["category"],
                    "date": doc["date"],
                    # id is auto-generated
                }
                insert_data.append(record)

            client.insert(collection_name=collection_name, data=insert_data)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")

    print(f"Successfully inserted {total_docs} chunks into {collection_name}")


if __name__ == "__main__":
    try:
        # Use data_v1.json as requested
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "00_data",
            "data_v1.json",
        )
        data = load_data(data_path)
        docs = prepare_documents(data)
        ingest_to_milvus(docs)
    except Exception as e:
        print(f"Error: {e}")
