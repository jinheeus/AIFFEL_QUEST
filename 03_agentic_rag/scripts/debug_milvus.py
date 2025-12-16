import sys
import os

# Add root directory to sys.path
# Add root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from modules.vector_retriever import get_retriever
import json


def debug_milvus():
    rag_pipeline = get_retriever()
    # Access underlying pymilvus client
    # Note: langchain_milvus.Milvus stores it as self.client in recent versions or self.col?
    # Actually, accessing private attributes might be risky.
    # But for a debug script, try .client or .pymilvus_client
    client = rag_pipeline.vector_store.client
    collection_name = "problems"

    print(f"Inspecting collection: {collection_name}")

    # 1. Describe Collection
    desc = client.describe_collection(collection_name)
    print("\n[Collection Description]")
    print(desc)

    # 2. Try Fetch All (Empty Filter)
    try:
        print("\n[Test 1] Query with filter=''")
        res = client.query(
            collection_name, filter="", output_fields=["count(*)"], limit=1
        )
        print("Success:", res)
    except Exception as e:
        print("Fail:", e)

    # 3. Try Fetch All (id >= 0) -> Assuming auto_id int64
    try:
        print("\n[Test 2] Query with filter='id >= 0'")
        res = client.query(
            collection_name, filter="id >= 0", output_fields=["idx", "date"], limit=1
        )
        print("Success:", res)
    except Exception as e:
        print("Fail:", e)

    # 4. Try Fetch All (idx >= 0)
    try:
        print("\n[Test 3] Query with filter='idx >= 0'")
        res = client.query(collection_name, filter="idx >= 0", limit=1)
        print("Success:", res)
    except Exception as e:
        print("Fail:", e)


if __name__ == "__main__":
    debug_milvus()
