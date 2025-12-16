import sys
import os

# Add root directory to sys.path
# Add root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from modules.shared import rag_pipeline
import json


def debug_milvus():
    client = rag_pipeline.milvus_client
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
