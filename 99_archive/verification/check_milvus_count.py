from pymilvus import Collection, connections
from config import Config


def check_count():
    try:
        connections.connect("default", uri=Config.MILVUS_URI, token=Config.MILVUS_TOKEN)
        if not utility.has_collection("markdown_rag_parent_child_v1"):
            print("Collection does not exist.")
            return

        c = Collection("markdown_rag_parent_child_v1")
        c.flush()  # Force flush to see updates
        print(f"Row count: {c.num_entities}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    from pymilvus import utility

    check_count()
