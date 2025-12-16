from pymilvus import connections, Collection
from config import Config


def verify():
    connections.connect("default", uri=Config.MILVUS_URI, token=Config.MILVUS_TOKEN)
    collection = Collection("markdown_rag_parent_child_v1")
    collection.flush()
    collection.load()

    print(f"Count: {collection.num_entities}")

    # Get one record
    res = collection.query(
        expr="id > 0", output_fields=["text", "parent_text", "breadcrumbs"], limit=1
    )
    if res:
        print("Sample Record:")
        print(f"Breadcrumbs: {res[0]['breadcrumbs']}")
        print(f"Parent Text (First 100): {res[0]['parent_text'][:100]}")
        print(f"Parent Text Length: {len(res[0]['parent_text'])}")
        print(f"Text (First 100): {res[0]['text'][:100]}")
    else:
        print("No records found.")


if __name__ == "__main__":
    verify()
