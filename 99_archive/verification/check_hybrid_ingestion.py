from pymilvus import connections, Collection
from config import Config


def check_ingestion():
    print("Connecting to Milvus...")
    connections.connect("default", uri=Config.MILVUS_URI, token=Config.MILVUS_TOKEN)

    collection_name = "markdown_rag_hybrid_v1"
    collection = Collection(collection_name)
    collection.load()

    print(f"Collection Loaded. Num Entities: {collection.num_entities}")

    # Query for a BAI file to check Context Injection
    # We look for source_type = 'BAI'
    res = collection.query(
        expr="source_type == 'BAI'",
        output_fields=[
            "id",
            "file_name",
            "parent_text",
            "related_idxs",
            "company_code",
            "breadcrumbs",
        ],
        limit=3,
    )

    if not res:
        print("No BAI data found yet.")
    else:
        print("\n--- Verification Sample (BAI) ---")
        for item in res:
            print(f"File: {item['file_name']}")
            print(f"Related IDXs: {item['related_idxs']}")
            print(f"Breadcrumbs: {item['breadcrumbs']}")
            print("-" * 20)
            print("Parent Text (Snippet):")
            # Print first 500 chars to see the injected header
            print(item["parent_text"][:500])
            print("\n" + "=" * 40 + "\n")

    # Check ALIO
    res_alio = collection.query(
        expr="source_type == 'ALIO'",
        output_fields=["id", "file_name", "company_code"],
        limit=1,
    )
    if res_alio:
        print(f"ALIO Sample Found: {res_alio[0]['file_name']}")
    else:
        print("No ALIO data found yet.")


if __name__ == "__main__":
    check_ingestion()
