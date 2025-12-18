from typing import List, Dict, Any
from pymilvus import MilvusClient
from tqdm import tqdm


def load_documents_from_milvus(
    client: MilvusClient, collection_name: str, batch_size: int = 1000
) -> List[Dict[str, Any]]:
    """
    Fetches all documents (text and metadata) from a Milvus collection.
    Used to build local indices (like BM25) that need to mirror the vector DB.
    """
    print(f"📥 Loading corpus from Milvus collection: {collection_name}...")

    # 1. Get total count (approximation) or just iterate
    # Milvus query/iterator is the way.
    # For < 10k items, a single query with large limit is fine.
    # But let's be robust with an iterator-like approach using PKs if possible.
    # Since we set auto_id=True, PKs are generic Int64.

    all_docs = []
    limit = batch_size
    offset = 0

    try:
        # Milvus iterator is safer for large datasets, but simple offset loop works for < 100k
        while True:
            res = client.query(
                collection_name=collection_name,
                filter="id >= 0",
                output_fields=["text", "parent_text", "doc_id"],
                limit=limit,
                offset=offset,
            )

            if not res:
                break

            all_docs.extend(res)
            offset += len(res)
            print(f" -> Fetched {len(res)} / Total {len(all_docs)}")

            if len(res) < limit:
                break

        print(f"✅ Loaded {len(all_docs)} total documents.")
        return all_docs

    except Exception as e:
        print(f"❌ Failed to load documents from Milvus: {e}")
        return []


if __name__ == "__main__":
    # Test Block
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from config import Config

    client = MilvusClient(uri=Config.MILVUS_URI, token=Config.MILVUS_TOKEN)
    docs = load_documents_from_milvus(client, "audit_rag_hybrid_v1")
    print(f"Sample Doc 0: {docs[0] if docs else 'None'}")
