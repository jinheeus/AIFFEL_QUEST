from typing import List, Dict, Any
import os
import sys

# Add root for common import if needed, though usually handled by entry point
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from langchain_milvus import Milvus

from config import Config
from langchain_naver import ClovaXEmbeddings


class VectorRetriever:
    """
    Dedicated Vector Retriever for Agentic RAG.
    Implements the 'Hybrid Strategy':
    1. Dense Vector Search (Child Chunks)
    2. Parent Context Expansion (Metadata Lookup)
    """

    def __init__(self):
        # 1. Initialize Embeddings
        # Use ModelFactory or direct ClovaXEmbeddings based on Config
        # Config.EMBEDDING_MODEL is 'bge-m3' or similar, but here we use ClovaXEmbeddings wrapper
        # if using BGE-M3 remotely or locally.
        # However, previous pipeline used:
        # self.embedding_model = ClovaXEmbeddings(model=Config.EMBEDDING_MODEL)
        self.embedding_model = ClovaXEmbeddings(model=Config.EMBEDDING_MODEL)

        # 2. Initialize Vector Store (Milvus)
        # Using langchain_milvus for consistency
        self.vector_store = Milvus(
            embedding_function=self.embedding_model,
            connection_args={
                "uri": Config.MILVUS_URI,
                "token": Config.MILVUS_TOKEN,
            },
            collection_name=Config.MILVUS_COLLECTION_NAME_MARKDOWN,  # Ensure this matches ingestion
            auto_id=True,
        )
        # Note: Ingestion used 'markdown_rag_hybrid_v1' or similar.
        # Check Config.MILVUS_COLLECTION_NAME_MARKDOWN value.
        # In config.py: MILVUS_COLLECTION_NAME_MARKDOWN = "markdown_rag_parent_child_v1"
        # BUT pipeline.py used "markdown_rag_hybrid_v1".
        # I must ensure consistency. I will check pipeline.py again to be sure of the collection name.
        # pipeline.py: self.collections = {"hybrid": "markdown_rag_hybrid_v1"}
        # config.py doesn't seem to have "markdown_rag_hybrid_v1" explicitly defined as a constant?
        # Let's override for safety or better, add to Config.

        self.collection_name = (
            "markdown_rag_hybrid_v1"  # Hardcoded in pipeline.py, moving here.
        )
        # Re-init with correct collection if different
        self.vector_store = Milvus(
            embedding_function=self.embedding_model,
            connection_args={
                "uri": Config.MILVUS_URI,
                "token": Config.MILVUS_TOKEN,
            },
            collection_name=self.collection_name,
            auto_id=True,
        )

        print(f"✅ [VectorRetriever] Initialized. Collection: {self.collection_name}")

    def search_and_merge(
        self, query: str, top_k: int = 5, filters: Dict[str, Any] = {}
    ) -> List[str]:
        """
        Search for child chunks and return unique 'parent_text' contexts.
        """
        print(f"   [Vector] Searching for: '{query}' (k={top_k})")

        # 1. Search (Fetch more candidates for dedup)
        # Filters can be applied here if Milvus supports expr
        expr = None
        if filters:
            # Simple implementation for equality filters
            # e.g. filters={'source_type': 'BAI'} -> expr="source_type == 'BAI'"
            conditions = []
            for k, v in filters.items():
                if isinstance(v, str):
                    conditions.append(f"{k} == '{v}'")
                else:
                    conditions.append(f"{k} == {v}")
            if conditions:
                expr = " and ".join(conditions)
                print(f"   [Vector] Applying Filter Expr: {expr}")

        try:
            results = self.vector_store.similarity_search(
                query,
                k=top_k * 3,
                expr=expr,  # Fetch 3x for sufficiency after dedup
            )
        except Exception as e:
            print(f"   ⚠️ [Vector] Search Error: {e}")
            return []

        # 2. Deduplicate & Expand
        seen_parents = set()
        final_contexts = []

        for doc in results:
            # Try to get parent_text from metadata
            parent_text = doc.metadata.get("parent_text")

            # Fallback to page_content if parent_text is missing (legacy data?)
            if not parent_text:
                parent_text = doc.page_content

            # Dedup by content hash
            h = hash(parent_text)
            if h not in seen_parents:
                seen_parents.add(h)
                final_contexts.append(parent_text)

            if len(final_contexts) >= top_k:
                break

        print(f"   -> Retrieved {len(final_contexts)} unique contexts.")
        return final_contexts


# --- Singleton Accessor ---

_vector_retriever_instance = None


def get_retriever() -> VectorRetriever:
    global _vector_retriever_instance
    if _vector_retriever_instance is None:
        print("⚡ [Singleton] Initializing VectorRetriever...")
        _vector_retriever_instance = VectorRetriever()
    return _vector_retriever_instance
