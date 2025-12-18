from typing import List, Dict, Any
import os
import sys
from pymilvus import MilvusClient
from langchain_milvus import Milvus
from langchain_naver import ClovaXEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from sentence_transformers import CrossEncoder
import pickle
import time

# Add root for common import if needed
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config import Config


class VectorRetriever:
    """
    Dedicated Hybrid Retriever for Agentic RAG.
    Implements:
    1. Dense Vector Search (Milvus)
    2. Sparse Keyword Search (BM25)
    3. Hybrid Fusion (RRF)
    4. Reranking (BGE-Reranker)
    """

    def __init__(self):
        print(f"[VectorRetriever] Initializing Hybrid Engine...")

        # 1. Initialize Embeddings
        self.embedding_model = ClovaXEmbeddings(model=Config.EMBEDDING_MODEL)
        self.collection_name = "audit_rag_hybrid_v1"

        # 2. Milvus Clients
        # Pymilvus for BM25 loading
        self.milvus_client = MilvusClient(
            uri=Config.MILVUS_URI, token=Config.MILVUS_TOKEN
        )
        # LangChain Store for Dense Search
        self.vector_store = Milvus(
            embedding_function=self.embedding_model,
            connection_args={
                "uri": Config.MILVUS_URI,
                "token": Config.MILVUS_TOKEN,
            },
            collection_name=self.collection_name,
            auto_id=True,
        )

        # 3. Reranker (Cross-Encoder)
        try:
            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
            print("   [Retriever] Reranker Loaded: BAAI/bge-reranker-v2-m3")
        except:
            print(
                "   [Retriever] Warning: Failed to load BGE-M3, falling back to MiniLM"
            )
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # 4. Build BM25 Index
        # This is expensive on startup, but necessary for Hybrid.
        self._build_bm25_index()

    def _load_documents_from_milvus(self) -> List[Dict[str, Any]]:
        """
        Fetches all documents for BM25 indexing.
        """
        print(f"   [Retriever] Loading corpus from Milvus: {self.collection_name}...")
        all_docs = []
        limit = 200  # Safe batch size for scalar fields
        offset = 0

        try:
            while True:
                res = self.milvus_client.query(
                    collection_name=self.collection_name,
                    filter="id >= 0",
                    # Explicitly list fields to EXCLUDE 'vector' (which causes gRPC limit errors)
                    output_fields=[
                        "id",
                        "text",
                        "parent_text",
                        "source_type",
                        "idx",
                        "date",
                        "download_url",
                        "file_path",
                        "category",
                        "cat",
                        "sub_cat",
                        "company_code",
                        "company_name",
                    ],
                    limit=limit,
                    offset=offset,
                )
                if not res:
                    break
                all_docs.extend(res)
                offset += len(res)
                if len(res) < limit:
                    break
            print(f"   [Retriever] Loaded {len(all_docs)} total documents.")
            return all_docs
        except Exception as e:
            print(f"   [Retriever] ❌ Failed to load corpus: {e}")
            return []

    def _build_bm25_index(self):
        print("   [Retriever] Building BM25 Index...")
        self.tokenizer = Kiwi()

        # Cache Path
        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "bm25_cache.pkl"
        )

        # 1. Try Loading from Cache
        if os.path.exists(cache_path):
            try:
                print(f"   [Retriever] Found BM25 cache at {cache_path}. Loading...")
                start_time = time.time()
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                    self.bm25 = cache_data["bm25"]
                    self.bm25_corpus = cache_data["bm25_corpus"]
                    self.bm25_docs = cache_data["bm25_docs"]
                    self.id_to_doc = cache_data["id_to_doc"]
                print(
                    f"   [Retriever] ✅ BM25 Cache Loaded in {time.time() - start_time:.2f}s."
                )
                return
            except Exception as e:
                print(f"   [Retriever] ⚠️ Cache load failed ({e}). Rebuilding...")

        # 2. Rebuild Index (If cache missing or failed)
        raw_docs = self._load_documents_from_milvus()

        self.bm25_corpus = []
        self.bm25_docs = []
        self.id_to_doc = {}

        for d in raw_docs:
            text = d.get("text", "")
            # d is the full entity dict. Convert it to metadata.
            # Pop vector if present to save memory (though output_fields=["*"] usually excludes vector unless asked, wait, * includes vector in some ver, but let's be safe).
            if "vector" in d:
                del d["vector"]

            # Use 'id' as doc_id
            doc_id = d.get("id", "unknown")

            if doc_id == "unknown" and len(self.bm25_corpus) < 3:
                print(f"   [Retriever] ⚠️ Warning: ID missing for doc. Keys: {d.keys()}")

            # Tokenize
            tokens = [t.form for t in self.tokenizer.tokenize(text)]
            self.bm25_corpus.append(tokens)

            # Doc Object (Metadata = All fields)
            doc_obj = Document(
                page_content=text,
                metadata=d,  # Pass full dictionary as metadata
            )
            self.bm25_docs.append(doc_obj)

            # Fast Lookup
            self.id_to_doc[text] = doc_obj

        self.bm25 = BM25Okapi(self.bm25_corpus)

        # 3. Save to Cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "bm25": self.bm25,
                        "bm25_corpus": self.bm25_corpus,
                        "bm25_docs": self.bm25_docs,
                        "id_to_doc": self.id_to_doc,
                    },
                    f,
                )
            print(f"   [Retriever] ✅ BM25 Index Saved to {cache_path}")
        except Exception as e:
            print(f"   [Retriever] ⚠️ Failed to save cache: {e}")

        print(f"   [Retriever] ✅ BM25 Index Ready with {len(self.bm25_corpus)} docs.")

    def search_and_merge(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = {},
        use_reranker: bool = True,
    ) -> List[Document]:
        """
        Hybrid Strategy: Dense + Sparse + RRF + Rerank -> Parent Contexts
        """
        # Override top_k if specified in filters
        if "k" in filters:
            top_k = int(filters["k"])
            print(f"   [Hybrid] Override Top-K: {top_k}")

        print(f"   [Hybrid] Searching for: '{query}'")

        # --- 1. Dense Retrieval ---
        # Note: 'expr' for filters not fully implemented here for simplicity
        # If strict filtering is needed, add expr construction
        dense_results = self.vector_store.similarity_search(query, k=50)

        # --- 2. Sparse Retrieval ---
        tokenized_query = [t.form for t in self.tokenizer.tokenize(query)]
        sparse_docs = self.bm25.get_top_n(tokenized_query, self.bm25_docs, n=50)

        # --- 3. RRF Fusion ---
        dense_ranks = {doc.page_content: i for i, doc in enumerate(dense_results)}
        sparse_ranks = {doc.page_content: i for i, doc in enumerate(sparse_docs)}

        all_content = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        fused_scores = []
        k_const = 60

        for content in all_content:
            rank_d = dense_ranks.get(content, float("inf"))
            rank_s = sparse_ranks.get(content, float("inf"))

            score = 0.0
            if rank_d != float("inf"):
                score += 1.0 / (k_const + rank_d)
            if rank_s != float("inf"):
                score += 1.0 / (k_const + rank_s)

            fused_scores.append((content, score))

        fused_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = fused_scores[:50]

        # --- 4. Parent Logic ---
        seen_parents = set()
        candidates = []

        for content, rrf_score in top_candidates:
            doc_obj = self.id_to_doc.get(content)
            # If missing from lookup (rare), check dense results
            if not doc_obj:
                # Minimal fallback
                doc_obj = Document(
                    page_content=content,
                    metadata={"parent_text": content, "doc_id": "unknown"},
                )

            parent_text = doc_obj.metadata.get("parent_text") or content

            # Create Parent Document with Metadata
            parent_doc = Document(page_content=parent_text, metadata=doc_obj.metadata)

            h = hash(parent_text)
            if h not in seen_parents:
                seen_parents.add(h)
                candidates.append(parent_doc)

        # --- 5. Reranking ---
        final_docs = candidates
        if use_reranker and candidates:
            # [Filters]
            # If sorting by date (Latest), we should be lenient with semantic score.
            # "Latest cases" might have low semantic match to specific content but high user intent for recency.
            min_score = 0.35
            if filters and filters.get("sort") == "date_desc":
                print(
                    f"   [Hybrid] Sort='date_desc' detected. Lowering threshold to 0.1 to capture recent docs."
                )
                min_score = 0.1

            # Rank top 30 candidates
            pool = candidates[:30]
            # Extract content for reranker
            pool_texts = [d.page_content for d in pool]
            pairs = [[query, text] for text in pool_texts]
            scores = self.reranker.predict(pairs)

            scored = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)

            # [Threshold Filtering]
            # Filter out low relevance docs (Noise) which confuse the LLM
            # Score range for BGE-M3 Reranker is typically sigmoid (0-1) but can vary.
            # 0.35 is a safe conservative threshold for "Relevant".
            scored = [s for s in scored if s[1] >= 0.35]

            final_docs = [doc for doc, score in scored]

            if scored:
                print(f"   [Reranker] Top Score: {scored[0][1]:.4f}")
                # [DEBUG] Log actual retrieved titles
                for i, (d, s) in enumerate(scored[:5]):  # show top 5 logic
                    title = d.metadata.get(
                        "title", d.metadata.get("source_type", "No Title")
                    )
                    print(f"   -> [Doc {i + 1}] Score: {s:.4f} | Title: {title}")
            else:
                print(f"   [Reranker] All candidates filtered by threshold (0.35)")
        else:
            final_docs = candidates

        # --- 6. Post-Retrieval Sorting (e.g. Date) ---
        if "sort" in filters:
            final_docs = self._apply_sorting(final_docs, filters["sort"])

        # Final Truncation after Sort
        final_docs = final_docs[:top_k]

        print(f"   [Hybrid] Retrieved {len(final_docs)} final contexts.")
        return final_docs

    def _apply_sorting(self, docs: List[Document], sort_mode: str) -> List[Document]:
        """
        Applies metadata-based sorting.
        Current support: 'date_desc' (Latest first)
        """
        if sort_mode == "date_desc":
            print("   [Hybrid] Sorting by Date (Latest)...")
            try:
                from datetime import datetime

                def parse_date(doc):
                    # Assume metadata['date'] format: YYYY.MM.DD
                    # Fallback to older date if missing/invalid
                    d_str = doc.metadata.get("date", "1900.01.01")
                    try:
                        # Clean up potential whitespace
                        d_str = str(d_str).strip()
                        return datetime.strptime(d_str, "%Y.%m.%d")
                    except Exception:
                        return datetime(1900, 1, 1)

                # Sort: Latest date first
                docs.sort(key=parse_date, reverse=True)
            except Exception as e:
                print(f"   [Hybrid] ⚠️ Date sort failed: {e}")

        return docs


# --- Singleton ---
_vector_retriever_instance = None


def get_retriever() -> VectorRetriever:
    global _vector_retriever_instance
    if _vector_retriever_instance is None:
        _vector_retriever_instance = VectorRetriever()
    return _vector_retriever_instance
