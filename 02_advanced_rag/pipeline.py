import sys
import os
import json
import argparse
from typing import List, Dict, Any

# Add root directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from langchain_naver import ChatClovaX, ClovaXEmbeddings
from langchain_milvus import Milvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pymilvus import MilvusClient
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from langchain_core.documents import Document
from config import Config

try:
    from data_loader import load_documents_from_milvus
except ImportError:
    from .data_loader import load_documents_from_milvus


class HighContextRAGPipeline:
    def __init__(self):
        # 1. Initialize Components
        self.embedding_model = ClovaXEmbeddings(model=Config.EMBEDDING_MODEL)
        self.llm = ChatClovaX(model=Config.LLM_MODEL, temperature=0.1, max_tokens=2048)

        # Initialize Reranker (Cross-Encoder)
        # Using a multilingual model suitable for Korean if available, or a standard one.
        # Assuming 'BAAI/bge-reranker-v2-m3' or similar is desired, but sticking to a standard one for safety unless specified.
        # User didn't specify, so I'll use a multilingual one that is good for Korean.
        # 'BAAI/bge-reranker-v2-m3' is heavy. 'cross-encoder/ms-marco-MiniLM-L-6-v2' is English focused.
        # I'll use 'BAAI/bge-reranker-v2-m3' as it's state-of-the-art for multilingual.
        # If it fails to load, I'll fallback or the user will see an error.
        try:
            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
        except:
            # Fallback to a smaller model if the large one fails or takes too long
            print(
                "Warning: Failed to load BAAI/bge-reranker-v2-m3, falling back to simple cross-encoder"
            )
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # 2. Milvus Connections
        # We use pymilvus.MilvusClient for flexible querying (Hydration)
        self.milvus_client = MilvusClient(
            uri=Config.MILVUS_URI, token=Config.MILVUS_TOKEN
        )

        # Define Collection Names
        # Hybrid Strategy: Single Collection for Text, but we can filter by 'field' metadata if supported.
        # However, for now, we just map key fields to the hybrid collection.
        self.collections = {
            "hybrid": "markdown_rag_hybrid_v1",
        }

        # We also need LangChain Milvus stores for vector search
        self.vector_stores = {}
        # Load Hybrid collection
        self.vector_stores["hybrid"] = Milvus(
            embedding_function=self.embedding_model,
            connection_args={
                "uri": Config.MILVUS_URI,
                "token": Config.MILVUS_TOKEN,
            },
            collection_name="audit_rag_hybrid_v1",
            auto_id=True,
        )

        # 4. Sparse (BM25) Init
        print("🏗️  Building BM25 Index (This may take a few seconds)...")
        self.tokenizer = Kiwi()

        # Load Corpus from Milvus
        raw_docs = load_documents_from_milvus(
            self.milvus_client, "audit_rag_hybrid_v1", batch_size=200
        )

        # Structure for BM25
        self.bm25_corpus = []  # List of list of tokens
        self.bm25_docs = []  # List of Document objects (1:1 with corpus)
        self.id_to_doc = {}  # Hash(text) -> Document(metadata) - For RRF lookup

        for i, d in enumerate(raw_docs):
            text = d.get("text", "")
            parent = d.get("parent_text", "")

            # Tokenize text for index
            tokens = [t.form for t in self.tokenizer.tokenize(text)]
            self.bm25_corpus.append(tokens)

            # Create Document object
            doc_obj = Document(page_content=text, metadata={"parent_text": parent})
            self.bm25_docs.append(doc_obj)

            # Also store in dict for fast lookup by content (for Dense results)
            self.id_to_doc[text] = doc_obj

        self.bm25 = BM25Okapi(self.bm25_corpus)
        print(f"✅ BM25 Index Built with {len(self.bm25_corpus)} docs.")

        # 3. Prompts
        self.gen_system_prompt = """
당신은 공공기관 감사보고서 기반 "High-Context RAG" 어시스턴트입니다.
제공된 [Context]는 여러 파편화된 정보를 재조립하여 구성된 완전한 감사 보고서 내용입니다.

지침:
1. [Context]에 있는 정보를 종합적으로 분석하여 답변하세요.
2. 특정 '문제점'에 대한 '판단기준'과 '조치사항'이 어떻게 연결되는지 파악하세요.
3. 문서에 명시된 사실(기관명, 금액, 날짜, 법령 등)은 정확하게 인용하세요.
4. 추론이나 상상은 금지하며, 오직 문서에 기반해서만 답변하세요.
"""
        self.gen_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.gen_system_prompt),
                (
                    "human",
                    """
[Context]
{context}

[Question]
{input}
""",
                ),
            ]
        )

    def search_and_merge(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = {},
        use_hyde: bool = False,
        use_reranker: bool = True,
    ) -> List[str]:
        """
        Hybrid Strategy (Dense + Sparse) + Reranker
        """
        print(f"1. [Search] Searching '{query}'...")

        # --- 1. Dense Retrieval (Milvus) ---
        dense_results = self.vector_stores["hybrid"].similarity_search(
            query, k=50
        )  # Fetch 50
        print(f"   [Dense] Retrieved {len(dense_results)} docs.")

        # --- 2. Sparse Retrieval (BM25) ---
        tokenized_query = self.tokenizer.tokenize(query)
        # Get simplified string tokens for BM25
        q_tokens = [t.form for t in tokenized_query]

        sparse_docs = self.bm25.get_top_n(q_tokens, self.bm25_docs, n=50)
        print(f"   [Sparse] Retrieved {len(sparse_docs)} docs.")

        # --- 3. Hybrid Fusion (RRF) ---
        # Map content -> rank (smaller is better)
        dense_ranks = {doc.page_content: i for i, doc in enumerate(dense_results)}
        sparse_ranks = {doc.page_content: i for i, doc in enumerate(sparse_docs)}

        all_content = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        fused_scores = []
        k = 60  # RRF constant

        for content in all_content:
            rank_d = dense_ranks.get(content, float("inf"))
            rank_s = sparse_ranks.get(content, float("inf"))

            # RRF Score
            score = 0.0
            if rank_d != float("inf"):
                score += 1.0 / (k + rank_d)
            if rank_s != float("inf"):
                score += 1.0 / (k + rank_s)

            fused_scores.append((content, score))

        # Sort by RRF score descending
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        top_fusion = fused_scores[:50]  # Take Top 50 Fusion Candidates

        # 4. Prepare for Reranking (Parent Context Extraction)
        # We need to map the "Child Chunk Text" back to "Parent Context"

        seen_parents = set()
        candidates = []  # List of parent_text string

        for content, rrf_score in top_fusion:
            # Lookup Document object
            # For Dense, we can find it in dense_results list.
            # For Sparse, we can look in self.id_to_doc

            doc_obj = self.id_to_doc.get(content)

            if not doc_obj:
                # Fallback: Check dense results if not in id_to_doc key (unlikely if synced)
                # But Dense keys came from search, so they must be in DB.
                pass

            if doc_obj:
                parent_text = doc_obj.metadata.get("parent_text")
                if not parent_text:
                    parent_text = content

                h = hash(parent_text)
                if h not in seen_parents:
                    seen_parents.add(h)
                    candidates.append(parent_text)

        print(f"   [Hybrid] Unique Candidates for Reranking: {len(candidates)}")

        # 5. Reranking (Cross-Encoder)
        if use_reranker and self.reranker and candidates:
            # Rank candidates (Limit to top 20-30 to save inference time)
            rerank_pool = candidates[:30]

            # Predict scores for (query, doc) pairs. Always rerank against ORIGINAL query.
            pairs = [[query, doc] for doc in rerank_pool]
            scores = self.reranker.predict(pairs)

            # Sort by score descending
            scored_candidates = sorted(
                zip(rerank_pool, scores), key=lambda x: x[1], reverse=True
            )

            # Extract text
            final_docs = [doc for doc, score in scored_candidates[:top_k]]

            # Debug: Print top reranked scores
            print(
                f"   [Reranker] Top Scores: {[f'{s:.4f}' for _, s in scored_candidates[:3]]}"
            )
        else:
            final_docs = candidates[:top_k]

        print(f"   -> Retrieved {len(final_docs)} final contexts.")
        return final_docs

    def run(self, query: str):
        # 1. Retrieve & Merge
        context_docs = self.search_and_merge(query)

        if not context_docs:
            return "검색 결과가 없습니다."

        context_text = "\n\n".join(context_docs)

        # 2. Generate
        print("6. [Generation] Calling LLM...")
        chain = (
            {"context": lambda x: context_text, "input": RunnablePassthrough()}
            | self.gen_prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run High-Context RAG (Split & Merge)")
    parser.add_argument("--query", type=str, required=True, help="The question to ask")
    args = parser.parse_args()

    pipeline = HighContextRAGPipeline()
    answer = pipeline.run(args.query)

    print("\n=== Answer (High-Context) ===\n")
    print(answer)
