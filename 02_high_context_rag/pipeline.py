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
from config import Config


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
            collection_name="markdown_rag_hybrid_v1",
            auto_id=True,
        )

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
        self, query: str, top_k: int = 5, filters: Dict[str, Any] = {}
    ) -> List[str]:
        """
        Modified for 'Hybrid Strategy':
        Since we already enriched 'parent_text' during ingestion,
        we don't need complex ID-based reconstruction.
        Just retrieve top chunks and return their 'parent_text'.
        """
        print(f"1. [Search] Searching 'hybrid' for: '{query}'")

        # Target Collection
        target = self.vector_stores["hybrid"]

        # Search
        # We need 'parent_text' from metadata.
        # LangChain's similarity_search returns Documents with metadata.
        results = target.similarity_search(query, k=top_k * 2)

        # Deduplicate by parent_text (or doc_id) to avoid redundancy
        seen = set()
        final_docs = []

        for doc in results:
            parent_text = doc.metadata.get("parent_text")
            if not parent_text:
                # Fallback to text if parent_text missing
                parent_text = doc.page_content

            # Simple hash check for dedup
            h = hash(parent_text)
            if h not in seen:
                seen.add(h)
                final_docs.append(parent_text)

            if len(final_docs) >= top_k:
                break

        print(f"   -> Retrieved {len(final_docs)} unique contexts.")
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
