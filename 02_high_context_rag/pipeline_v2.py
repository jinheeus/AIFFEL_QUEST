import sys
import os
import argparse
from typing import List

# Add root directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from langchain_naver import ChatClovaX
from langchain_milvus import Milvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import CrossEncoder
from config import Config

# NEW Collection Name for V2
MILVUS_COLLECTION = Config.MILVUS_COLLECTION_NAME_MARKDOWN


class HighContextRAGPipelineV2:
    def __init__(self):
        # 1. Initialize Components
        # Use HuggingFaceEmbeddings if Config.EMBEDDING_MODEL implies local,
        # but here we follow Config or try to match ingest logic.
        # ingest used BGE-M3 local. pipeline.py used ClovaXEmbeddings.
        # We MUST use the SAME embedding model as ingestion.
        # Ingestion used: HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        # So we should use that here too.

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            import torch

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model_kwargs = {"device": device}
            encode_kwargs = {"normalize_embeddings": True}
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
            print(f"Loaded Local BGE-M3 on {device}")
        except Exception as e:
            print(f"Failed to load local embeddings provided in ingest: {e}")
            # Fallback to Config defaults or error out
            raise e

        self.llm = ChatClovaX(model=Config.LLM_MODEL, temperature=0.1, max_tokens=2048)

        # Reranker (Same as before)
        try:
            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
        except:
            print("Warning: Failed to load BAAI/bge-reranker-v2-m3, falling back.")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # 2. Milvus Vector Store
        self.vector_store = Milvus(
            embedding_function=self.embedding_model,
            connection_args={
                "uri": Config.MILVUS_URI,
                "token": Config.MILVUS_TOKEN,
            },
            collection_name=MILVUS_COLLECTION,
            auto_id=True,
        )

        # 3. Prompts
        self.gen_system_prompt = """
당신은 공공기관 감사보고서 기반 "High-Context RAG" 어시스턴트입니다.
제공된 [Context]는 감사 보고서의 관련 섹션들입니다.

지침:
1. [Context]에 있는 정보를 종합적으로 분석하여 답변하세요.
2. 문서의 계층 구조(Path)를 참고하여 문맥을 파악하세요.
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

    def retrieve_and_rerank(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieves generic chunks, then reranks them.
        """
        print(f"1. [Search] Searching for: '{query}'")

        # Search more candidates for reranking
        search_k = top_k * 4
        results = self.vector_store.similarity_search(query, k=search_k)

        candidates = []  # List of (text_for_rerank, full_doc_obj)

        for doc in results:
            # Metadata keys: 'breadcrumbs', 'parent_text', 'file_name', etc.
            # We want to feed the LLM the 'parent_text' (Context) + 'breadcrumbs'.
            # Reranker should judge based on parent_text primarily.

            breadcrumbs = doc.metadata.get("breadcrumbs", "")
            parent_text = doc.metadata.get("parent_text", "")
            # If parent_text is empty (unexpected), use page_content (child text)
            if not parent_text:
                parent_text = doc.page_content

            formatted_text = f"[Path: {breadcrumbs}]\n{parent_text}"
            candidates.append(formatted_text)

        if not candidates:
            return []

        # Rerank
        print(f"2. [Reranking] Scoring {len(candidates)} candidates...")
        pairs = [[query, cand] for cand in candidates]
        scores = self.reranker.predict(pairs)

        # Sort and Top-K
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        final_docs = [cand for cand, score in scored[:top_k]]

        print(f"   Top-{top_k} selected.")
        return final_docs

    def run(self, query: str):
        # 1. Retrieve
        context_docs = self.retrieve_and_rerank(query)

        if not context_docs:
            return "검색 결과가 없습니다."

        context_text = "\n\n".join(context_docs)

        # 2. Generate
        print("3. [Generation] Calling LLM...")
        chain = (
            {"context": lambda x: context_text, "input": RunnablePassthrough()}
            | self.gen_prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run High-Context RAG V2 (Markdown)")
    parser.add_argument("--query", type=str, required=True, help="The question to ask")
    args = parser.parse_args()

    pipeline = HighContextRAGPipelineV2()
    answer = pipeline.run(args.query)

    print("\n=== Answer (High-Context V2) ===\n")
    print(answer)
