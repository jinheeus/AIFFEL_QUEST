import sys
import os
# Add root directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import json
from langchain_naver import ChatClovaX, ClovaXEmbeddings
from langchain_milvus import Milvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pymilvus import MilvusClient
from config import Config

class RAGPipelineV1:
    def __init__(self):
        self.embedding_model = ClovaXEmbeddings(model=Config.EMBEDDING_MODEL)
        
        # Vector Store
        self.vector_store = Milvus(
            embedding_function=self.embedding_model,
            connection_args={
                "uri": Config.MILVUS_URI,
                "token": Config.MILVUS_TOKEN
            },
            collection_name=Config.MILVUS_COLLECTION_NAME_V1,
            auto_id=True
        )
        
        # LLM
        self.llm = ChatClovaX(
            model=Config.LLM_MODEL, 
            temperature=0.1,
            max_tokens=1024
        )
        
        # Router Prompt
        self.router_system_prompt = """너는 공공기관 감사 보고서 분류 전문가야.
사용자의 질문이 다음 5개 카테고리 중 어디에 속하는지 분류해서 JSON으로 반환해.
카테고리: ['예산·회계·재정', '건설·시설·안전', '계약·구매·입찰', '일반행정·보안', '인사·복무·조직']

예시 질문: "직원 횡령 시 처분 기준은?"
예시 답변: {{"category": "인사·복무·조직"}}

반드시 JSON 형식으로만 답변해."""

        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", self.router_system_prompt),
            ("human", "{input}")
        ])
        
        self.router_chain = self.router_prompt | self.llm | StrOutputParser()
        
        # Generation Prompt (Strict Baseline)
        self.gen_system_prompt = """
너는 공공기관 감사보고서 기반 RAG 어시스턴트이다.

반드시 아래 원칙을 준수하라:

1. 너의 모든 답변은 RAG가 제공한 문서(context)에 근거해야 한다.
2. 문서에 명시되지 않은 내용은 절대로 추론하거나 상상하지 않는다.
3. 문서에 근거가 없으면 "제공된 문서만으로는 답변할 수 없습니다."라고 답한다.
4. 문서에 포함된 사실, 수치, 기관명, 조치 내용 등은 있는 그대로 전달한다.
5. 내용을 요약·정리할 수는 있지만 새로운 사실을 생성해서는 안 된다.
6. 문서를 인용할 때는 자연스럽게 문장 안에서 재서술한다.
7. 여러 문서를 제공받으면 서로 모순되는지 먼저 확인하고, 모순될 경우 문서별로 구분해 설명한다.
8. 감사 업무 특성상, 문서의 “기관 판단 및 조치 내용”을 최우선적으로 반영한다.
9. 질문이 모호하면 문서 범위 내에서 가장 명확한 답을 제공하되, 문서 외 해석은 하지 않는다.

너의 목표는 사용자가 문서를 정확히 이해할 수 있도록,
문서 내용을 기반으로만 정확하고 신뢰 가능한 해석을 제공하는 것이다.
"""
        self.gen_prompt = ChatPromptTemplate.from_messages([
            ("system", self.gen_system_prompt),
            ("human", "{context}\n\ninput: {input}")
        ])

    def route_query(self, query):
        try:
            response = self.router_chain.invoke({"input": query})
            # Clean up response if it contains markdown code blocks
            response = response.replace("```json", "").replace("```", "").strip()
            category_json = json.loads(response)
            return category_json.get("category")
        except Exception as e:
            print(f"Router Error: {e}")
            return None

    def run(self, query):
        print(f"Processing query (V1): {query}")
        
        # 1. Route
        category = self.route_query(query)
        print(f"Detected Category: {category}")
        
        # 2. Retrieve with Filter
        search_kwargs = {"k": 5}
        if category:
            search_kwargs["expr"] = f"cat_L1 == '{category}'"
            print(f"Applying Filter: {search_kwargs['expr']}")
        else:
            print("No category detected, performing full search.")
            
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        
        # 3. Generate
        rag_chain = (
            RunnableParallel({
                "context": retriever | self._format_docs,
                "input": RunnablePassthrough()
            })
            | self.gen_prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke(query)
        return response

    def _format_docs(self, docs):
        return "\n\n".join([doc.page_content for doc in docs])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V1 Metadata Filtered RAG")
    parser.add_argument("--query", type=str, required=True, help="The question to ask")
    args = parser.parse_args()
    
    pipeline = RAGPipelineV1()
    answer = pipeline.run(args.query)
    
    print("\n=== Answer (V1) ===\n")
    print(answer)
