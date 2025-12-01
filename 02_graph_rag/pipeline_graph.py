import sys
import os
# Add root directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from langchain_naver import ChatClovaX
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from neo4j import GraphDatabase
from config import Config
import json

class GraphRAGPipeline:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )
        
        self.llm = ChatClovaX(
            model=Config.LLM_MODEL, 
            temperature=0.1,
            max_tokens=1024
        )
        
        # Entity Extraction Prompt
        self.extraction_system_prompt = """
        너는 사용자의 질문에서 검색에 필요한 엔티티(Entity)를 추출하는 역할을 맡았어.
        추출해야 할 엔티티는 다음과 같아:
        1. Organization (기관명): 예) "코레일", "한국철도공사", "JDC", "제주국제자유도시개발센터" 등
        2. Category (감사 분야): 예) "예산", "회계", "인사", "계약", "안전" 등
        
        사용자의 질문을 분석해서 JSON 형식으로 반환해.
        값이 없으면 null로 반환해.
        
        예시 1:
        질문: "코레일의 예산 관련 감사 지적 사항 알려줘"
        답변: {{"organization": "한국철도공사", "category": "예산"}}
        
        예시 2:
        질문: "직원 횡령 사례 있어?"
        답변: {{"organization": null, "category": "인사"}}
        
        주의: 기관명은 가능한 공식 명칭으로 정규화해서 추출해줘 (예: 코레일 -> 한국철도공사).
        """
        
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", self.extraction_system_prompt),
            ("human", "{input}")
        ])
        
        self.extractor_chain = self.extraction_prompt | self.llm | JsonOutputParser()
        
        # Generation Prompt (Reusing Baseline Strict Prompt)
        self.gen_system_prompt = """
        너는 공공기관 감사보고서 기반 RAG 어시스턴트이다.
        제공된 감사 보고서 요약(Context)을 바탕으로 질문에 답변하라.
        
        반드시 아래 원칙을 준수하라:
        1. 제공된 문서(context)에 근거해서만 답변한다.
        2. 문서에 없는 내용은 지어내지 않는다.
        3. "기관 판단 및 조치 내용"을 중요하게 다룬다.
        """
        
        self.gen_prompt = ChatPromptTemplate.from_messages([
            ("system", self.gen_system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {input}")
        ])
        
        self.gen_chain = self.gen_prompt | self.llm | StrOutputParser()

    def close(self):
        self.driver.close()

    def extract_entities(self, query):
        print(f"Extracting entities from: {query}")
        try:
            return self.extractor_chain.invoke({"input": query})
        except Exception as e:
            print(f"Extraction Error: {e}")
            return {}

    def retrieve_documents(self, entities):
        org = entities.get("organization")
        cat = entities.get("category")
        
        print(f"Search Targets - Org: {org}, Cat: {cat}")
        
        # Dynamic Cypher Construction
        # This is a simple example. For production, use more robust Cypher generation or Vector Index in Neo4j.
        
        cypher = "MATCH (d:Document)"
        params = {}
        
        conditions = []
        
        if org:
            cypher += "-[:ISSUED_BY]->(o:Organization)"
            conditions.append("o.name CONTAINS $org")
            params["org"] = org
            
        if cat:
            cypher += "-[:BELONGS_TO]->(c:Category)"
            conditions.append("c.name CONTAINS $cat")
            params["cat"] = cat
            
        if conditions:
            cypher += " WHERE " + " AND ".join(conditions)
            
        cypher += " RETURN d.title as title, d.problem_summary as problem, d.date as date LIMIT 5"
        
        print(f"Executing Cypher: {cypher}")
        
        results = []
        with self.driver.session() as session:
            result = session.run(cypher, params)
            for record in result:
                results.append({
                    "title": record["title"],
                    "problem": record["problem"],
                    "date": record["date"]
                })
        
        return results

    def run(self, query):
        # 1. Extract Entities
        entities = self.extract_entities(query)
        
        # 2. Retrieve from Graph
        docs = self.retrieve_documents(entities)
        
        if not docs:
            return "죄송합니다. 조건에 맞는 감사 보고서를 찾지 못했습니다."
            
        # 3. Format Context
        context = "\n\n".join([
            f"Title: {doc['title']}\nDate: {doc['date']}\nProblem: {doc['problem']}" 
            for doc in docs
        ])
        
        print(f"Retrieved {len(docs)} documents.")
        
        # 4. Generate Answer
        answer = self.gen_chain.invoke({"context": context, "input": query})
        return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Graph RAG")
    parser.add_argument("--query", type=str, required=True, help="The question to ask")
    args = parser.parse_args()
    
    pipeline = GraphRAGPipeline()
    try:
        answer = pipeline.run(args.query)
        print("\n=== Answer (Graph RAG) ===\n")
        print(answer)
    finally:
        pipeline.close()
