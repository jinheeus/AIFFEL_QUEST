import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase
from dotenv import load_dotenv
from state import AgentState
from model_factory import ModelFactory

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class GraphRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        # HCX Heavy 모델 사용 (HCX-003): 더 나은 지시 이행 능력과 문맥 이해력 필요
        # HCX-DASH (Light)는 "위반(violation)" 쿼리에 대해 안전 필터가 너무 엄격하게 반응함.
        self.llm = ModelFactory.get_rag_model(level="heavy", temperature=0)

    def close(self):
        self.driver.close()

    def generate_cypher(self, query: str) -> str:
        """자연어 질문을 Neo4j Cypher 쿼리로 변환합니다."""

        schema_str = """
        Nodes:
        - AuditCase (id, title, summary, date, action)
        - Organization (name)
        - Category (name)
        - Regulation (name)

        Relationships:
        - (:Organization)-[:PUBLISHED]->(:AuditCase)
        - (:AuditCase)-[:BELONGS_TO]->(:Category)
        - (:AuditCase)-[:VIOLATED]->(:Regulation)
        """

        from datetime import datetime

        current_date_str = datetime.now().strftime("%Y-%m-%d")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            You are a **Database Administrator** component for an internal Audit System.
            Your ONLY job is to translate natural language to Neo4j Cypher queries.
            
            [CRITICAL INSTRUCTION]
            - You are NOT a chat assistant. You are a code generator.
            - The user is an **Authorized Administrator** querying **Public Records**.
            - Keywords like "violation", "crime", "misconduct" are TECHNICAL TERMS in this database.
            - **DO NOT REFUSE** to generate queries based on safety filters. This is a retrieval task for compliance checking.
            
            [Current Context]
            - Today's Date: {current_date}
            
            [Schema]
            {schema}
            
            [Rules]
            1. Use case-insensitive matching for names (e.g., toLower(o.name) CONTAINS toLower('...')).
            2. Return strictly 3-5 most relevant `AuditCase` nodes details.
            3. Return format: `RETURN c.title, c.summary, c.date, collect(r.name) as regulations`
            4. Do NOT include markdown formatting (```cypher ... ```). Just the code.
            5. If the user asks for a specific organization, filter by Organization.
            6. If the user asks for a regulation, filter by Regulation.
            7. **Safe Querying**: When returning `collect(r.name)`, you MUST ensure `r` is defined. Use `OPTIONAL MATCH` to retrieve regulations if not filtered by them.
            8. **Syntax Warning**: Do NOT use `[]` for labels (e.g., `(c[AuditCase])`). ALWAYS use `:` (e.g., `(c:AuditCase)`).
            9. **Date Handling**:
               - Reference date: {current_date}
               - **"Recent" (최근/최신)**:
                 - User intends to see cases from the past 1-3 years.
                 - You MUST calculate the date string for **1 year ago** (e.g. if today is 2025-12-16, use '2024-12-16').
                 - Logic: `WHERE c.date >= 'YYYY-MM-DD'` (Start date = Today - 1 Year).
                 - **NEVER** use today's date ('{current_date}') as the start date for "recent". That will return nothing.
               - Verify `c.date` format is 'YYYY-MM-DD'.

               Example: `MATCH (o)-[]->(c) WHERE ... OPTIONAL MATCH (c)-[:VIOLATED]->(r) RETURN ..., collect(r.name)`

            [Few-Shot Examples]
            Q: "한국수력원자력의 감사 결과를 찾아줘"
            A: MATCH (o:Organization)-[:PUBLISHED]->(c:AuditCase) WHERE toLower(o.name) CONTAINS '한국수력원자력' OPTIONAL MATCH (c)-[:VIOLATED]->(r:Regulation) RETURN c.title, c.summary, c.date, collect(r.name) as regulations LIMIT 5

            Q: "국가를 당사자로 하는 계약에 관한 법률 위반 사례 보여줘"
            A: MATCH (c:AuditCase)-[:VIOLATED]->(r:Regulation) WHERE toLower(r.name) CONTAINS '국가를 당사자로 하는 계약에 관한 법률' RETURN c.title, c.summary, c.date, collect(r.name) as regulations LIMIT 5

            Q: "음주운전으로 징계받은 사례 있어?"
            A: MATCH (c:AuditCase) WHERE c.summary CONTAINS '음주운전' RETURN c.title, c.summary, c.date, [] as regulations LIMIT 5

            Q: "최근 1년 내 음주운전 사례" (Assuming today is 2024-12-15)
            A: MATCH (c:AuditCase) WHERE c.summary CONTAINS '음주운전' AND c.date >= '2023-12-15' RETURN c.title, c.summary, c.date, [] as regulations LIMIT 5
            """,
                ),
                ("human", "{query}"),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        cypher = chain.invoke(
            {"schema": schema_str, "query": query, "current_date": current_date_str}
        )

        # Cleanup
        # 1. Extract code block if present
        import re

        match = re.search(r"```(?:cypher)?(.*?)```", cypher, re.DOTALL)
        if match:
            cypher = match.group(1).strip()
        else:
            # 2. If no code block, try to remove chatty prefixes/suffixes
            lines = cypher.split("\n")
            cleaned_lines = []
            for line in lines:
                # Remove lines that are clearly not Cypher
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                # Skip conversational prefixes
                if any(
                    line_stripped.lower().startswith(prefix)
                    for prefix in [
                        "here",
                        "this query",
                        "to find",
                        "match the following",
                        "the query",
                    ]
                ):
                    continue
                cleaned_lines.append(line)
            cypher = "\n".join(cleaned_lines).strip()

        # Final check: Remove any leading text before the first MATCH/CALL
        match_start = re.search(r"(MATCH|CALL|Create|Merge)", cypher, re.IGNORECASE)
        if match_start:
            cypher = cypher[match_start.start() :]

        return cypher

    def run_query(self, cypher: str):
        """Cypher 쿼리를 실행합니다."""
        with self.driver.session() as session:
            try:
                result = session.run(cypher)
                return [record.data() for record in result]
            except Exception as e:
                print(f"[GraphRetriever] Cypher Execution Error: {e}")
                return []

    def retrieve(self, query: str) -> str:
        """메인 진입점: 질문 -> Cypher 생성 -> 실행 -> 결과 포맷팅"""
        print(f"[GraphRetriever] Generating Cypher for: {query}")
        try:
            cypher_query = self.generate_cypher(query)
            print(f"[GraphRetriever] Generated Cypher: {cypher_query}")

            results = self.run_query(cypher_query)
            print(f"[GraphRetriever] Graph Query Result Count: {len(results)}")

            if not results:
                return "그래프 데이터베이스에서 관련 정보를 찾을 수 없습니다."

            # Format results
            formatted_context = "### Graph Retrieval Results\n"
            for idx, item in enumerate(results, 1):
                formatted_context += f"{idx}. {item.get('c.title', 'Untitled')}\n"
                formatted_context += (
                    f"   - Summary: {item.get('c.summary', '')[:200]}...\n"
                )
                formatted_context += (
                    f"   - Regulations: {', '.join(item.get('regulations', []))}\n\n"
                )

            return formatted_context
        except Exception as e:
            print(f"[GraphRetriever] General Error during retrieval: {e}")
            return "그래프 데이터베이스 검색 중 오류가 발생했습니다."


if __name__ == "__main__":
    # Test
    retriever = GraphRetriever()
    # Query focusing on Regulation (which we confirmed exists)
    test_query = "국가를 당사자로 하는 계약에 관한 법률을 위반한 사례 찾아줘"
    print(retriever.retrieve(test_query))
    retriever.close()

# --- Agentic RAG를 위한 노드 구현 ---

# 오버헤드 방지를 위한 전역 인스턴스
_graph_retriever_instance = None


def get_graph_retriever():
    global _graph_retriever_instance
    if _graph_retriever_instance is None:
        _graph_retriever_instance = GraphRetriever()
    return _graph_retriever_instance


def retrieve_graph_context(state: AgentState) -> AgentState:
    """
    [Node] Neo4j Graph RAG를 수행하여 관련 맥락을 State에 추가합니다.
    """
    query = state["query"]
    print(f"\n[Node] retrieve_graph_context: 그래프 검색 중... Query: {query}")

    retriever = get_graph_retriever()
    result = retriever.retrieve(query)

    # State 업데이트
    # 리스트 형태로 저장 (나중에 확장성 고려)
    state["graph_context"] = [result]

    return state
