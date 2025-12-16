from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState
from .shared import rag_pipeline

# Access LLM
llm = rag_pipeline.llm


def decompose_query(state: AgentState) -> AgentState:
    """
    [Node] 복잡한 질문을 여러 개의 하위 검색 질문(Sub-queries)으로 분해합니다.
    """
    original_query = state["query"]
    print(f"\n[Node] decompose_query: 질문 분석 및 분해 시도... '{original_query}'")

    # Decomposer Prompt
    system_prompt = """
    당신은 RAG 검색 최적화를 위한 'Query Decomposer'입니다.
    사용자의 질문이 복합적이거나(두 가지 이상의 주제), 비교 질문일 경우 이를 개별적인 검색 쿼리로 분해하세요.
    단, 단순한 질문이라면 굳이 분해하지 말고 원본 질문을 그대로 리스트에 담아 반환하세요.

    [목표]
    - 검색 엔진이 더 정확한 문서를 찾을 수 있도록 구체적이고 독립적인 질문으로 나눈다.
    - 불필요한 분해는 지양한다.

    [예시]
    1. "A 기관과 B 기관의 횡령 처분 기준 비교해줘" -> ["A 기관 횡령 처분 기준", "B 기관 횡령 처분 기준"]
    2. "직원 채용 비리 시 징계 수위는?" -> ["직원 채용 비리 징계 수위"] (단순 질문 유지)
    3. "음주운전 징계 기준이랑 출장비 유용 기준 알려줘" -> ["음주운전 징계 기준", "출장비 유용 기준"]

    [출력 형식]
    - 반드시 Python List[str] 형태의 JSON으로 출력하세요.
    - 예: ["query1", "query2"]
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{query}")]
    )

    try:
        chain = prompt | llm | JsonOutputParser()
        sub_queries = chain.invoke({"query": original_query})

        # Validation
        if not isinstance(sub_queries, list):
            sub_queries = [original_query]

        # If empty, fallback
        if not sub_queries:
            sub_queries = [original_query]

        print(f" -> 분해 결과: {sub_queries}")

    except Exception as e:
        print(f" -> 분해 실패 (Fallback to original): {e}")
        sub_queries = [original_query]

    state["sub_queries"] = sub_queries
    return state
