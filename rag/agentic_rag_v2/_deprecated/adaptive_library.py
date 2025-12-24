from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from state import AgentState
from model_factory import ModelFactory

# --- 1. 문서 평가 (Grade Documents) ---


class GradeResult(BaseModel):
    """문서 적합성 평가 결과"""

    reasoning: str = Field(description="문서가 적합한지/부적합한지에 대한 추론")
    relevant: str = Field(description="'yes' 또는 'no'")


def grade_documents(state: AgentState) -> AgentState:
    """
    [Node] 검색된 문서를 평가합니다.
    문서가 부적합하다면, 질문을 재작성(Rewrite)합니다.
    """
    print(f"\n[Node] grade_documents: 문서 적합성 평가 중...")

    docs = state.get("documents", [])
    if not docs or docs == ["검색 결과가 없습니다."]:
        print(" -> 문서 없음. 재검색 필요.")
        return {"documents": [], "grade_status": "fail"}

    # We only check the first few docs to save time, or concatenate them
    context_sample = "\n\n".join([str(d)[:500] for d in docs[:3]])
    query = state["query"]

    llm = ModelFactory.get_rag_model(level="light")  # Use light model for grading

    parser = JsonOutputParser(pydantic_object=GradeResult)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 검색된 문서가 사용자 질문과 관련이 있는지 평가하는 채점관입니다.
        문서에 질문과 관련된 키워드나 의미가 포함되어 있다면 'yes'로 평가하세요.
        문서가 질문과 관련이 있는지 여부를 'yes' 또는 'no'의 이진 점수로 표시하세요.
        
        Output JSON format:
        {{
            "reasoning": "...",
            "relevant": "yes" or "no"
        }}
        """,
            ),
            (
                "human",
                "Retrieved document: \n\n {context} \n\n User question: {question}",
            ),
        ]
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke({"context": context_sample, "question": query})
        score = result["relevant"].lower()
        print(f" -> Grading Result: {score} ({result['reasoning']})")

        if score == "yes":
            return {"grade_status": "success"}
        else:
            return {"grade_status": "fail"}

    except Exception as e:
        print(f" -> Grading Error: {e}")
        # Default to success to avoid infinite loops on error, or fail if strictly needed
        return {"grade_status": "success"}


# --- 2. 질문 재작성 (Rewrite Query) ---


def rewrite_query(state: AgentState) -> AgentState:
    """
    [Node] 검색 성능을 높이기 위해 질문을 재작성합니다.
    """
    print(f"\n[Node] rewrite_query: 질문 재작성 중...")

    query = state["query"]
    llm = ModelFactory.get_rag_model(level="light")

    msg = [
        (
            "system",
            "당신은 검색 쿼리를 최적화하는 유용한 비서입니다. 초기 질문을 보고 의미 기반 검색엔진에 적합한 더 나은 쿼리를 작성하세요. 오직 재작성된 쿼리 문자열만 출력하세요.",
        ),
        ("human", f"Initial Query: {query}"),
    ]

    response = llm.invoke(msg)
    new_query = response.content.strip()

    print(f" -> Originally: '{query}'")
    print(f" -> Rewritten: '{new_query}'")

    # Update the query for the next retrieval step
    # We might want to keep original query in history, but for simple loop we update 'sub_queries' or 'query'
    # Let's update 'sub_queries' to be safe for the retriever logic

    return {
        "sub_queries": [new_query],
        "retrieval_count": state.get("retrieval_count", 0) + 1,
    }
