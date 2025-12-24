from typing import Dict, List, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

from state import AgentState
from .graph_retriever import get_graph_retriever
from model_factory import ModelFactory


# 구조화된 출력을 위한 모델 정의 (Pydantic Models)
class FactSchema(BaseModel):
    subject: str = Field(description="행위 주체 (예: 부서명, 담당자)")
    action: str = Field(description="감사 대상 핵심 행위 (예: 수의계약 체결, 여비지급)")
    amount: str = Field(description="금액 정보 (예: 2,000만원)")
    date: str = Field(description="발생 시점 또는 기간")
    key_details: str = Field(description="판단에 필요한 기타 핵심 정황")


class ComplianceSchema(BaseModel):
    status: str = Field(description="'Violated'(위반) 또는 'Compliant'(준수)")
    reasoning: str = Field(description="Fact와 Regulation을 비교한 단계별 논리")
    matched_regulation: str = Field(description="적용된 구체적 법규/조항")


class DispositionSchema(BaseModel):
    action: str = Field(description="권고되는 처분 (예: 주의, 환수, 징계)")
    basis: str = Field(description="처분 기준표에 근거한 이유")


# --- 노드 정의 (Nodes) ---


def extract_facts(state: AgentState) -> AgentState:
    """
    [Node] SOP 1단계: 팩트 추출 (Fact Extraction)
    문맥(Context)에서 육하원칙(5W1H)에 해당하는 핵심 사실관계를 구조화하여 추출합니다.
    Model: HCX Light
    """
    print("\n[Node] extract_facts: 팩트 추출 중... (Model: HCX Light)")

    # 문맥 병합 (Context Merging)
    context_text = "\n".join(
        state.get("documents", []) + state.get("graph_context", [])
    )[:5000]
    query = state["query"]

    # Light 모델 사용 (Factory 패턴)
    llm = ModelFactory.get_rag_model(level="light", temperature=0)
    parser = JsonOutputParser(pydantic_object=FactSchema)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 깐깐한 감사관입니다. 감사 문맥에서 사실관계(Fact)를 엄밀하게 추출하세요.\n"
                "없는 내용은 'Unknown'으로 표기하고, 절대 지어내지 마세요.\nFormat:\n{format_instructions}",
            ),
            ("human", "질문: {query}\n\n문맥(Context):\n{context}"),
        ]
    )

    chain = prompt | llm | parser

    try:
        facts = chain.invoke(
            {
                "query": query,
                "context": context_text,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        state["facts"] = facts
        print(f" -> 추출된 팩트: {facts}")
    except Exception as e:
        print(f" -> 팩트 추출 실패: {e}")
        state["facts"] = {}

    return state


def match_regulations(state: AgentState) -> AgentState:
    """
    [Node] SOP 2단계: 규정 매칭 (Regulation Matching)
    추출된 행위(Action)와 주체(Subject)를 바탕으로, Graph DB에서 관련 규정을 정밀 검색합니다.
    Model: HCX Light (passed to retriever)
    """
    print("\n[Node] match_regulations: 관련 규정 탐색 중... (Model: HCX Light)")

    facts = state.get("facts", {})
    if not facts:
        print(" -> 매칭할 팩트가 없습니다.")
        state["matched_regulations"] = []
        return state

    # GraphRetriever를 사용하여 구체적인 조항을 검색
    target_query = f"{facts.get('action', 'General')} 및 {facts.get('subject', 'Agency')} 관련 규정 및 위반 시 처분 기준 찾아줘"

    retriever = get_graph_retriever()

    # 힌트를 추가하여 규정 위주로 검색 유도
    result = retriever.retrieve(target_query)

    # 실제로는 조항 단위 파싱이 필요하지만, 여기서는 검색된 텍스트 통째로 사용
    state["matched_regulations"] = [result]
    print(f" -> 규정 검색 결과: {len(result)} 자")

    return state


def evaluate_compliance(state: AgentState) -> AgentState:
    """
    [Node] SOP 3단계: 규정 준수 여부 판단 (Compliance Logic)
    'Fact'와 'Regulation'을 1:1로 비교하여 위반 여부를 논리적으로 도출합니다.
    Model: HCX Heavy
    """
    print("\n[Node] evaluate_compliance: 위반 여부 판단 중... (Model: HCX Heavy)")

    facts = state.get("facts", {})
    regs = state.get("matched_regulations", [])

    if not facts or not regs:
        state["compliance_result"] = "판단 불가 (정보 부족)"
        return state

    # Use Factory (Heavy)
    llm = ModelFactory.get_rag_model(level="heavy", temperature=0)
    parser = JsonOutputParser(pydantic_object=ComplianceSchema)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 수석 감사관입니다. [Fact]와 [Regulation]을 대조하여 위반 여부를 판정하세요.\n"
                "위반(Violated) 또는 준수(Compliant)로 결론 내리고, 논리적인 근거를 제시하세요.\nFormat:\n{format_instructions}",
            ),
            ("human", "Facts: {facts}\n\nRegulations: {regs}"),
        ]
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke(
            {
                "facts": str(facts),
                "regs": "\n".join(regs),
                "format_instructions": parser.get_format_instructions(),
            }
        )
        state["compliance_result"] = result
        print(f" -> 판정 결과: {result['status']}")
    except Exception as e:
        print(f" -> 판정 중 오류 발생: {e}")
        state["compliance_result"] = "Error"

    return state


def determine_disposition(state: AgentState) -> AgentState:
    """
    [Node] SOP 4단계: 처분 결정 (Disposition)
    위반 확정 시, 관련 처분 기준에 따라 제재 수위(주의/경고/환수 등)를 결정하고 보고서를 작성합니다.
    Model: HCX Heavy
    """
    print("\n[Node] determine_disposition: 처분 수위 결정 중... (Model: HCX Heavy)")

    compliance = state.get("compliance_result", {})

    # 위반이 아니면 종료
    if not isinstance(compliance, dict) or compliance.get("status") != "Violated":
        state["answer"] = (
            f"검토 완료. 판정: {compliance if isinstance(compliance, str) else compliance.get('status', 'Unknown')}"
        )
        return state

    # Heavy 모델 사용 (Factory 패턴)
    llm = ModelFactory.get_rag_model(level="heavy", temperature=0)
    parser = JsonOutputParser(pydantic_object=DispositionSchema)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 징계위원회입니다. 위반 사항에 대해 적절한 처분(Disposition)을 결정하세요.\nFormat:\n{format_instructions}",
            ),
            ("human", "위반 내용: {compliance}"),
        ]
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke(
            {
                "compliance": str(compliance),
                "format_instructions": parser.get_format_instructions(),
            }
        )

        # 최종 보고서 형식으로 작성 (Final Report Formatting)
        final_answer = (
            f"### 감사 결과 보고서 (SOP 기반)\n\n"
            f"**1. 사실 관계 (Fact)**\n"
            f"- 주체: {state['facts'].get('subject')}\n"
            f"- 행위: {state['facts'].get('action')}\n"
            f"- 금액/상세: {state['facts'].get('amount')} / {state['facts'].get('key_details')}\n\n"
            f"**2. 관련 규정 (Regulation)**\n"
            f"- 적발 조항: {compliance.get('matched_regulation')}\n\n"
            f"**3. 위반 판단 (Violation)**\n"
            f"- 판정: {compliance.get('status')}\n"
            f"- 근거: {compliance.get('reasoning')}\n\n"
            f"**4. 처분 결과 (Disposition)**\n"
            f"- 조치: {result.get('action')}\n"
            f"- 처분 기준: {result.get('basis')}"
        )

        state["answer"] = final_answer
        print(f" -> 처분 결과: {result['action']}")

    except Exception as e:
        print(f" -> 처분 결정 실패: {e}")
        state["answer"] = str(compliance)

    return state
