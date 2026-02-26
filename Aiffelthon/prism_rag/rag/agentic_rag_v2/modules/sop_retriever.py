from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

from state import AgentState
from common.model_factory import ModelFactory
from common.logger_config import setup_logger

logger = setup_logger("SOP_RETRIEVER")


# --- Data Models (Pydantic) ---
class FactOutput(BaseModel):
    subject: str = Field(description="행위 주체 (예: OO본부, 담당자)")
    action: str = Field(description="핵심 감사 대상 행위")
    amount: str = Field(description="관련 금액 (없으면 'N/A')")
    date: str = Field(description="행위 시점")


class ComplianceOutput(BaseModel):
    status: str = Field(description="판정 결과 (위반/준수/판단불가)")
    reasoning: str = Field(description="규정과 사실관계를 대조한 논리")
    matched_regulation: str = Field(
        description="적용된 근거 규정/지침 (LLM 내부 지식 활용)"
    )


class DispositionOutput(BaseModel):
    disposition: str = Field(
        description="처분 유형 (주의, 경고, 시정, 징계, 변상, 권고 등)"
    )
    detail: str = Field(description="구체적인 처분 내용 및 근거")


# --- Prompts ---
# --- SOP Master Prompt (Consolidated) ---
SOP_MASTER_PROMPT = """
[역할]
당신은 대한민국 공공부문 감사 전문가입니다.
제공된 "문서(Context)"를 근거로 사용자 "질문(Query)"에 대한 감사 사실관계를 분석하고,
관련 규정을 적용하여 위반 여부와 처분 수위를 한 번에 판단하십시오.

[분석 단계]
1. **사실관계(Facts)**: 문서에서 핵심 사실(주체, 행위, 금액, 시점)을 추출하십시오.
2. **관련 규정(Regulations)**: 해당 사안에 적용되는 법령/규정을 식별하십시오. (LLM 내부 지식 활용 가능)
3. **위반 판단(Compliance)**: 사실관계가 규정에 위반되는지 판단하십시오. (위반/준수/판단불가)
4. **처분 결정(Disposition)**: 위반 시 적절한 처분 유형(주의/경고/징계 등)을 제시하십시오.

[작성 원칙]
- 분석 내용은 반드시 제공된 문서에 기반해야 합니다.
- 추측성 판단은 배제하고, 근거가 명확한 경우에만 단정적으로 기술하십시오.

[질문]
{query}

[문서]
{context}

[출력 형식 - JSON]
다음 JSON 형식으로만 출력하십시오. Markdown block 없이 순수 JSON이어야 합니다:
{{
    "facts": {{
        "subject": "주체",
        "action": "행위 내용",
        "amount": "금액 (없으면 -)",
        "date": "시점 (없으면 -)"
    }},
    "compliance": {{
        "status": "위반/준수/판단불가",
        "reasoning": "판단 근거",
        "matched_regulation": "적용 규정"
    }},
    "disposition": {{
        "type": "처분 유형",
        "detail": "처분 내용 및 사유"
    }}
}}
"""


class SOPOutput(BaseModel):
    facts: dict
    compliance: dict
    disposition: dict


def sop_retriever(state: AgentState) -> dict:
    """
    SOP 실행 노드 (Optimized):
    4단계 체인을 단일 LLM 호출로 통합하여 속도를 개선함.
    """
    logger.info("--- [Node] SOP Generator (Optimized Single-Call) ---")
    query = state.get("search_query") or state["query"]
    docs = state.get("documents", [])

    # Document 객체 처리
    docs_text = []
    for d in docs:
        if hasattr(d, "page_content"):
            docs_text.append(d.page_content)
        else:
            docs_text.append(str(d))

    context_text = "\n\n".join(docs_text)[:15000]  # 충분한 컨텍스트 제공

    # [Optimization] Use 'heavy' model but only ONCE
    llm = ModelFactory.get_rag_model(level="heavy", temperature=0)

    # Single Chain
    sop_chain = (
        ChatPromptTemplate.from_template(SOP_MASTER_PROMPT)
        | llm
        | JsonOutputParser(pydantic_object=SOPOutput)
    )

    try:
        logger.info(" -> Analyzing SOP (Consolidated)...")
        result = sop_chain.invoke({"query": query, "context": context_text})

        # Parse Result
        facts = result.get("facts", {})
        comp = result.get("compliance", {})
        disp = result.get("disposition", {})

    except Exception as e:
        logger.error(f" -> SOP Analysis Failed: {e}")
        facts = {"subject": "Error", "action": "Analysis Failed"}
        comp = {"status": "Unknown", "reasoning": str(e), "matched_regulation": "-"}
        disp = {"type": "Manual Review", "detail": "System Error"}

    # Format Output
    sop_result = f"""
[SOP Analysis Result]
1. **Facts**: {facts}
2. **Regulations**: {comp.get("matched_regulation", "-")}
3. **Compliance**: {comp.get("status", "Unknown")} ({comp.get("reasoning", "-")})
4. **Disposition**: {disp.get("type", "Unknown")} - {disp.get("detail", "-")}
"""
    logger.info(f" -> SOP Result: {comp.get('status')} / {disp.get('type')}")

    return {"sop_context": sop_result}
