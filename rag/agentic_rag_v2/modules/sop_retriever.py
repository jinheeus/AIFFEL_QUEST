from typing import Dict, Any
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
FACT_PROMPT = """
[역할]
당신은 "감사 사실관계 조사관(Audit Fact Finder)"입니다.
제공된 문서에서 감사 사안의 핵심 사실(행위 주체, 행위 내용, 금액, 시점)을 정확하게 추출하는 역할을 수행합니다.

[과업]
제공된 "문서(Documents)"와 사용자 "질문(Query)"을 바탕으로,
질문과 관련된 특정 사안의 핵심 사실을 아래 기준에 따라 추출하십시오.

[추출 항목]
- **Subject (주체)**: 행위를 한 기관, 부서, 또는 담당자
- **Action (행위)**: 지적된 문제점이나 감사 대상이 된 구체적인 행위
- **Amount (금액)**: 관련 금액 (예산, 손실액 등). 없으면 'N/A'로 표기.
- **Date (시점)**: 해당 행위가 발생한 시점 (년/월/일)

[작성 원칙]
1. 문서에 명시된 내용에만 근거하여 추출하십시오. 추측은 금지됩니다.
2. 여러 사건이 섞여 있을 경우, 사용자의 "질문(Query)"과 가장 직접적으로 관련된 사건 하나에 집중하십시오.
3. 핵심 사실을 간결하고 명확하게 요약하십시오.

[질문]
{query}

[문서]
{context}
"""

REGULATION_PROMPT = """
[역할]
당신은 "대한민국 공공부문 감사·법령 규정 전문가"입니다.
감사 실무에서 적용되는 법령, 시행령, 시행규칙, 행정규칙, 내부 지침에 정통한 전문가로서 행동하십시오.

[과업]
제공된 사실관계(Facts)를 바탕으로,
해당 사안에 "적용 가능성이 있는 법령·규정·지침"을 식별하십시오.

- 법률(법), 시행령, 시행규칙, 고시, 훈령, 예규, 지침 등 모두 포함될 수 있습니다.
- 가능한 경우, "구체적인 조항 번호"까지 명시하십시오.
  (예: 「국가를 당사자로 하는 계약에 관한 법률」 제27조)
- 하나의 사안에 여러 규정이 동시에 적용될 수 있으면 모두 제시하십시오.

[작성 원칙]
1. 사실관계에서 직접 도출 가능한 범위 내에서만 법령을 연결하십시오.
2. 명확한 근거가 없는 경우, 단정하지 말고
   “적용 가능성 있음”, “관련 규정으로 검토 가능”과 같이 표현하십시오.
3. 일반론적 설명보다 "감사 지적에 실제로 인용되는 규정"을 우선 제시하십시오.
4. 존재하지 않거나 확실하지 않은 조항 번호를 임의로 만들어내지 마십시오.
5. 불필요한 서론·결론 없이, 규정 식별과 간단한 적용 이유 중심으로 작성하십시오.

[사실관계]
{facts}
"""

COMPLIANCE_PROMPT = """
[역할]
당신은 "감사 판단관(Audit Judge)"입니다.
감사 실무에서 사실관계와 관련 규정을 비교·검토하여, 위반 여부를 합리적으로 판단하는 역할을 수행합니다.

[과업]
제공된 "사실관계(Facts)"와 "적용 가능 규정(Regulations)"을 비교하여,
해당 행위가 "위반(위반)"인지, "규정 준수(준수)"인지 판단하십시오.

[판단 원칙]
1. 판단은 반드시 제공된 사실관계와 규정 내용에 근거해야 합니다.
2. 규정의 적용 요건이 사실관계와 "명확히 충족"되는 경우에만 “위반”으로 판단하십시오.
3. 규정 해석의 여지가 있거나, 사실관계가 불충분하여 단정하기 어려운 경우에는 위반으로 단정하지 말고 "준수 또는 판단 유보에 가까운 표현"을 사용하십시오.
4. 규정과 사실관계가 명확히 부합하지 않는 경우, 그 불일치 지점을 구체적으로 언급하십시오.
5. 형사책임, 고의성, 제재 수위 등은 판단하지 말고, "감사 관점의 규정 위반 여부"에만 집중하십시오.

[작성 가이드]
- 결론(위반/준수)을 우선 제시한 후, 그 판단의 근거가 되는 사실과 규정의 연결 관계를 간단히 설명하십시오.
- 불필요한 감정적 표현이나 단정적인 법적 문구는 사용하지 마십시오.

[사실관계]
{facts}

[적용 규정]
{regulations}
"""

DISPOSITION_PROMPT = """
[역할]
당신은 "감사 처분 심의위원회(Audit Sentencing Committee)"입니다.
감사 결과에 따라 위반 정도와 감사 기준을 종합적으로 고려하여, 적정한 처분 수준을 판단하는 역할을 수행합니다.

[과업]
제공된 "위반 판단 결과(Violation Context)"를 바탕으로, 감사 실무 기준에 따라 적절한 "처분(Disposition)"을 결정하십시오.

[처분 유형 예시]
- 시정(Correction): 절차·운영상 미흡 사항에 대한 개선 조치 요구
- 주의(Caution): 경미한 위반 또는 관리상 부주의에 대한 주의 환기
- 경고(Warning): 반복 가능성 또는 영향이 있는 위반에 대한 경고
- 징계(Disciplinary Action): 고의·중과실 또는 중대한 위반에 대한 인사 조치 요구
- 변상(Reimbursement): 재정적 손실 발생 시 금전적 책임 요구

[판단 원칙]
1. 처분은 "위반의 명확성, 중대성, 영향 범위, 반복성"을 종합적으로 고려하여 결정하십시오.
2. 위반이 경미하거나, 사실관계·규정 적용이 명확하지 않은 경우에는 과도한 처분(경고·징계)을 피하고 **시정 또는 주의** 중심으로 판단하십시오.
3. 고의성·형사 책임·구체적 징계 수위(파면·해임 등)는 판단하지 말고, "감사 관점의 처분 유형 수준"까지만 제시하십시오.
4. 처분이 필요한 경우, 왜 해당 처분이 적절한지 "간단한 사유"를 함께 설명하십시오.
5. 처분이 불명확하거나 추가 사실 확인이 필요한 경우, 그 사유를 명시하고 "보수적으로 판단"하십시오.

[작성 가이드]
- 우선적으로 추천 처분 유형을 제시한 후, 그 판단 근거를 간결하게 설명하십시오.
- 불필요한 단정적 표현이나 감정적 문구는 사용하지 마십시오.

[위반 판단 결과]
{compliance_result}
"""


def sop_retriever(state: AgentState) -> dict:
    """
    SOP 실행 노드 (4단계 체인):
    1. 사실 추출 (Fact Extraction)
    2. 규정 매칭 (Regulation Matching - 내부 지식)
    3. 규정 준수 확인 (Compliance Check)
    4. 처분 결정 (Disposition Decision)
    """
    logger.info("--- [Node] SOP Generator (4-Step Chain) ---")
    query = state.get("search_query") or state["query"]
    docs = state.get("documents", [])

    # Document 객체 처리 (Handle Document objects)
    docs_text = []
    for d in docs:
        if hasattr(d, "page_content"):
            docs_text.append(d.page_content)
        else:
            docs_text.append(str(d))

    context_text = "\n\n".join(docs_text)[
        :10000
    ]  # 컨텍스트 길이 제한 (Limit context length)

    llm = ModelFactory.get_rag_model(
        level="heavy", temperature=0
    )  # 추론을 위해 Heavy 모델 사용

    # 1단계: 사실 추출 (Step 1: Fact Extraction)
    logger.info(" -> 1. Extracting Facts...")
    fact_chain = (
        ChatPromptTemplate.from_template(FACT_PROMPT)
        | llm
        | JsonOutputParser(pydantic_object=FactOutput)
    )
    try:
        facts = fact_chain.invoke({"query": query, "context": context_text})
    except:
        facts = {
            "subject": "Unknown",
            "action": "General Query",
            "amount": "-",
            "date": "-",
        }

    # 2단계: 규정 매칭 (Step 2: Regulation Matching)
    logger.info(" -> 2. Matching Regulations...")
    reg_chain = (
        ChatPromptTemplate.from_template(REGULATION_PROMPT) | llm | StrOutputParser()
    )
    regs = reg_chain.invoke({"facts": str(facts)})

    # 3단계: 규정 준수 확인 (Step 3: Compliance Check)
    logger.info(" -> 3. Checking Compliance...")
    comp_chain = (
        ChatPromptTemplate.from_template(COMPLIANCE_PROMPT)
        | llm
        | JsonOutputParser(pydantic_object=ComplianceOutput)
    )
    try:
        compliance = comp_chain.invoke({"facts": str(facts), "regulations": regs})
    except:
        compliance = {
            "status": "Unknown",
            "reasoning": "Error in logic",
            "matched_regulation": regs,
        }

    # 4단계: 처분 결정 (Step 4: Disposition)
    logger.info(" -> 4. Determining Disposition...")
    disp_chain = (
        ChatPromptTemplate.from_template(DISPOSITION_PROMPT)
        | llm
        | JsonOutputParser(pydantic_object=DispositionOutput)
    )
    try:
        disposition = disp_chain.invoke({"compliance_result": str(compliance)})
    except:
        disposition = {"disposition": "Refer to Manual", "detail": "Logic Error"}

    # Generator를 위한 최종 출력 포맷팅 (Format Final Output)
    sop_result = f"""
[SOP Analysis Result]
1. **Facts**: {facts}
2. **Regulations**: {compliance["matched_regulation"]}
3. **Compliance**: {compliance["status"]} ({compliance["reasoning"]})
4. **Disposition**: {disposition["disposition"]} - {disposition["detail"]}
"""
    logger.info(
        f" -> SOP Result: {compliance['status']} / {disposition['disposition']}"
    )

    return {"sop_context": sop_result}
