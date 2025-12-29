from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from common.model_factory import ModelFactory
from common.logger_config import setup_logger

logger = setup_logger("DRAFTING_AGENT")


class DraftingAgent:
    """
    Drafting Agent for generating formal Audit Reports based on chat history.
    Uses HCX-003 (Heavy) for high-quality generation.
    """

    def __init__(self):
        # Upgrade to 'reasoning' (HCX-007) for better adherence to instructions
        self.llm = ModelFactory.get_rag_model(level="reasoning", temperature=0.1)
        self.checker_llm = ModelFactory.get_rag_model(level="light", temperature=0.0)

    def analyze_requirements(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyzes conversation history to check for missing report requirements.
        """
        logger.info("--- [DraftingAgent] Analyzing Requirements ---")

        # Format history
        formatted_history = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"[{role}]: {msg['content']}\n\n"

        system_prompt = """
        당신은 보고서 요건 분석가(Request Analyst)입니다.
        대화 내역을 분석하여 감사 보고서 작성에 필요한 필수 정보가 포함되어 있는지 확인하십시오.

        [판단 기준 - Interactive Mode]
        - 필수 항목(사건 제목, 대상 기관, 문제점) 중 2개 이상이 누락되었거나, 대화 내용이 턱없이 부족하면 "status": "missing_info"를 반환하여 사용자에게 되물으십시오.
        - 단, 사용자가 "그냥 써줘", "임의로 작성해", "알아서 해"라고 강제하거나 정보가 80% 이상 충족되면 "status": "ready"로 진행하십시오.

        [데이터 소스 규칙]
        - **사용자가 제공한 모든 정보를 고려하십시오.**
        - 사용자가 "위 사례를 사용해" 또는 "위와 비슷하게"라고 하면, 어시스턴트의 맥락을 사용하는 것을 허용하십시오.

        [확인할 필수 항목]
        1. 사건 제목
        2. 감사 배경
        3. 감사 목적
        4. 감사 방법
        5. 감사 기간
        6. 대상 기관
        7. 문제점/지적사항

        다음 필드를 포함한 JSON 객체를 반환하십시오:
        - "missing_fields": 누락된 필드 이름 리스트 (한국어).
        - "status": 항상 "ready". (플레이스홀더를 사용하여 진행하므로).

        예시 JSON:
        {{
            "status": "ready",
            "missing_fields": ["감사 기간", "대상 기관"]
        }}
        """

        from langchain_core.output_parsers import JsonOutputParser

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Conversation:\n{history}\n\nAnalyze requirements:"),
            ]
        )

        chain = prompt | self.checker_llm | JsonOutputParser()

        try:
            result = chain.invoke({"history": formatted_history})

            if result.get("status") == "missing_info" and not result.get(
                "missing_fields"
            ):
                result["missing_fields"] = ["사건 개요", "감사 기간", "대상 기관"]

            return result
        except Exception as e:
            logger.error(f"Error analyzing requirements: {e}")
            return {
                "status": "ready",
                "missing_fields": [],
            }  # Fallback to ready to not block

    def generate_report(
        self,
        messages: List[Dict[str, str]],
        retrieved_docs: List[Any],
        additional_info: Dict[str, str] = None,
    ) -> str:
        """
        Generates a structured audit report with strict Source A vs Source B separation.
        """
        logger.info("--- [DraftingAgent] Generating Report (Structured) ---")

        # 1. Format References (Source B)
        formatted_references = ""
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            formatted_references += f"--- [참고 사례 #{i}] ---\n{content}\n\n"

        if not formatted_references:
            formatted_references = "(참고할 만한 유사 사례가 감지되지 않음)"

        # 2. Format History (Source A)
        formatted_history = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"[{role}]: {msg['content']}\n\n"

        # 3. Format Additional Info (Source A+)
        add_info_str = "None"
        if additional_info:
            add_info_str = "\n".join(
                [f"- {k}: {v}" for k, v in additional_info.items()]
            )

        system_prompt = """
[Role]
당신은 대한민국 감사원(BAI) 표준 양식을 준수하는 '공공감사 보고서 작성 전문 에이전트'입니다.
오직 제공된 데이터만을 바탕으로 객관적이고 사실적인 보고서를 작성합니다.

[Data Source Hierarchy & Rule]
1. **Source A (Fact - 최우선)**: {history} 및 {additional_info}
   - 보고서의 모든 '사실관계(성명, 금액, 일자, 장소)'는 오직 여기서만 가져옵니다.
   - 정보가 없으면 임의 생성하지 말고 [확인 필요]로 표기하십시오.
   - **주의**: '김과장', '약 90만원', '전철 이용' 외의 정보는 모두 [미정] 상태입니다.

2. **Source B (Logic/Law - 참고용)**: {references}
   - **[절대 금지]**: Source B에 등장하는 숫자(916,000원 등), 부서명(영업팀 등), 날짜를 Source A의 사건에 대입하지 마십시오.
   - **[허용 범위]**: 관련 법령(국고금 관리법 등), 처벌 수위(정직, 감봉 등), 재발 방지 대책의 '논리적 구조'만 참고하십시오.

[Hallucination Guardrail]
- 사용자가 "약 90만원"이라고 했다면, 절대로 "916,000원"이나 "900,000원"으로 확정 짓지 마십시오. "약 90만원 (확정 금액 조사 필요)"라고 기술하십시오.
- 출처가 불분명한 '영업팀', '2024년 1월' 등의 단어가 포함될 경우 이는 실패한 보고서로 간주됩니다.

[Report Format]
# 감사 보고서

## 감사 실시 개요
### 감사 배경 및 목적
#### 사건 제목
- (Source A 기반: 예: 출장비 부당 수령 의혹 건)
#### 감사 배경
- (Source A 기반: 자가용 이용 보고 후 대중교통 이용 등 실태 기술)
#### 감사 목적
- (공공 예산 집행의 투명성 확보 및 부당 이득 환수)

### 감사 방법 및 기간
#### 감사 방법
- (Source A의 조사 방식 및 Source B의 표준 감사 절차 참고)
#### 감사 기간
- [감사 실시 기간 확인 필요]

## 감사 결과
#### 사건 개요
- (누가, 무엇을 했는지 Fact 위주 기술. 금액은 '약 90만원' 유지)
#### 주요 문제점
- (허위 보고를 통한 유류비 과다 청구 등 위반 사항)
#### 관련 법령 및 규정
- (Source B에서 언급된 법령 적용: 예: 국고금 관리법, 공공기관 회계규정 등)
#### 관계기관 의견 및 감사 판단
- [피감사자 소명 및 기관 의견 확인 필요]
#### 조치 사항
- (Source B의 유사 수위를 참고하되, 이번 사건에 맞춰 제안: 예: 전액 환수 및 징계 검토)
#### 개선 기준 및 재발 방지 대책
- (Source B의 시스템적 개선안 참고: 예: 출장 증빙 강화, 모니터링 시스템 도입 등)

[Tone and Manner]
- 문체는 "~함", "~임" 형태의 개조식 보고서 전문 용어 사용.
- 감정적 표현 배제, 중립적 태도 유지.
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    """
### [Source A: 현재 사건의 사실관계 (채팅 내역)]
{history}

### [Source B: 참고용 유사 감사 사례 (RAG 결과)]
{references}

### [추가 요구사항]
{additional_info}

위 데이터를 바탕으로 감서 보고서 초안을 작성하십시오. Source B의 내용을 Source A인 것처럼 작성하면 안 됩니다.
""",
                ),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()

        try:
            report_draft = chain.invoke(
                {
                    "history": formatted_history,
                    "references": formatted_references,
                    "additional_info": add_info_str,
                }
            )

            # 4. Self-Correction (Refinement)
            return self.refine_report(report_draft, formatted_history, add_info_str)
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return "보고서 생성 중 오류가 발생했습니다. 다시 시도해 주세요."

    def refine_report(self, draft: str, source_facts: str, add_info: str) -> str:
        """
        Self-Correction step: Checks if the draft contains hallucinated punishments or details.
        """
        logger.info("--- [DraftingAgent] Refining Report (Self-Correction) ---")

        system_prompt = """
당신은 공공감사 보고서의 최종 검수관입니다. '초안 보고서'가 '사실 근거(Fact Source)'를 위반했는지 검사하고 교정하십시오.

[Fact Matching Rule]
1. **Source A(채팅 내역)에 없는 정보**는 절대로 확정적으로 기재하지 마십시오.
2. 특히 '감사 기간', '소속 부서', '징계 수위'는 사실 근거에 명시되지 않았다면 아래 [Placeholder Rule]을 따르십시오.

[Placeholder Rule - CRITICAL]
- 사실 근거가 부족하여 모델이 임의로 생성한 정보 뒤에는 반드시 대괄호 가이드를 추가하십시오.
- 형식: [생성된 정보] [해당 내용은 실제 정보로 기입/수정해주세요]
- 예시 1 (기간): 2024-01-01 ~ 2024-01-31 [해당 내용은 실제 감사 기간으로 기입해주세요]
- 예시 2 (부서): 영업팀 [해당 내용은 실제 소속 부서로 수정해주세요]
- 예시 3 (금액): 약 90만원 [해당 금액은 정산 후 확정 금액으로 기입해주세요]

[Correction Logic]
- 징계 수위: 사용자가 "파면" 등을 명시하지 않았다면 "규정에 따른 조치 권고 [징계 수위는 인사위원회 결정에 따름]"으로 수정하십시오.
- 사실 관계: 사실 근거와 초안의 숫자가 다르면 사실 근거의 숫자로 복구하십시오.

[Output]
- 수정된 보고서의 마크다운 본문만 출력하십시오. 설명이나 인사는 생략합니다.
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    """
### [Fact Source (Chat History & Info)]
{source_facts}
{add_info}

### [Draft Report (To be audited)]
{draft}

위 초안 보고서를 [Fact Source]와 대조하여 최종 교정본을 작성하십시오.

**[출력 규칙 - 반드시 지킬 것]**
1. **서두/결미 금지**: '[Corrected Draft Report]', '교정본:', '알겠습니다' 등의 모든 사족을 금지합니다.
2. **요약 금지**: 보고서 하단에 '교정 사항 요약'이나 '변경점'을 절대 작성하지 마십시오.
3. **첫 토큰 제한**: 답변의 첫 번째 글자는 반드시 마크다운 제목 기호인 `#`으로 시작해야 합니다.
4. **내용만 출력**: 오직 교정된 감사 보고서의 본문 마크다운만 출력하십시오.
""",
                ),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()

        try:
            refined_report = chain.invoke(
                {
                    "source_facts": source_facts,
                    "add_info": add_info,
                    "draft": draft,
                }
            )
            return refined_report
        except Exception as e:
            logger.error(f"Error refining report: {e}")
            return draft  # Fallback to original draft
