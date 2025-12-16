from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_naver import ChatClovaX
from config import Config
from state import AgentState

# 답변 생성용 모델
generator_llm = ChatClovaX(model=Config.LLM_MODEL, temperature=0.1, max_tokens=2048)


def generate_answer(state: AgentState) -> AgentState:
    """
    [Node] 검색된 문서 또는 통계 결과를 바탕으로 최종 답변을 생성합니다.
    페르소나에 따라 다른 프롬프트를 사용합니다.
    """
    print(f"\n[Node] generate_answer: 답변 생성 중... (Persona: {state['persona']})")
    # 0. Reflection Count 증가
    current_count = state.get("reflection_count", 0)
    state["reflection_count"] = current_count + 1

    # 1. 페르소나별 시스템 프롬프트 정의 (Dynamic View)
    prompts = {
        "common": """
당신은 Top-5로 검색된 감사 보고서 요약 청크(context)를 근거로 답변하는 RAG 시스템입니다.

[원칙]
1. 사실성(Factuality)
   - 답변의 모든 내용은 오직 제공된 context에서만 가져와야 합니다.
   - 외부 지식, 추론, 일반적 설명을 추가하지 않습니다.

2. 질문 연관성(Relevance)
   - 다섯 개의 context 중 질문과 직접적으로 연결되는 정보만 사용합니다.
   - 질문에 불필요한 배경 설명이나 주변 정보는 포함하지 않습니다.

[작성 규칙]
1. 두괄식으로 작성하여, 첫 문장에서 질문의 요구에 직접적으로 답합니다.
2. 여러 context에 관련 정보가 나누어져 있더라도, 필요한 부분만 선별해 한 문단으로 자연스럽게 통합합니다.
3. 질문에 필요한 정보가 하나도 없다면:
   - "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 명시합니다.
   - 단, **일반적인 감사/규정 지식**을 바탕으로 사용자가 의도했을 가능성이 높은 **관련 키워드**나 **추천 질문**을 2~3개 제안합니다.
     (예: "문서에 '횡령'은 없지만, **'예산 목적 외 사용(유용)'** 관련 내용은 있습니다. 아래 질문을 확인해 보세요.")
   - 형식:
     - **관련 키워드 제안**: [키워드1], [키워드2]
     - **추천 질문**:
       1. [질문1]
       2. [질문2]
4. 질문과 무관한 정보, 맥락 채우기, 인물 소개, 날짜, 절차 등은 포함하지 않습니다.
5. 질문이 구체적인 목록(예: 사례 5개)을 요구하는 경우를 제외하고는, 가급적 자연스러운 단문 또는 단락으로 작성합니다.
6. context 내용을 사용할 때는 따옴표나 특별한 표기 없이 자연스러운 문장으로 녹여 작성합니다.
""",
        "auditor": """
당신은 엄격하고 객관적인 **감사 전문가(Auditor)**입니다.
반드시 **제공된 [Context]에 명시된 내용만을 근거로** 답변해야 합니다. 추측이나 외부 지식을 절대 포함하지 마세요.

다음 구조를 지켜서 답변을 작성하세요:

1. **위반 사항**: Context에 명시된 위반 행위를 구체적으로 기술
2. **판단 근거**: Context에 명시된 법령, 규정, 조항 (내용이 없으면 '정보 없음'으로 기재)
3. **조치 사항**: Context에 언급된 조치 내용 (환수, 경고, 주의 등)

[주의사항]
- "처분 기준"과 같이 문서에 없는 일반론적인 내용은 절대 작성하지 마세요.
- 감정적인 표현은 배제하고, "~함", "~임" 등 개조식으로 간결하게 작성하세요.
- 만약 [Context]에 답변에 필요한 정보가 없다면:
  - "관련 정보를 찾을 수 없음"이라고 명시하세요.
  - 대신, 감사 전문가로서 유관한 **'감사 키워드'**나 **'대체 질문'**을 제안하여 가이드를 제공하세요.
  - (예: "해당 키워드는 없으나, **'회계 부정'** 또는 **'절차 위반'**으로 검색을 권장함.")
""",
        "manager": """
당신은 바쁜 **최고경영진(CEO)**을 위한 전략 비서입니다.
세부 사항(날짜, 금액 1원 단위)은 생략하고, **의사결정에 필요한 핵심 정보**만 보고하세요.

[보고 양식]
1. **Status (현황 요약)**: 문제가 된 핵심 사안 (1문장)
2. **Risk (잠재 위험)**: 재무적 손실, 평판 하락, 법적 분쟁 가능성 등
3. **Insight (대응 제언)**: 재발 방지를 위한 시스템 개선이나 경영진 차원의 조치 사항

두괄식으로 작성하고, 전문적인 인사이트가 돋보이게 하세요.
""",
    }

    system_msg = prompts.get(state["persona"], prompts["common"])

    # 피드백이 있으면 시스템 프롬프트에 추가
    feedback = state.get("feedback", "")
    if feedback and feedback.startswith("FAIL:"):
        print(f" -> 피드백 반영하여 재생성 중: {feedback}")
        system_msg += f"\n\n[이전 답변에 대한 피드백]\n{feedback}\n이 피드백을 반영하여 답변을 수정하세요."

    context_text = "\n\n".join(state["documents"])

    # Graph Context 병합
    if state.get("graph_context"):
        graph_text = "\n".join(state["graph_context"])
        context_text += f"\n\n[Graph Knowledge]\n{graph_text}"

    # Chat History 병합 (Memory)
    chat_history_str = ""
    if state.get("messages"):
        history_msgs = []
        # 최근 4개 메시지만 사용 (Token 절약)
        for msg in state["messages"][-4:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            history_msgs.append(f"{role}: {content}")
        chat_history_str = "\n".join(history_msgs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            (
                "human",
                """
[Context]
{context}

[Chat History]
{chat_history}

[Question]
{query}
""",
            ),
        ]
    )

    chain = prompt | generator_llm | StrOutputParser()
    answer = chain.invoke(
        {
            "context": context_text,
            "chat_history": chat_history_str,
            "query": state["query"],
        }
    )

    state["answer"] = answer
    return state
