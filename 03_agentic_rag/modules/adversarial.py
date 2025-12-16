from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import AgentState
from model_factory import ModelFactory


# Defense Agent (Auditee)
def defense_agent(state: AgentState) -> AgentState:
    """
    [Node] 피감기관(Auditee) 역할 (변호인)
    위반 판정에 대해 '사업의 필요성', '긴급성', '예외 조항'을 들어 소명(변론)합니다.
    """
    print("\n[Node] defense_agent: 피감기관(변호) 입장 변론 중... (Model: HCX Heavy)")

    facts = state.get("facts", {})
    compliance = state.get("compliance_result", {})
    context_text = "\n".join(
        state.get("documents", []) + state.get("graph_context", [])
    )[:3000]

    # HCX Heavy for creative defense
    llm = ModelFactory.get_rag_model(level="heavy", temperature=0.7)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 **감사를 받고 있는 부서장(피감기관)**입니다.
         위반 지적에 대해 **제공된 사실(Facts)과 정황(Context Details)에만 근거하여** 방어해야 합니다.
         
         [Strict Rules]
         1. **절대 없는 사실을 지어내지 마세요.** (예: 코로나19, 천재지변, 인력 부족 등 팩트에 없는 핑계 금지)
         2. 팩트 내에서 소명할 거리가 없다면, "고의성이 없었음"이나 "초범임"을 들어 선처를 호소하세요.
         3. 거짓 변명을 하는 것보다 잘못을 인정하고 재발 방지를 약속하는 것이 낫습니다.
         
         [전략]
         1. **사실 기반 소명**: 팩트에 명시된 업무 상황이나 지시 사항이 있다면 이를 근거로 드세요.
         2. **규정 해석**: 규정의 모호함이나 예외 적용 가능성을 타진하세요.
         3. **정상 참작**: 고의가 아님을 강조하세요.
         4. 한국어로 작성하세요 (약 3문장).
         """,
            ),
            (
                "human",
                """
         [Facts]
         {facts}

         [Preliminary Violation Finding]
         {compliance}

         [Context Details]
         {context}
         """,
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    try:
        defense = chain.invoke(
            {
                "facts": str(facts),
                "compliance": str(compliance),
                "context": context_text,
            }
        )
        state["defense_argument"] = defense
        print(f" -> 변론(Defense): {defense}")
    except Exception as e:
        print(f" -> 변론 생성 실패: {e}")
        state["defense_argument"] = "별도의 소명 자료 없음."

    return state


# Prosecution Agent (Auditor)
def prosecution_agent(state: AgentState) -> AgentState:
    """
    [Node] 감사관(Auditor) 역할 (검사)
    피감기관의 변론을 반박하고, 규정과 절차의 엄격함을 강조합니다.
    """
    print("\n[Node] prosecution_agent: 감사관(검사) 입장 반박 중... (Model: HCX Heavy)")

    defense = state.get("defense_argument", "")
    regs = state.get("matched_regulations", [])

    # HCX Heavy for strict logic
    llm = ModelFactory.get_rag_model(level="heavy", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 **수석 감사관**입니다.
         피감기관의 변명을 **반박(Rebut)**하고 위반 사실을 확정해야 합니다.
         
         [전략]
         1. **논리적 허점 공격**: 피감기관의 해명이 법적 근거가 없음을 지적하세요.
         2. **절차 강조**: 결과가 좋았더라도, **사전 승인**이나 **문서화** 절차가 누락되었음을 강조하세요.
         3. **위험성 경고**: 예외를 인정할 경우 발생할 **도덕적 해이(Moral Hazard)**를 경고하세요.
         4. 한국어로 날카롭게 반박하세요 (약 3문장).
         """,
            ),
            (
                "human",
                """
         [Auditee's Defense]
         {defense}

         [Relevant Regulations]
         {regs}
         """,
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    try:
        prosecution = chain.invoke({"defense": defense, "regs": "\n".join(regs)})
        state["prosecution_argument"] = prosecution
        print(f" -> 반박(Prosecution): {prosecution}")
    except Exception as e:
        print(f" -> 반박 생성 실패: {e}")
        state["prosecution_argument"] = "규정에 근거하여 위반임이 명백함."

    return state


# Judge (Final Verdict)
def judge_verdict(state: AgentState) -> AgentState:
    """
    [Node] 감사위원장(Judge) 역할 (판사)
    양측의 주장을 종합하여 최종 판결을 내립니다.
    """
    print("\n[Node] judge_verdict: 최종 판결 중... (Model: HCX Heavy)")

    facts = state.get("facts", {})
    defense = state.get("defense_argument", "")
    prosecution = state.get("prosecution_argument", "")

    llm = ModelFactory.get_rag_model(level="heavy", temperature=0.1)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 **감사위원회 위원장**입니다.
         피감기관(변호)과 감사관(검사)의 주장을 듣고 **최종 판결(Verdict)**을 내리세요.
         
         [Strict Rules]
         1. **가상의 시나리오 배제**: 만약 피감기관이나 감사관이 팩트에 없는 내용(예: 코로나19 등)을 근거로 삼았다면, 이를 **'근거 없는 주장'으로 기각**하십시오.
         2. **객관적 서술**: 감정적인 호소보다는 규정과 사실관계에 입각해 판단하세요.
         
         [출력 형식]
         **최종 판결문**
         1. **쟁점 요약**: (위반 행위와 적용 규정 간의 핵심 쟁점 1문장)
         2. **주요 고려사항**:
            - 소명 검토: (피감기관 주장의 타당성 여부 검토)
            - 규정 적용: (감사관 주장의 법적 타당성 검토)
         3. **판단**: (종합적인 판단 소견)
         4. **최종 처분**: (위반 유지 / 기각 / 처분 감경 중 선택)
         """,
            ),
            (
                "human",
                """
         [Case Facts]
         {facts}

         [Defense Argument]
         {defense}

         [Prosecution Argument]
         {prosecution}
         """,
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    try:
        verdict = chain.invoke(
            {"facts": str(facts), "defense": defense, "prosecution": prosecution}
        )
        state["final_judgment"] = verdict
        state["answer"] = verdict  # 최종 답변을 판결문으로 덮어씀
        print(f" -> 최종 판결 완료.")
    except Exception as e:
        print(f" -> 판결 생성 실패: {e}")
        state["answer"] = "판결 생성 실패."

    return state
