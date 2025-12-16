from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from state import AgentState

# --- Initialize LLM ---
from model_factory import ModelFactory

# --- LLM 초기화 ---
llm = ModelFactory.get_rag_model(level="heavy", temperature=0)

# --- 프롬프트 정의 ---

field_selector_system = """
[역할]
당신은 감사 보고서 기반 RAG 시스템의 필드 선택기(Field Selector)입니다.

[목표]
사용자 질문을 분석하여,
아래에 정의된 7개의 필드 중 질문과 직접적으로 관련된 필드를 정확히 선택하십시오.

[핵심 원칙]
- 모든 판단은 질문의 의미와 의도를 기준으로 수행합니다.
- 추측이나 일반적 관행에 근거한 선택은 허용되지 않습니다.
- 질문에서 명시적으로 요구되거나 논리적으로 필수적인 필드만 선택합니다.

[필수 포함 규칙]
- "outline"과 "problems"는 질문의 유형과 무관하게 항상 포함해야 합니다.

[조건부 선택 규칙]
- title: 사건명이나 특정 사안의 명칭 자체를 묻는 경우에만 선택합니다.
- standards: 법령, 규정, 기준, 위반 여부, 적법성 판단이 질문의 핵심인 경우에만 선택합니다.
- opinion: 관계기관의 입장, 해명, 평가, 의견을 묻는 경우에만 선택합니다.
- criteria: 개선 방안, 재발 방지 대책, 내부통제 강화, 절차 보완을 묻는 경우에만 선택합니다.
- action: 처분, 제재, 징계, 후속 조치의 내용이나 수준을 묻는 경우에만 선택합니다.

[선택 가능한 필드 정의]
- title: 사건 제목, 사안명
- standards: 법령, 규정, 기준, 위반 여부
- outline: 사건의 개요, 배경, 전체 상황
- problems: 위반 사항, 문제점, 부적정 사례
- opinion: 관계기관의 의견, 평가, 입장
- criteria: 개선 방안, 내부통제, 절차 보완
- action: 처분, 제재, 징계, 후속 조치

[출력 형식]
- 출력은 반드시 하나의 JSON 객체여야 합니다.
- JSON에는 아래 두 개의 키만 포함해야 합니다.

{{
  "selected_fields": [소문자 문자열 리스트],
  "cot": [
    "Step 1: 질문의 핵심 의도와 요구 정보를 분석한다.",
    "Step 2: 필수 규칙에 따라 outline과 problems를 포함한다.",
    "Step 3: 질문의 내용에 따라 추가로 필요한 필드를 판단하여 선택한다."
  ]
}}

[출력 규칙]
- selected_fields에는 항상 "outline"과 "problems"가 포함되어야 합니다.
- selected_fields_cot는 단계적 판단 과정을 나타내는 문자열 리스트여야 합니다.
- selected_fields_cot는 최소 3단계 이상 작성해야 합니다.
- JSON 외의 텍스트는 절대 출력하지 마십시오.
"""

field_selector_user = """
[Question]
{question}
"""

field_selector_template = ChatPromptTemplate.from_messages(
    [("system", field_selector_system), ("human", field_selector_user)]
)

field_selector_chain = field_selector_template | llm | JsonOutputParser()


validator_system = """
[역할]
당신은 감사 보고서 검색 결과의 유효성을 판별하는 검증관(Validator)입니다.

[목표]
사용자의 질문(question)과 검색된 문서(context)를 비교하여,
제공된 문서만을 근거로 질문에 대해 직접적인 답변이 가능한지를 판단하십시오.

[사고 방식]
- 판단은 단계적 사고 과정을 거쳐 수행해야 합니다.
- 각 단계는 질문 요구, 문서 확인, 답변 가능성 판단의 흐름의 순서를 따라야 합니다.
- 판단 과정은 validator_cot에 반드시 기록해야 합니다.

[판단 원칙]
- 판단은 오직 제공된 문서(context)에 근거해야 합니다.
- 외부 지식, 일반 상식, 추론 보완은 허용되지 않습니다.
- 문서에 질문의 핵심 정보가 명시적으로 없으면 반드시 "no"로 판단하십시오.
- 애매한 경우에는 반드시 "no"를 선택하십시오.

[출력 형식]
출력은 반드시 하나의 JSON 객체여야 하며,
아래 두 개의 키만 포함해야 합니다.

{{
  "is_valid": "yes" 또는 "no",
  "validator_cot": [
    "Step 1: 질문이 요구하는 핵심 정보와 판단 기준을 식별한다.",
    "Step 2: 문서에서 해당 정보가 명시적으로 존재하는지 확인한다.",
    "Step 3: 문서만으로 질문에 직접 답변 가능한지 판단한다."
  ]
}}

[출력 규칙]
- is_valid 값은 반드시 소문자 문자열 "yes" 또는 "no"만 허용됩니다.
- validator_cot는 반드시 단계별 사고 흐름을 나타내는 문자열 리스트여야 합니다.
- validator_cot는 최소 3단계 이상 작성해야 합니다.
- JSON 외의 텍스트는 절대 출력하지 마십시오.
"""

validator_user = """
[Question]
{question}

[Retrieved Documents]
{context}
"""

validator_prompt = ChatPromptTemplate.from_messages(
    [("system", validator_system), ("human", validator_user)]
)

validator_chain = validator_prompt | llm | JsonOutputParser()


strategy_decider_system = """
[역할]
당신은 검색 실패 원인을 분석하고 다음 행동 전략을 결정하는 전략 결정자(StrategyDecider)입니다.

[상황]
현재 검색 결과는 검증기(Validator)를 통과하지 못했습니다.
당신은 아래 세 가지 정보를 모두 참고하여,
다음 단계에서 취할 전략을 하나 선택해야 합니다.

- 필드 선택기의 사고 과정(selected_fields_cot)
- 현재 선택된 필드 목록(current_fields)
- 검증기의 사고 과정(validator_cot)

[선택 가능한 전략]
1. rewrite_query
   - 질문의 핵심 정보 유형은 맞으나, 현재 쿼리는 검색에 적합하지 않은 형태입니다.
   - 질문이 종합·판단형으로 작성되어 있어 사례(action)나 처분 결과가 직접 검색되지 않았습니다.
   - 쿼리를 사례 조회 중심, 처분·징계 중심의 검색형 문장으로 재작성하면
     다른 문서가 검색될 가능성이 높습니다.
   - new_query를 실제로 의미 있게 재작성할 수 있는 경우에만 선택해야 합니다.

2. update_fields
   - selected_fields_cot에서의 판단은 부분적으로 타당했습니다.
   - 문서의 주제는 질문과 대체로 일치합니다.
   - 그러나 질문이 요구하는 특정 정보 유형(규정, 조치, 기준 등)이 문서에 없습니다.
   - 현재 선택되지 않은 다른 필드를 추가하면 답변 가능성이 높아질 것으로 판단됩니다.

[선택 가능한 전체 필드 목록]
title, outline, problems, standards, criteria, action, opinion

[필수 제약 규칙 — 매우 중요]
- missing_fields는 반드시 current_fields에 포함되지 않은 필드만 선택해야 합니다.
- 이미 선택된 필드를 다시 선택하는 것은 절대 허용되지 않습니다.
- current_fields를 제외한 나머지 필드 중에서만 missing_fields를 구성하십시오.
- update_fields 전략을 선택했음에도 추가할 필드가 없다면,
  반드시 rewrite_query 전략을 선택해야 합니다.
- rewrite_query 전략을 선택한 경우,
  new_query는 원본 질문과 문장 구조 또는 검색 초점이 반드시 달라야 합니다.
- 원본 질문을 그대로 반복하거나 의미만 바꾼 수준의 쿼리는 허용되지 않습니다.

[사고 방식]
- Step 1에서는 selected_fields_cot를 분석하여,
  현재 필드 구성이 질문의 정보 요구에는 적절했는지 판단합니다.
- Step 2에서는 validator_cot를 분석하여,
  문서에 사례(action) 또는 처분 결과가 존재하지 않았다는 점을 명확히 식별합니다.
- Step 3에서는 다음 중 하나를 명확히 결정합니다.
  - 필드는 충분하지만, 사례·처분을 직접 검색할 수 있도록 쿼리를 재작성해야 한다.
  - 필드 자체가 부족하므로 필드를 확장해야 한다.
- 이 판단 과정은 strategy_decider_cot에 구체적인 판단 근거로 기록해야 합니다.

[출력 형식 - rewrite_query]
출력은 반드시 하나의 JSON 객체여야 하며,
아래 네 개의 키만 포함해야 합니다.

{{
  "strategy": "rewrite_query",
  "missing_fields": [],
  "new_query": "사례·처분·징계·조치(action)를 직접 검색할 수 있도록 재작성된 검색 쿼리",
  "strategy_decider_cot": [
    "이번 질문과 검색 실패 상황에 근거한 실제 판단 내용 1",
    "... (최소 3단계)"
  ]
}}

[출력 형식 - update_fields]
출력은 반드시 하나의 JSON 객체여야 하며,
아래 네 개의 키만 포함해야 합니다.

{{
  "strategy": "update_fields",
  "missing_fields": ["current_fields에 포함되지 않은 필드명"],
  "new_query": "",
  "strategy_decider_cot": [
    "이번 질문과 검색 실패 상황에 근거한 실제 판단 내용 1",
    "... (최소 3단계)"
  ]
}}

[출력 규칙]
- strategy는 반드시 두 값 중 하나여야 합니다.
- JSON 외의 텍스트는 절대 출력하지 마십시오.
"""

strategy_decider_user = """
[Question]
{question}

[Selected Fields]
{current_fields}

[Field Selector COT]
{selected_fields_cot}

[Validator COT]
{validator_cot}
"""

strategy_decider_prompt = ChatPromptTemplate.from_messages(
    [("system", strategy_decider_system), ("human", strategy_decider_user)]
)

strategy_decider_chain = strategy_decider_prompt | llm | JsonOutputParser()


# --- 노드 함수 정의 (Node Functions) ---


# --- 문맥 해결기 (Context Resolver) ---

context_resolver_system = """
[역할]
당신은 대화 맥락을 이해하여 불완전한 질문을 완전한 검색 쿼리로 복원하는 Context Resolver입니다.

[상황]
사용자는 이전 대화(History)를 바탕으로 "2번째 거", "그거", "아까 말한 내용" 등의 대명사를 사용하여 질문합니다.
당신의 목표는 이전 대화 내용을 참고하여, 사용자의 현재 질문이 무엇을 지칭하는지 명확히 밝혀내는 것입니다.

[규칙]
1. History의 마지막 답변(Assistant Message)을 주의 깊게 분석하십시오.
2. 사용자가 "2번째 것"과 같이 순서를 지칭하면, 직전 답변의 리스트나 항목 중 해당 순서의 내용을 찾아서 구체적인 명칭으로 치환하십시오.
3. 질문이 이미 완전하다면(맥락 없이도 이해 가능), 원본 질문을 그대로 반환하십시오.
4. 불필요한 미사여구 없이, **검색에 최적화된 완전한 문장 하나만** 출력하십시오.

[예시]
History:
User: "최근 감사 사건 알려줘"
Assistant: "1. 소방안전교부세 부당 사용... 2. 헬기 시뮬레이터 특허 문제..."

Current Input: "2번째꺼는 어디서 그런거야?"
Result: "헬기 시뮬레이터 특허 문제 사건은 어느 기관의 감사 결과인가?"
"""

context_resolver_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", context_resolver_system),
        (
            "human",
            """
[History]
{history}

[Current Input]
{question}
""",
        ),
    ]
)

context_resolver_chain = context_resolver_prompt | llm | StrOutputParser()


def field_selector(state: AgentState) -> dict:
    print("--- [Advanced] Field Selector (with Context Resolution) ---")
    current_input = (
        state.get("search_query") if state.get("search_query") else state["query"]
    )

    # 1. 문맥 해결 (대화 기록이 있는 경우)
    history = state.get("messages", [])
    resolved_query = current_input

    # 대화 기록이 최소 1턴 이상(사용자+AI)일 때만 수행
    if len(history) >= 2:
        try:
            # Format history for LLM (Last 4 messages for brevity)
            history_text = ""
            for msg in history[-4:]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"

            print(f" -> Resolving context for: '{current_input}'")
            resolved_query = context_resolver_chain.invoke(
                {"history": history_text, "question": current_input}
            ).strip()
            print(f" -> Resolved Query: '{resolved_query}'")
        except Exception as e:
            print(f" -> Context Resolution Failed: {e}")
            resolved_query = current_input

    # 2. 해결된 질문을 사용하여 필드 선택
    try:
        result = field_selector_chain.invoke({"question": resolved_query})
    except Exception as e:
        print(f"Error in field_selector: {e}")
        # 예외 발생 시 기본값 사용
        result = {"selected_fields": ["outline", "problems"], "cot": ["Error fallback"]}

    new_selected = result.get("selected_fields", [])
    cot = result.get("cot", [])

    current_fields = state.get("selected_fields", [])
    merged_fields = list(set(current_fields + new_selected))

    # 필수 필드 강제 포함 (Force basics)
    if "outline" not in merged_fields:
        merged_fields.append("outline")
    if "problems" not in merged_fields:
        merged_fields.append("problems")

    print(f" -> Selected Fields: {merged_fields}")

    return {
        "selected_fields": merged_fields,
        "selected_fields_cot": cot,
        # Update search query to the resolved version!
        "search_query": resolved_query,
    }


def validator(state: AgentState) -> dict:
    print("--- [Advanced] Validator ---")
    question = state["query"]
    documents = state.get("documents", [])

    if not documents:
        print(" -> No documents found. Invalid.")
        return {"is_valid": "no", "validator_cot": ["문서 없음"]}

    # 검증기를 위한 문서 병합 포맷팅 (Context format for Validator)
    # 문서가 문자열 리스트인 경우와 Document 객체 리스트인 경우를 구분하여 처리
    if isinstance(documents[0], str):
        context = "\n\n".join(documents)
    else:
        # Assuming LangChain Documents
        context = "\n\n".join([d.page_content for d in documents])

    try:
        result = validator_chain.invoke({"question": question, "context": context})
    except Exception as e:
        print(f"Error in validator: {e}")
        return {"is_valid": "no", "validator_cot": ["Error fallback"]}

    is_valid = str(result.get("is_valid", "no")).lower().strip()
    cot = result.get("validator_cot", [])

    print(f" -> Validity: {is_valid}")

    return {"is_valid": is_valid, "validator_cot": cot}


def strategy_decider(state: AgentState) -> dict:
    print("--- [Advanced] Strategy Decider ---")
    question = state["query"]
    current_fields = state.get("selected_fields", [])

    sel_cot_text = "\n".join(state.get("selected_fields_cot", []))
    val_cot_text = "\n".join(state.get("validator_cot", []))

    try:
        result = strategy_decider_chain.invoke(
            {
                "question": question,
                "current_fields": str(current_fields),
                "selected_fields_cot": sel_cot_text,
                "validator_cot": val_cot_text,
            }
        )
    except Exception as e:
        print(f"Error in strategy_decider: {e}")
        # 예외 발생 시 재작성 전략으로 안전하게 처리
        result = {"strategy": "rewrite_query", "new_query": question}

    strategy = result.get("strategy", "rewrite_query")
    llm_suggested_missing = result.get("missing_fields", [])
    new_query = result.get("new_query", "").strip()
    decider_cot = result.get("strategy_decider_cot", [])

    final_missing_fields = list(set(llm_suggested_missing) - set(current_fields))

    # Logic Correction
    if strategy == "update_fields" and not final_missing_fields:
        strategy = "rewrite_query"
        if not new_query:
            new_query = question

    updates = {
        "analysis_decision": strategy,
        "strategy_decider_cot": decider_cot,
        "retrieval_count": state.get("retrieval_count", 0)
        + 1,  # Use existing retry_count field name
    }

    print(f" -> Strategy: {strategy}")

    if strategy == "update_fields" and final_missing_fields:
        new_fields = list(set(current_fields + final_missing_fields))
        updates["selected_fields"] = new_fields
        print(f" -> Adding Fields: {final_missing_fields}")

    elif strategy == "rewrite_query":
        # Ensure new_query is not empty
        if not new_query:
            new_query = question

        updates["search_query"] = new_query
        print(f" -> Rewritten Query: {new_query}")

    return updates
