from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from state import GraphState
from utils.llm import llm_hcx

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

[출력 예시]
{{
  "selected_fields": ["outline", "problems", "standards"],
  "selected_fields_cot": [
    "Step 1: 질문은 사건에서 법령이나 규정 위반 여부를 확인하려는 목적이다.",
    "Step 2: 모든 질문에 필수로 포함해야 하는 outline과 problems를 선택한다.",
    "Step 3: 위반된 법령과 기준 판단이 필요하므로 standards를 추가한다."
  ]
}}
"""

field_selector_user = """
[Question]
{question}
"""

field_selector_template = ChatPromptTemplate.from_messages([
    ("system", field_selector_system),
    ("human", field_selector_user)
])

field_selector_chain = (
    field_selector_template
    | llm_hcx
    | JsonOutputParser()
)

def field_selector(state: GraphState) -> dict:
    print("\n[NODE] FIELD_SELECTOR_RUNNING")
    
    question = state["question"]
    
    try:
        result = field_selector_chain.invoke({"question": question})
        new_selected = result.get("selected_fields", [])
        cot = result.get("selected_fields_cot", [])

        normalized_selected = []
        for field in new_selected:
            field = field.lower()
            if field in ["outlines", "outline"]: normalized_selected.append("outline")
            elif field in ["problem", "problems"]: normalized_selected.append("problems")
            else: normalized_selected.append(field)
        
        merged_fields = list(set(normalized_selected))
        if "outline" not in merged_fields: merged_fields.append("outline")
        if "problems" not in merged_fields: merged_fields.append("problems")

        print(f"[INFO] SELECTED_FIELDS: {merged_fields}")
        
        return {
            "selected_fields": merged_fields,
            "selected_fields_cot": cot
        }

    except Exception as e:
        print(f"[ERROR] Field Selector: {e}")
        return {
            "selected_fields": ["outline", "problems"],
            "selected_fields_cot": ["Error occurred, returning default fields."]
        }