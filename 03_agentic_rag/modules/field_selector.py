from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from state import AgentState
from model_factory import ModelFactory


# --- Output Schema for Validation ---
class FieldSelectorOutput(BaseModel):
    selected_fields: List[str] = Field(
        description="List of selected fields (must include 'outline' and 'problems')"
    )
    selected_fields_cot: List[str] = Field(
        description="Chain of thought reasoning steps"
    )
    limit: int = Field(
        description="Number of documents to retrieve (default 5, extract if user specifies quantity e.g. '2개', 'top 3')",
        default=5,
    )
    sort: str = Field(
        description="Sort order: 'date_desc' (latest/최신), 'date_asc' (oldest), or 'relevance' (default).",
        default="relevance",
    )


# --- Prompts from Legacy Notebook ---
FIELD_SELECTOR_SYSTEM = """
[역할]
당신은 감사 보고서 기반 RAG 시스템의 필드 선택기(Field Selector)입니다.

[목표]
1. 사용자 질문을 분석하여, 아래 정의된 7개의 필드 중 질문과 직접적으로 관련된 필드를 선택하십시오.
2. 질문에서 **문서의 개수(예: 2개, Top 3)**나 **정렬 기준(예: 최신순, 최근)**에 대한 요구사항을 추출하십시오.

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

[추가 추출 항목]
- limit: 질문에서 명시적으로 요구하는 문서의 개수 (기본값: 5). 예: "2개만 보여줘" -> 2
- sort: 질문에서 '최신', '최근', '마지막' 등의 시간적 순서를 요구하는 경우 "date_desc"로 설정. 그 외에는 "relevance".

[선택 가능한 필드 정의]
- title: 사건 제목, 사안명
- standards: 법령, 규정, 기준, 위반 여부
- outline: 사건의 개요, 배경, 전체 상황
- problems: 위반 사항, 문제점, 부적정 사례
- opinion: 관계기관의 의견, 평가, 입장
- criteria: 개선 방안, 내부통제, 절차 보완
- action: 처분, 제재, 징계, 후속 조치

[출력 형식]
- 반드시 아래 JSON 포맷을 준수해야 합니다.
- JSON 외의 텍스트(Markdown block 등)는 포함하지 마십시오.

예시:
Qual: "최신 내부통제 위반 사례 2개 알려줘"
Resp:
{{
  "selected_fields": ["problems", "outline"],
  "selected_fields_cot": [
    "Step 1: 사용자가 내부통제 위반 사례를 요청했으므로 problems와 outline을 선택한다.",
    "Step 2: '최신'이라는 단어에서 시간 순서(sort='date_desc')를 추출한다.",
    "Step 3: '2개'라는 단어에서 문서 개수(limit=2)를 추출한다."
  ],
  "limit": 2,
  "sort": "date_desc"
}}
"""

FIELD_SELECTOR_USER = """
[Chat History]
{history}

[Question]
{question}
"""


def field_selector(state: AgentState) -> dict:
    """
    Field Selector Node:
    Analyzes the user's query to extract metadata filters and select relevant fields
    using Chain-of-Thought (CoT) reasoning.
    """
    print("--- [Node] Field Selector (CoT) ---")
    # Priority: Original Query (User Input) preserves "latest", "2 docs" intents best.
    # search_query might be keyword-optimized and strip these modifiers.
    question = state["query"]

    # Format History
    history_msgs = state.get("messages", [])
    history_text = ""
    if history_msgs:
        # Take last 3 turns
        recent = history_msgs[-3:]
        history_text = "\n".join(
            [f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in recent]
        )

    # 1. Initialize LLM
    llm = ModelFactory.get_eval_model(level="light", temperature=0)
    parser = JsonOutputParser(pydantic_object=FieldSelectorOutput)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FIELD_SELECTOR_SYSTEM + "\n\n{format_instructions}"),
            ("human", FIELD_SELECTOR_USER),
        ]
    )

    # Inject format instructions
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    # 2. Invoke Chain
    chain = prompt | llm | parser

    try:
        result = chain.invoke({"question": question, "history": history_text})

        # 3. Post-process Results
        selected_fields = result.get("selected_fields", [])
        cot = result.get("selected_fields_cot", [])

        # Enforce duplicates removal and mandatory fields
        merged_fields = list(set(selected_fields))
        if "outline" not in merged_fields:
            merged_fields.append("outline")
        if "problems" not in merged_fields:
            merged_fields.append("problems")

        # Metadata logic (Simple mapping based on fields/keywords for now)
        # We assume specific category extraction is handled if 'category' is implied,
        # but sticking to notebook logic which focuses on FIELDS.
        extracted_filters = {}

        # Extract limit
        limit = result.get("limit", 5)
        if limit and limit != 5:  # Only add if strictly specified
            extracted_filters["k"] = limit

        # Extract strict sort
        sort_order = result.get("sort", "relevance")
        if sort_order and sort_order != "relevance":
            extracted_filters["sort"] = sort_order

        print(f" -> CoT: {cot}")
        print(f" -> Fields: {merged_fields}")
        print(f" -> Filters: {extracted_filters}")

        return {
            "selected_fields": merged_fields,
            "selected_fields_cot": cot,
            "metadata_filters": extracted_filters,
        }

    except Exception as e:
        print(f" -> Field Selector Failed: {e}")
        return {
            "selected_fields": ["outline", "problems"],
            "selected_fields_cot": [f"Error: {str(e)}"],
            "metadata_filters": {},
        }
