from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_naver import ChatClovaX
from config import Config
from state import AgentState
import json

# Router용 모델 (ClovaX 사용)
router_llm = ChatClovaX(model=Config.LLM_MODEL, temperature=0.1, max_tokens=1024)


def analyze_query(state: AgentState) -> AgentState:
    """
    [Node] 사용자의 질문을 분석하여 카테고리와 페르소나를 결정합니다.
    (현재 시스템에서는 Supervisor로 대체되어 사용되지 않을 수 있음)
    """
    print(f"\n[Node] analyze_query: 질문 분석 중... '{state['query']}'")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
당신은 공공기관 감사 챗봇의 'Router'입니다.
사용자 질문을 분석하여 다음 세 가지 정보를 JSON 형식으로 추출하세요.

1. category:    - 'search': 일반적인 규정, 감사 사례, 처분 기준, 제도 개선 방안 등을 묻는 경우.
    - 'stats': 다음 표현이 포함된 정량적 분석 질문.
      > "몇 건이야?", "총액은?", "합계는?", "가장 많이", "상위 3개", "빈도", "추이", "통계"
      > 주의: "사례를 찾아줘"는 'search'지만, "사례가 몇 건이야?"는 'stats'입니다.
    - 'compare': 두 개 이상의 대상(기관, 연도, 규정 등)을 비교하여 차이점이나 공통점을 묻는 경우.
    - 'judgment': 특정 상황이 규정 위반인지, 어떤 처분이 내려져야 하는지 등 '판단'을 요하는 질문.
        > "이거 위반이야?", "수의계약 2천만 원 넘으면 어떻게 돼?", "처분 수위 알려줘"

2. persona: 
   - 'common': 일반 사용자 (친절하고 알기 쉬운 답변)
   - 'auditor': 감사관 (규정, 판단기준, 적용 법조항 중심의 전문적 답변)
   - 'manager': 경영진/관리자 (현황 파악, 요약, 시사점 위주의 브리핑형 답변)

3. metadata_filters: (stats 카테고리일 경우 추출)
   - 질문에서 특정 연도(예: 2021년)나 기관명이 언급되면 추출. 
   - 예: {{"year": 2021, "org": "A기관"}}

출력 형식:
{{
  "category": "...",
  "persona": "...",
  "metadata_filters": {{...}}
}}
""",
            ),
            ("human", "{query}"),
        ]
    )

    chain = prompt | router_llm | StrOutputParser()
    try:
        result_text = chain.invoke(state["query"])
        start = result_text.find("{")
        end = result_text.rfind("}") + 1
        result = json.loads(result_text[start:end])

        state["category"] = result.get("category", "search")
        state["persona"] = result.get("persona", "common")
        state["metadata_filters"] = result.get("metadata_filters", {})

        print(f" -> 분석 결과: {state['category']} / {state['persona']}")

    except Exception as e:
        print(f" -> 분석 오류: {e}. 기본값 'search'로 진행합니다.")
        state["category"] = "search"
        state["persona"] = "common"

    return state
