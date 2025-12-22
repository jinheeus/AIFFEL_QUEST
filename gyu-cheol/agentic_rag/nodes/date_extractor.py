import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from state import GraphState
from utils.llm import llm_hcx

date_extractor_system = """
[역할]
당신은 감사문서 RAG 시스템의 날짜 조건 추출기(Date Extractor)입니다.

[목표]
사용자 질문에서 날짜 관련 정보를 식별하여,
Milvus 컬렉션의 [date] 필드("YYYY.MM.DD")에 적용 가능한
필터 표현식(expr)으로 변환하십시오.

[입력]
- 자연어 질문.

[출력 목적]
- 검색 단계에서 사용할 날짜 필터 수식을 정확히 생성하는 것
- 날짜 조건이 없을 경우 필터를 적용하지 않기 위해 빈 문자열을 반환하는 것

[변환 규칙]
1. 연도만 언급된 경우
   - "22년", "22년도", "22년도에",
   → date like "2022%"

2. 연도만 언급된 경우
   - "2023년", "2023년도", "23년도에",
   → date like "2023%"

3. 연도 범위가 언급된 경우
   - "2020년부터 2022년까지"
   → (date >= "2020.01.01" and date <= "2022.12.31")

4. 연도 + 월 범위가 언급된 경우
   - "2022년 6월부터 9월까지"
   → (date >= "2022.06.01" and date <= "2022.09.30")

5. 특정 연도 이전 또는 이후
   - "2021년 이전"
   → date < "2021.01.01"
   - "2022년 이후"
   → date >= "2022.01.01"

6. 날짜 정보가 전혀 없는 경우
   → 빈 문자열 "" 반환

[중요 제약 사항]
- 오직 [date] 필드에 대한 조건만 생성하십시오.
- organization은 절대 추가하면 안됩니다.
- 날짜 조건은 반드시 Milvus expr 문법으로 작성해야 합니다.
- AND 조건은 반드시 괄호로 묶어야 합니다.
- 날짜 해석이 애매하거나 확신할 수 없는 경우에는 빈 문자열을 반환하십시오.
- 출력은 반드시 하나의 JSON 객체여야 합니다.
- JSON 외의 텍스트는 절대 출력하지 마십시오.

[출력 형식]
{{
  "milvus_expr": "변환된 날짜 필터 수식 또는 빈 문자열"
}}

[출력 예시]

질문: "2021년에 발생한 감사원 계약 위반 사례를 알려줘."
출력:
{{
  "milvus_expr": "date like \"2021%\""
}}

질문: "2020년부터 2022년까지 국책과제 관련 문제점"
출력:
{{
  "milvus_expr": "(date >= \"2020.01.01\" and date <= \"2022.12.31\")"
}}

질문: "2022년 6월부터 9월까지 발생한 계약 부적정 사례"
출력:
{{
  "milvus_expr": "(date >= \"2022.06.01\" and date <= \"2022.09.30\")"
}}

질문: "2021년 이전에 발생한 감사 결과"
출력:
{{
  "milvus_expr": "date < \"2021.01.01\""
}}

질문: "최근 감사 결과에서 확인된 내부통제 미흡 사례"
출력:
{{
  "milvus_expr": ""
}}
"""

date_extractor_user = """
[Question]
{question}
"""

date_extractor_template = ChatPromptTemplate.from_messages([
    ("system", date_extractor_system),
    ("human", date_extractor_user)
])

date_extractor_chain = (
    date_extractor_template
    | llm_hcx
    | JsonOutputParser()
)

def date_extractor(state: GraphState) -> dict:
    print("\n[NODE] DATE_EXTRACTOR_RUNNING")
    
    updates = {}
    
    if not state.get("original_question"):
        updates["original_question"] = state["question"]
        print(f"[INFO] ORIGINAL QUESTION SAVED: {state['question']}")

    try:
        result = date_extractor_chain.invoke({"question": state["question"]})
        raw_expr = result.get("milvus_expr", "")
        
        date_pattern = r'date\s*(?:like|==|!=|>=|<=|>|<)\s*(?:["\'][^"\']+["\'])'
        found_conditions = re.findall(date_pattern, raw_expr, re.IGNORECASE)
        
        if found_conditions:
            clean_expr = " and ".join(found_conditions)
            if len(found_conditions) > 1:
                clean_expr = f"({clean_expr})"
            
            milvus_expr = clean_expr.replace("'", '"')
            
            print(f"[INFO] EXTRACTED_DATE: {milvus_expr}")
            
        else:
            milvus_expr = ""
            print("[INFO] EXTRACTED_DATE: NONE")
            
    except Exception as e:
        print(f"[ERROR] Date Extractor: {e}")
        milvus_expr = ""
    
    updates["filter_date"] = milvus_expr
    
    return updates