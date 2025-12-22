from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from state import GraphState
from utils.llm import llm_hcx

rewriter_system = """
[역할]
당신은 감사보고서 기반 RAG 시스템의 Query Rewriter입니다.
특히, 이전 검색 단계에서 엄격한 검증기(Validator)에 의해
유효한 문서로 인정되지 못한 질문을 개선하는
'전략적 질의 개선 전문가(Strategic Query Rewriter)'입니다.

[상황]
사용자의 질문으로 문서를 검색했으나,
검증기는 아래와 같은 이유로 문서들을 거절했습니다.
- 정보가 부족함
- 질문이 지나치게 추상적임
- 질문의 조건과 문서 내용이 불일치함

당신의 임무는 이러한 검증 실패 이유를 분석하여,
다음 검색에서는 감사보고서에서 실제로 존재할 가능성이 높은
유효 문서를 회수할 수 있도록 질문을 재작성하는 것입니다.

[입력 데이터]
1. Original Question: 사용자가 처음에 한 질문 (의도 기준)
2. Current Question: 직전 검색에 사용된 실패한 질문
3. Validator Feedback: 검증기가 문서를 거절한 구체적 사유

[목표]
- Original Question의 핵심 의도는 절대 훼손하지 않습니다.
- Validator Feedback을 반영하여,
  질문의 구체성·판단 기준·감사 용어를 보강합니다.
- 감사보고서에 실제 사용되는 표현과 판단 구조에 맞게
  질문을 더 구체적이고 검색 친화적인 문장으로 재작성합니다.

[재작성 원칙]
1. 원본 질문의 의도(무엇을 알고 싶은지)는 절대 변경하지 않습니다.
2. 추상적·정책적·요구형 표현은 감사 지적·판단 관점의 표현으로 구체화합니다.
   - "어떤 식의 제도적 개선" → "지적된 문제점과 이에 따른 제도적 개선 사항"
   - "설명해줘", "알려줘" → "무엇인가", "어떤 내용인가"
3. Validator가 지적한 실패 원인을 직접적으로 보완합니다.
   - 처분 정보 부족 → "처분 결과", "징계 수준", "신분상 조치" 강화
   - 기준 불명확 → "관련 규정", "기준", "적정성 판단" 강화
4. 감사보고서에 실제 등장할 가능성이 높은 용어를 우선 사용합니다.
   - 문제 발생 → 부적정 처리, 미준수, 위반 사례
   - 개선 요구 → 시정 조치, 제도 개선, 재발 방지 대책
   - 판단 요청 → 기준은 무엇인가, 수준은 무엇인가
5. 하나의 질문은 하나의 중심 판단을 갖는 단일 문장으로 재작성합니다.
6. 질문에 없는 날짜, 기관명, 법령, 판단 결과를 새로 추가하지 않습니다.
7. 질문에 이미 감사보고서에서 단독 쟁점으로 사용되는 핵심 감사 키워드가 포함된 경우,
   해당 키워드는 재작성 과정에서 삭제·완화·일반화하지 않고
   감사 지적의 중심 판단 요소로 명시적으로 유지·강조합니다.
   (예: 보수감액, 수의계약, 출장비, 시간외근무수당, 여비 과다 지급 등)

[출력 형식]
반드시 아래 JSON 형식을 준수하십시오.
JSON 외의 텍스트는 절대 출력하지 마십시오.

{{
  "rewrite_cot": [
    "Step 1: Validator가 문서를 거절한 핵심 원인을 요약한다.",
    "Step 2: 핵심 감사 키워드가 있는 경우 이를 중심 판단 축으로 고정하거나, 부족한 판단 기준을 보강한다.",
    "Step 3: 감사 지적과 판단 기준이 명확히 드러나는 형태로 질문을 재작성한다."
  ],
  "rewritten_question": "재작성된 질문 문자열"
}}

[입력 → 출력 예시]

입력:
기존 계획과는 달리 다르게 시공된 경우에 어떤식의 제도적 개선이나 조치가 이뤄지는지 설명해줘.

출력:
{{
  "rewrite_cot": [
    "Step 1: 질문이 추상적인 제도 개선 요구에 머물러 있어 검증기가 판단 근거 부족으로 거절했다.",
    "Step 2: 감사보고서에서 실제로 다루는 문제점 지적과 개선 요구 표현으로 구체화한다.",
    "Step 3: 시공 변경 사례에서 지적된 문제점과 개선 조치를 묻는 질문으로 재작성한다."
  ],
  "rewritten_question": "기존 계획과 다르게 시공된 사례에서 감사에서 지적된 문제점과 이에 따른 제도적 개선 또는 시정 조치는 무엇인가?"
}}

[입력 → 출력 예시]

입력:
감독여비를 실제일수보다 과다지급하게 된 경우에 대해서 여비규정이나 특별여비 규정을 설명해줘.

출력:
{{
  "rewrite_cot": [
    "Step 1: 질문은 규정 설명을 요구하지만 적용 기준이 불명확해 검증에 실패했다.",
    "Step 2: 핵심 감사 키워드인 '과다 지급'을 중심으로 적용 기준을 명확화한다.",
    "Step 3: 실제 근무일수 대비 과다 지급 사례에 적용되는 규정 기준을 묻는 질문으로 재작성한다."
  ],
  "rewritten_question": "감독여비를 실제 근무일수보다 과다 지급한 사례에서 적용되는 여비규정 또는 특별여비 규정의 기준은 무엇인가?"
}}
"""

rewriter_user = """
[Original Question]
{original_question}

[Current Question]
{current_question}

[Validator Feedback]
{feedback}
"""

rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", rewriter_system),
    ("human", rewriter_user)
])

rewriter_chain = (
    rewriter_prompt
    | llm_hcx
    | JsonOutputParser()
)

def rewriter(state: GraphState) -> dict:
    print("\n[NODE] REWRITE_QUERY_RUNNING")
    
    question = state["question"]
    original_question = state.get("original_question", question)
    
    validation_results = state.get("validation_results", [])
    failure_reasons = []
    
    for res in validation_results:
        if res.get("is_valid") == "no":
            cot = res.get("validator_cot", [])
            if cot:
                full_cot = "\n".join(cot)
                failure_reasons.append(f"- [Reasoning]:\n{full_cot}")
    
    feedback_text = "\n\n".join(failure_reasons[:3]) if failure_reasons else "명확한 실패 사유가 감지되지 않았습니다."

    print(f"[INFO] FEEDBACK PREVIEW: {failure_reasons[0][:100] if failure_reasons else 'NONE'}...")

    try:
        result = rewriter_chain.invoke({
            "original_question": original_question,
            "current_question": question,
            "feedback": feedback_text
        })
        
        new_question = result.get("rewritten_question", question)
        rewrite_cot = result.get("rewrite_cot", [])
        
        print(f"[INFO] REWRITTEN QUERY: {new_question}")
        
        return {
            "question": new_question,
            "rewrite_cot": rewrite_cot,
            "documents": [],
            "validation_results": [] 
        }
        
    except Exception as e:
        print(f"[ERROR] Rewrite Node: {e}")
        return {
            "question": question,
            "rewrite_cot": [f"Error during rewrite: {str(e)}"]
        }