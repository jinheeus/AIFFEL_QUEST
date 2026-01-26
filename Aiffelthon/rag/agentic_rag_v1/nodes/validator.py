from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict, Any
from langchain_core.documents import Document
from state import GraphState
from utils.llm import llm_hcx

validator_system = """
[역할]
당신은 감사보고서 기반 RAG 시스템의 문서 유효성 검증기(Validator)입니다.

[목표]
사용자 질문(question)과 단일 감사보고서 문서(document)를 비교하여,
해당 문서가 질문에 대해 단정적인 답변을 제공할 수 있는 유효 문서인지
매우 엄격하게 판단하십시오.

[핵심 판단 기준]
아래 조건을 모두 충족할 때만 유효(is_valid = "yes")로 판단합니다.

1. 질문 적합성
- 질문이 요구하는 핵심 정보의 유형(사실, 판단, 결과, 기준 등)이 명확해야 합니다.

2. 정보 존재성
- 문서에 질문이 요구하는 정보가 직접적이고 명시적으로 존재해야 합니다.
- 유사한 맥락, 배경 설명, 일반론만 존재하는 경우는 불충족입니다.

3. 답변 완결성
- 문서 내용만으로 질문에 대해 단정적이고 확정적인 답변이 가능해야 합니다.
- 추론, 보완 검색, 다른 문서의 결합이 필요하면 불충족입니다.

4. 조건 일치성
- 질문에 특정 기간, 연도, 대상, 행위 조건이 포함된 경우
  문서 내용이 그 조건을 정확히 만족해야 합니다.
- 조건 불일치 또는 불명확한 경우는 불충족입니다.

[판정 원칙]
- 하나라도 불충족되면 is_valid는 반드시 "no"입니다.
- 애매하거나 판단이 어려운 경우는 항상 "no"로 판단하십시오.
- 관대하게 판단하지 마십시오.

[출력 형식]
출력은 반드시 아래 JSON 형식을 정확히 따라야 합니다.

{{
  "is_valid": "yes" 또는 "no",
  "validator_cot": [
    "Step 1: 질문이 요구하는 핵심 정보의 유형과 답변 조건을 명확히 식별한다.",
    "Step 2: 문서에서 해당 정보가 직접적이고 명시적으로 존재하는지 확인한다.",
    "Step 3: 문서 내용만으로 질문에 대해 단정적인 답변이 가능한지 판단한다."
  ]
}}

[출력 규칙]
- is_valid 값은 반드시 소문자 문자열 "yes" 또는 "no"만 허용됩니다.
- validator_cot는 반드시 단계별 사고 흐름을 나타내는 문자열 리스트여야 합니다.
- validator_cot는 최소 3단계 이상 작성해야 합니다.
- JSON 외의 텍스트는 절대 출력하지 마십시오.

[출력 예시 "is_valid": "yes"]
{{
  "is_valid": "yes",
  "validator_cot": [
    "Step 1: 질문은 출장비 부당 집행과 관련하여 실제로 내려진 징계 처분의 내용을 요구하고 있다.",
    "Step 2: 문서에는 출장비 부당 집행 사실과 함께 정직 1개월이라는 구체적인 징계 처분이 명시되어 있다.",
    "Step 3: 문서 내용만으로 질문에 대해 단정적이고 확정적인 답변이 가능하다고 판단했다."
  ]
}}

[출력예시 "is_valid": "no"]
{{
  "is_valid": "no",
  "validator_cot": [
    "Step 1: 질문은 출장비 부당 집행에 대해 실제로 내려진 징계 처분의 존재 여부를 확인하려는 목적이다.",
    "Step 2: 문서에는 출장비 집행의 문제점과 개선 권고만 제시되어 있고, 징계 처분에 대한 정보는 존재하지 않는다.",
    "Step 3: 문서 내용만으로 질문에 대해 단정적인 답변을 제공할 수 없다고 판단했다."
  ]
}}
"""

validator_user = """
[Question]
{question}

[Document]
{document}
"""

validator_template = ChatPromptTemplate.from_messages([
    ("system", validator_system),
    ("human", validator_user)
])

validator_chain = (
    validator_template
    | llm_hcx
    | JsonOutputParser()
)

def validator(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state["documents"]
    validated_docs = state.get("validated_documents", [])
    
    if not documents:
        return {"validation_results": [], "retry_count": state.get("retry_count", 0) + 1}

    batch_inputs = [{"question": question, "document": doc.page_content} for doc in documents]
    batch_results = validator_chain.batch(batch_inputs)
    
    new_valid_docs = []
    for doc, result in zip(documents, batch_results):
        if result.get("is_valid") == "yes":
            doc.metadata["validator_cot"] = result.get("validator_cot")
            new_valid_docs.append(doc)

    return {
        "validated_documents": validated_docs + new_valid_docs,
        "validation_results": batch_results,
        "retry_count": state.get("retry_count", 0) + 1,
        "documents": [] 
    }