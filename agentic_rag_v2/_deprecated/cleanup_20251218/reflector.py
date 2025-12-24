from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_naver import ChatClovaX
from config import Config
from state import AgentState
import json

# 평가용 모델 (ClovaX 사용)
reflector_llm = ChatClovaX(model=Config.LLM_MODEL, temperature=0.1, max_tokens=1024)


def reflect_answer(state: AgentState) -> AgentState:
    """
    [Node] 생성된 답변을 스스로 평가(Self-Reflection)합니다.
    """
    print(
        f"\n[Node] reflect_answer: 답변 평가 중... (Count: {state.get('reflection_count', 0)})"
    )

    # 평가 프롬프트
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
당신은 공공기관 감사 챗봇의 'Evaluator'입니다.
생성된 답변을 평가하고 다음 3가지 등급 중 하나를 부여하세요.

[등급 기준]
1. pass: 답변이 질문에 대해 정확하고 완전하며, [Context]에 기반함.
2. soft_pass: **핵심 정보는 포함**되어 있으나, 일부 내용이 미흡하거나 Context가 약간 부족함. (재시작보다는 그대로 내보내는 것이 나음)
   - 예: 질문은 "처분 기준"을 묻는데 "관련 법령"만 있고 "구체적 금액"이 없는 경우.
   - 예: 답변이 다소 짧지만 틀린 말은 아닌 경우.
3. fail: 답변이 질문과 **완전히 무관**하거나, [Context]에 없는 내용을 지어냄(Hallucination).

[출력 형식]
JSON 형식으로 출력하세요.
- status: "pass", "soft_pass", "fail" 중 택 1
- feedback: 평가 이유 (한글)

출력 예시:
{{
  "status": "soft_pass",
  "feedback": "처분 근거는 명시되었으나, 감경 사유에 대한 설명이 부족함."
}}
""",
            ),
            (
                "human",
                """
[Question]
{query}

[Context]
{context}

[Generated Answer]
{answer}
""",
            ),
        ]
    )

    chain = prompt | reflector_llm | StrOutputParser()

    try:
        context_text = "\n\n".join(state["documents"])
        result_text = chain.invoke(
            {
                "query": state["query"],
                "context": context_text,
                "answer": state["answer"],
            }
        )

        start = result_text.find("{")
        end = result_text.rfind("}") + 1
        result = json.loads(result_text[start:end])

        status = result.get("status", "pass")
        feedback = result.get("feedback", "적절함")

        print(f" -> 평가 결과: {status} ({feedback})")

        # 상태 업데이트 및 로직 분기 처리
        state["feedback"] = feedback

        if status == "fail":
            state["feedback"] = f"FAIL: {feedback}"
        elif status == "soft_pass":
            # Soft Pass는 재시도를 하지 않고, 답변에 태그를 달아 종료시킴
            # graph.py의 should_retry에서 'FAIL:' 접두사가 없으면 통과 처리함.
            # 단, 사용자에게 알리기 위해 answer에 태그 추가 (선택 사항)
            state["answer"] = (
                f"[참고: 답변 내용이 일부 부족할 수 있습니다]\n{state['answer']}"
            )
            state["feedback"] = f"SOFT_PASS: {feedback}"
        else:
            state["feedback"] = "PASS"

    except Exception as e:
        print(f" -> 평가 오류: {e}. 일단 통과시킵니다.")
        state["feedback"] = "PASS"

    return state
