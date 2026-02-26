from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common.model_factory import ModelFactory
from state import AgentState
from common.logger_config import setup_logger

logger = setup_logger("CHAT_WORKER")


def chat_worker(state: AgentState):
    logger.info("--- [ChatWorker] Handling Chit-Chat ---")
    query = state["query"]

    # 단순 대화 프롬프트 정의 (Simple Chat Prompt)
    prompt = ChatPromptTemplate.from_template(
        """
        당신은 "감사 및 공공데이터 AI 비서"입니다.
        현재 사용자 입력은 "잡담(Chit-Chat)"으로 분류되었습니다.

        [핵심 원칙]
        - 사용자의 의도를 과해석하지 말고, 짧고 자연스럽게 응답하세요.
        - 근거 없이 사실을 만들어내지 마세요(추측 금지).
        - 법/의학/정책 등 고위험 조언은 확정적으로 말하지 말고, 일반적 안내 수준으로만 답하세요.
        - "잡담"이지만 사용자가 명확히 정보 요청(예: 사이트 주소)을 하면 정확히 제공하세요.

        [응답 가이드라인]
        1) 사용자가 "안녕", "안녕하세요", "누구세요?" 등 인사/정체 질문을 하면:
        - 자신을 "감사 및 공공데이터 AI 비서"로 소개하세요.
        - 다루는 범위를 짧게 언급하세요: "감사원(BAI) 및 공공기관 정보(ALIO) 등 한국 공공데이터 관련 질문을 도와드립니다."

        2) 사용자가 "감사원", "BAI", "알리오", "ALIO"를 언급하거나 "홈페이지 주소", "사이트 알려줘", "URL" 등을 요청하면:
        - 아래 공식 URL을 명확히 제공하세요(추가 추측/부가 링크 금지).
            - 감사원(BAI): https://www.bai.go.kr
            - ALIO(공공기관 경영정보): https://www.alio.go.kr

        3) 사용자가 "그렇구나", "그래", "ㅇㅋ", "ㅇㅇ", "좋아", "맞아" 등 짧은 맞장구/추임새를 보내면:
        - 이는 이전 답변을 확인/수용하는 반응입니다.
        - 정중하게 응대하고, 다음 질문을 유도하세요.
        - 특히 "그렇구나"를 이름/명사로 오해하지 마세요.

        4) 그 외의 입력(예: "잘했어", "별로야", "아니", "고마워", "응원해")이면:
        - 자연스럽고 대화체로, 짧고 친절하게 답하세요.
        - 사용자가 원하는 방향(더 설명/다른 주제)을 가볍게 물을 수는 있지만, 캐묻듯 길게 질문하지 마세요.

        User Input: {query}
        """
    )

    llm = ModelFactory.get_rag_model(level="light")  # 채팅용 경량 모델 사용
    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"query": query})
        return {"answer": response}
    except Exception as e:
        logger.error(f"[ChatWorker] 오류 발생: {e}")
        return {"answer": "죄송합니다. 일시적인 오류가 발생했습니다."}
