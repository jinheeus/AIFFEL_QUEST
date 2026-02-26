from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from common.model_factory import ModelFactory
from common.logger_config import setup_logger

logger = setup_logger("MEMORY")


# --- 요약 프롬프트 (Summary Prompt) ---
SUMMARY_SYSTEM_PROMPT = """
[Role]
You are a Conversation Summarizer for an Audit RAG Chatbot.
Your goal is to maintain a concise running summary of the conversation to preserve context while saving tokens.

[Input]
1. Current Summary: The existing summary of the conversation history.
2. New Lines: The most recent conversation turns that need to be added.

[Output]
Update the 'Current Summary' by incorporating the key information from 'New Lines'.
- Keep the summary concise (under 200 words).
- Preserve important entities (Case names, Regulation Article numbers, Specific user intent).
- If 'New Lines' is empty, return the 'Current Summary' as is.
"""


def summarize_conversation(state: AgentState) -> dict:
    """
    [Node] 대화 기록을 요약하여 토큰을 절약합니다.
    대화의 가장 최근 부분(마지막 요약 이후 추가된 내용)을 가져와 요약을 업데이트합니다.
    """
    logger.info("--- [Node] Summarize Conversation (Memory) ---")

    current_summary = state.get("summary", "")
    messages = state.get("messages", [])

    # 1. 요약 필요 여부 확인 (Check if summarization is needed)
    # 전략: 기록이 길어지면(> 6), 마지막 2개 메시지(현재 턴)를 제외한 모든 내용을 요약합니다.
    # 이는 최근 컨텍스트는 그대로 유지하면서 나머지를 압축하기 위함입니다.
    if len(messages) <= 6:
        logger.info(" -> History short, skipping summary.")
        # [UX Fix] State Passthrough for answer
        return {"summary": current_summary, "answer": state.get("answer", "")}

    # 2. 요약할 메시지 슬라이싱 (Slice messages to summarize)
    # 마지막 2개는 즉각적인 컨텍스트를 위해 남겨두고 나머지를 요약합니다.
    # 참고: 실제 시스템에서는 'last_summarized_index' 포인터가 필요할 수 있습니다.
    # 여기서는 매번 처음부터 다시 요약하거나 업데이트하는 방식을 사용합니다.
    # 업데이트 방식이 좋지만, 프론트엔드 입력에 의해 'messages' 리스트가 리셋될 수 있으므로 주의가 필요합니다.

    # 견고한 접근법 (Robust Approach):
    # 마지막 2개를 제외한 모든 메시지를 요약합니다.
    to_summarize = messages[:-2]

    text_to_summarize = "\n".join(
        [f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in to_summarize]
    )

    # 3. LLM 호출 (Invoke LLM)
    llm = ModelFactory.get_rag_model(level="heavy", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUMMARY_SYSTEM_PROMPT),
            (
                "human",
                "Current Summary: {current_summary}\n\nNew Lines to Add: {new_lines}",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        # 참고: 매번 전체 'to_summarize' 블록을 다시 요약한다면,
        # 점진적(incremental) 방식이 아닌 이상 'current_summary'를 재귀적으로 전달할 필요가 없습니다.
        # 이미 요약된 부분을 알 수 없으므로(포인터 부재),
        # 그냥 'to_summarize' 블록에 대해 새로운 요약을 생성합니다.
        # 이 방식이 중복 방지에 더 안전합니다.

        # 'Fresh Summary' 모드를 위해 프롬프트 재정의
        fresh_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SUMMARY_SYSTEM_PROMPT),
                ("human", "Summarize this conversation:\n{new_lines}"),
            ]
        )

        chain = fresh_prompt | llm | StrOutputParser()
        new_summary = chain.invoke({"new_lines": text_to_summarize})

        logger.info(f" -> Summary Updated: {new_summary[:50]}...")

        # [UX Fix] Return answer for frontend streaming
        return {"summary": new_summary, "answer": state.get("answer", "")}

    except Exception as e:
        logger.error(f" -> Summary Generation Failed: {e}")
        return {"summary": current_summary, "answer": state.get("answer", "")}
