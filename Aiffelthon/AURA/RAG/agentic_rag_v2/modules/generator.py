from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common.model_factory import ModelFactory
from common.logger_config import setup_logger
from state import AgentState

logger = setup_logger("GENERATOR")

# 답변 생성용 모델 (Analyst Persona -> Deep Reasoning Model)
generator_llm = ModelFactory.get_rag_model(level="reasoning", temperature=0.1)


def generate_answer(state: AgentState) -> AgentState:
    """
    [Node] 검색된 문서 또는 통계 결과를 바탕으로 최종 답변을 생성합니다.
    페르소나에 따라 적절한 프롬프트를 사용하여 응답을 구성합니다.
    """
    logger.info("generate_answer: 답변 생성 중...")
    # 0. Reflection Count 증가
    current_count = state.get("reflection_count", 0)
    state["reflection_count"] = current_count + 1

    # 1. 단일 시스템 프롬프트 정의 (Audit Assistant)
    system_msg = """
당신은 "AURA 감사 분석관(AURA Audit Analyst)"입니다.
당신의 역할은 사용자가 제시한 사례를 조사하고, 공공감사 규정(SOP)에 근거하여 사실관계를 분석하며 위반 여부를 진단하는 것입니다.

[핵심 역할: 분석가(Analyst)]
- 당신은 최종 보고서를 작성하는 '작가'가 아니라, 현 상황을 냉철하게 분석하고 규정 준수 여부를 판단하는 '전문 분석가'입니다.
- 분석 결과 위반 사항이나 중대한 리스크가 발견되면, 사용자가 '공식 감사 보고서'를 작성하도록 유도해야 합니다.

[정보 우선순위]
1순위: [Context] (검색된 문서 내용)
2순위: [Previous Context] (이전 대화에서 저장된 문서)
3순위: [Chat History] (참고용 대화 기록)
※ 절대 이 우선순위를 어기지 마십시오. 특히 대화 기록에만 의존해 법령이나 규정을 단정 짓지 마십시오.

[작성 가이드라인 - 필독]
1. **자연스러운 대화체**: "진단 결과:", "문제점:", "결론:" 같은 헤더나 말머리를 절대 사용하지 마십시오. 전문 지식을 가진 동료와 대화하듯 자연스러운 문장으로 시작하십시오.
2. **영어 사용 금지**: "Diagnosis", "Violation", "Fact" 등의 영어 단어나 헤더를 답변에 섞지 마십시오. 모든 내용은 품격 있는 한국어로 작성합니다.
3. **직설적인 답변**: 인사말이나 사족 없이 사용자의 질문에 대한 핵심 분석 결과를 즉시 제시하십시오.
4. **구조적 논리, 비구조적 형식**: 내용은 논리적이어야 하지만, 형식은 딱딱한 보고서 형태가 아닌 흐르는 글이어야 합니다. 핵심 팩트는 문장 속에 자연스럽게 녹여내십시오.

[필수 트리거 문구]
- 분석 결과 관련 사례가 존재하거나 규정 위반이 확인될 경우, 답변의 마지막에 반드시 아래 문구만 정확히 덧붙이십시오.
  > "이러한 사례를 바탕으로 공식 감사 보고서 작성이 필요하시면 알려주세요."

[예외 상황]
- 관련 정보를 찾을 수 없는 경우 "관련된 사례를 찾지 못했습니다"라고 정중히 답하고 추가 정보를 요청하십시오.
- 링크나 파일 정보가 [Context]에 있다면 문장 끝에 깔끔하게 나열하십시오.
"""

    # 피드백이 있으면 시스템 프롬프트에 추가
    feedback = state.get("feedback", "")
    if feedback and feedback.startswith("FAIL:"):
        logger.info(f" -> 피드백 반영하여 재생성 중: {feedback}")
        system_msg += f"\n\n[이전 답변에 대한 피드백]\n{feedback}\n이 피드백을 반영하여 답변을 수정하세요."

    # 컨텍스트 형식화 (Context Format)
    # 문자열과 Document 객체를 모두 처리합니다.
    docs_text = []
    documents = state.get("documents", [])
    for d in documents:
        if hasattr(d, "page_content"):
            # 생성기를 위한 메타데이터 포함 (Include Metadata)
            meta = d.metadata

            # 주요 필드 추출 (Extract key fields)
            date_info = meta.get("date", "Unknown")
            title_info = meta.get(
                "title", meta.get("source_type", "Unknown")
            )  # source_type을 제목 대체제로 사용
            company = meta.get("company_name", "Unknown")
            category = meta.get("category", meta.get("cat", "Unknown"))
            file_path = meta.get("file_path", "")
            download_url = meta.get("download_url", "")

            # 견고한 정보 블록 구성 (Construct robust info block)
            meta_block = f"""[[문서 정보]]
- 날짜: {date_info}
- 제목/출처: {title_info}
- 기관명: {company}
- 카테고리: {category}"""

            if file_path:
                meta_block += f"\n- 파일경로: {file_path}"
            if download_url:
                meta_block += f"\n- 다운로드: {download_url}"

            doc_str = f"{meta_block}\n\n[[내용]]\n{d.page_content}"
            docs_text.append(doc_str)
        else:
            docs_text.append(str(d))

    context_text = "\n\n---\n\n".join(docs_text)

    # [Safety] 컨텍스트가 비어있을 경우, 명시적인 플레이스홀더를 제공하여
    # API 400 에러나 환각(Hallucination)을 방지합니다.
    if not context_text.strip():
        context_text = "검색된 관련 문서가 없습니다. (No documents found)"

    # Graph Context 병합
    if state.get("graph_context"):
        graph_text = "\n".join(state["graph_context"])
        context_text += f"\n\n[Graph Knowledge]\n{graph_text}"

    # SOP Context 병합 (New)
    if state.get("sop_context"):
        sop_text = state["sop_context"]
        context_text += f"\n\n[Standard Operating Procedures (SOP)]\n{sop_text}"

    # [이전 컨텍스트 유지 (Previous Context Persistence)]
    # 이전 턴에서 검색되었던 문서들을 제공하여 "아까 그 문서", "2번 문서" 등의 참조를 해결합니다.
    persist_docs = state.get("persist_documents", [])
    logger.debug(f" -> [Generator Debug] Persisted Docs Available: {len(persist_docs)}")

    if persist_docs:
        p_docs_text = []
        for d in persist_docs:
            if hasattr(d, "page_content"):
                # 이전 컨텍스트를 위한 단순 포맷 (Metadata가 핵심)
                meta = d.metadata
                title_info = meta.get("title", meta.get("source_type", "Unknown"))
                date_info = meta.get("date", "Unknown")
                file_path = meta.get("file_path", "")
                download_url = meta.get("download_url", "")

                p_doc_str = f"[[이전 문서]]\n- 제목: {title_info}\n- 날짜: {date_info}\n- 파일: {file_path}\n- 다운로드: {download_url}\n[[내용 요약]]\n{d.page_content[:200]}..."
                p_docs_text.append(p_doc_str)
            else:
                p_docs_text.append(str(d))

        if p_docs_text:
            context_text += (
                f"\n\n[Previous Context (Reference Only)]\n"
                + "\n---\n".join(p_docs_text)
            )

    # 채팅 기록 병합 (Chat History Merger) - [Modified] 하이브리드 메모리 (요약 + 최신 대화)
    chat_history_str = ""
    summary = state.get("summary", "")

    if summary:
        chat_history_str += f"[Previous Conversation Summary]\n{summary}\n\n"

    if state.get("messages"):
        history_msgs = []
        # 최근 4개 메시지만 사용 (Token 절약) - 이미 Summary가 있으므로 짧게 유지합니다.
        for msg in state["messages"][-4:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            history_msgs.append(f"{role}: {content}")

        chat_history_str += "[Recent Messages]\n" + "\n".join(history_msgs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            (
                "human",
                """
[Context]
{context}

[Chat History]
{chat_history}

[Question]
{query}
""",
            ),
        ]
    )

    chain = prompt | generator_llm | StrOutputParser()
    answer = chain.invoke(
        {
            "context": context_text,
            "chat_history": chat_history_str,
            "query": state["query"],
        }
    )

    # [Source Citation Auto-Append]
    # 답변 하단에 [참고 문서] 섹션을 자동으로 추가하여 신뢰도를 높입니다.
    if documents:
        source_data = {}  # {title: {'link': link, 'count': count}}

        for d in documents:
            if hasattr(d, "page_content"):
                meta = d.metadata
                # 제목 추출 (없으면 source_type 사용)
                title = meta.get("title") or meta.get("source_type") or "무제 문서"
                # 링크 추출 (download_url 우선, 없으면 file_path)
                link = meta.get("download_url") or meta.get("file_path")

                # 데이터 집계
                if title not in source_data:
                    source_data[title] = {"link": link, "count": 1}
                else:
                    source_data[title]["count"] += 1

        if source_data:
            answer += "\n\n---\n**[참고 문서]**\n"
            for title, data in source_data.items():
                link = data["link"]
                count = data["count"]

                # 표시할 제목 (2건 이상이면 건수 표시)
                display_title = title
                if count > 1:
                    display_title = f"{title} ({count}건 관련)"

                if link and link.startswith("http"):
                    answer += f"- [{display_title}]({link})\n"
                elif link:
                    # 로컬 경로거나 URL이 아닌 경우 텍스트로 표기 (또는 필요시 file:// 처리)
                    answer += f"- {display_title} (파일: {link})\n"
                else:
                    answer += f"- {display_title}\n"

    state["answer"] = answer
    return state
