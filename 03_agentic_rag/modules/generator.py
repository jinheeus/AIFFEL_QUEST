from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_naver import ChatClovaX
from config import Config
from state import AgentState

# 답변 생성용 모델
generator_llm = ChatClovaX(model=Config.LLM_MODEL, temperature=0.1, max_tokens=2048)


def generate_answer(state: AgentState) -> AgentState:
    """
    [Node] 검색된 문서 또는 통계 결과를 바탕으로 최종 답변을 생성합니다.
    페르소나에 따라 적절한 프롬프트를 사용하여 응답을 구성합니다.
    """
    print(f"\n[Node] generate_answer: 답변 생성 중...")
    # 0. Reflection Count 증가
    current_count = state.get("reflection_count", 0)
    state["reflection_count"] = current_count + 1

    # 1. 페르소나별 시스템 프롬프트 정의 (Dynamic View)
    # 1. 단일 시스템 프롬프트 정의 (Audit Assistant)
    system_msg = """
당신은 **AURA 감사 비서(Audit Assistant)**입니다.
공공기관 감사 규정, 사례, 법령 정보에 기반하여 정확하고 신뢰할 수 있는 답변을 제공하는 것이 임무입니다.

[핵심 원칙]
1. **증거 기반(Factuality)**: 모든 답변은 우선적으로 제공된 [Context]에 명시된 내용에 기반해야 합니다. 단, 사용자가 **이전 대화 내용(Chat History)**을 묻거나 참조하는 경우(예: "아까 말한 3번 문서"), [Chat History]의 내용을 바탕으로 답변할 수 있습니다.
2. **구조화된 답변(Structure)**: 가독성을 높이기 위해 두괄식 요약, 불릿 포인트, 번호 매기기 등을 적극 활용하십시오.
3. **전문성(Expertise)**: 감사관이나 실무자가 이해할 수 있는 전문적인 용어를 사용하되, 설명은 명확하게 하십시오.

[답변 작성 가이드]
1. 질문에 대한 **핵심 결론**을 첫 문장에서 명확히 제시하십시오.
2. [Context]에서 **관련 근거(규정명, 조항, 사례 등)**를 구체적으로 인용하십시오.
3. **리스트 요청 처리 (List Handling)**: 사용자가 "3개만 알려줘", "리스트 보여줘" 등 목록을 요청하면, [Context]에 있는 문서 정보를 **무조건** 요약하여 나열하십시오. 질문이 모호하더라도 "검색된 문서는 다음과 같습니다"라고 시작하며 내용을 보여주십시오. 내용이 중복되어도 있는 그대로 보여주십시오.
4. 질문 해결에 필요한 정보가 **[Context], [Previous Context], [Chat History]** 중 어디에도 없을 경우에만 "관련 정보를 찾을 수 없습니다"라고 명시하십시오.
4. **링크/파일 요청**: 사용자가 "링크", "파일", "원본", "다운로드" 등을 요청하면, 반드시 [Context]의 **[[문서 정보]]** 또는 **[[이전 문서]]** 섹션에 있는 **'파일경로'** 또는 **'다운로드'** URL을 그대로 제공하십시오. 링크가 있다면 절대 "없다"고 하지 마십시오.
5. 불필요한 사족이나 인사는 생략하고, 정보 전달에 집중하십시오.
"""

    # 피드백이 있으면 시스템 프롬프트에 추가
    feedback = state.get("feedback", "")
    if feedback and feedback.startswith("FAIL:"):
        print(f" -> 피드백 반영하여 재생성 중: {feedback}")
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
    print(f" -> [Generator Debug] Persisted Docs Available: {len(persist_docs)}")

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

    state["answer"] = answer
    return state
