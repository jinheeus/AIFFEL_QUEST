from state import AgentState
from .vector_retriever import get_retriever


def retrieve_documents(state: AgentState) -> AgentState:
    """
    [Node] 'search' 카테고리일 때 실행. Vector DB에서 관련 문서를 검색합니다.
    (Hybrid Retrieval Engine 사용)
    """
    print("\n[Node] retrieve_documents: 문서 검색 중... (Hybrid Retrieval Engine)")

    # Lazy Load Retriever
    rag_pipeline = get_retriever()

    try:
        # 1. 문서 검색을 위한 쿼리 결정 (우선순위: search_query -> sub_queries -> query)
        # 'search_query'가 있으면 최우선으로 사용합니다.
        if state.get("search_query"):
            queries = [state["search_query"]]
        elif state.get("sub_queries"):
            queries = state["sub_queries"]
        else:
            queries = [state["query"]]

        all_docs = []

        # 2. 쿼리별 반복 검색 수행 (Iterative Search)
        for q in queries:
            # 선택된 필드(Selected Fields)를 쿼리에 주입하여 문맥 보강 (Context Injection)
            selected_fields = state.get("selected_fields", [])
            if selected_fields:
                enriched_q = f"{q} (Focus: {', '.join(selected_fields)})"
                print(f" -> Sub-Search: '{enriched_q}'")
                search_q = enriched_q
            else:
                print(f" -> Sub-Search: '{q}'")
                search_q = q

            # 복합 질문일 경우 top_k를 줄여서 토큰 절약 (기본 5 -> 3)
            k = 3 if len(queries) > 1 else 5

            # (New) 메타데이터 필터 적용 (Hybrid Retrieval)
            filters = state.get("metadata_filters", {})
            if filters:
                print(f" -> 필터 적용 (Applying Filters): {filters}")

            docs = rag_pipeline.search_and_merge(search_q, top_k=k, filters=filters)
            if docs:
                all_docs.extend(docs)

        # 3. 중복 제거 (Content-based deduplication)
        # 문서 객체의 내용을 기준으로 중복을 제거합니다.
        unique_docs = []
        seen_content = set()

        for d in all_docs:
            if d.page_content not in seen_content:
                unique_docs.append(d)
                seen_content.add(d.page_content)

        if not unique_docs:
            # [Fallback] 검색 결과가 없지만 이전 턴의 문서(Persistent Docs)가 있는 경우 사용합니다.
            # 예: "2번 문서에 대해 더 알려줘"라고 했을 때 검색어 매칭이 실패하더라도,
            # 사용자는 명확히 이전 문서를 지칭하고 있기 때문입니다.
            persist_docs = state.get("persist_documents", [])
            if persist_docs:
                print(
                    f" -> [Fallback] Search returned 0 results. Using {len(persist_docs)} persisted documents."
                )
                state["documents"] = persist_docs
            else:
                state["documents"] = ["검색 결과가 없습니다."]
        else:
            state["documents"] = unique_docs

        print(f" -> 검색 완료: 총 {len(unique_docs)}개 문서 병합됨")

    except Exception as e:
        print(f" -> 검색 실패 (Search Failed): {e}")
        state["documents"] = [f"검색 중 오류 발생: {str(e)}"]

    return state
