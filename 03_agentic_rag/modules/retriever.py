from state import AgentState
from .vector_retriever import get_retriever


def retrieve_documents(state: AgentState) -> AgentState:
    """
    [Node] 'search' 카테고리일 때 실행. Vector DB에서 관련 문서를 검색합니다.
    """
    print(f"\n[Node] retrieve_documents: 문서 검색 중... (High-Context Engine)")

    # Lazy Load Retriever
    rag_pipeline = get_retriever()

    try:
        # 1. 문서 검색을 위한 쿼리 결정 (우선순위: search_query -> sub_queries -> query)
        queries = []
        if state.get("search_query"):
            queries = [state["search_query"]]
        elif state.get("sub_queries"):
            queries = state["sub_queries"]
        else:
            queries = [state["query"]]

        all_docs = []

        # 2. 쿼리별 반복 검색 수행
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
                print(f" -> Applying Filters: {filters}")

            docs = rag_pipeline.search_and_merge(search_q, top_k=k, filters=filters)
            if docs:
                all_docs.extend(docs)

        # 3. 중복 제거 (내용 기반 단순 set 연산)
        unique_docs = list(set(all_docs))

        if not unique_docs:
            state["documents"] = ["검색 결과가 없습니다."]
        else:
            state["documents"] = unique_docs

        print(f" -> 검색 완료: 총 {len(unique_docs)}개 문서 병합됨")

    except Exception as e:
        print(f" -> 검색 실패: {e}")
        state["documents"] = [f"검색 중 오류 발생: {str(e)}"]

    return state
