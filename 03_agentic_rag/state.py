from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict):
    """
    Agentic RAG의 상태(State)를 정의하는 스키마입니다.
    그래프의 각 노드는 이 상태를 공유하고 업데이트합니다.
    """

    query: str  # 사용자의 원본 질문
    category: str  # 질문 분류 결과 ('search', 'stats', 'compare' 등)
    mode: str  # (New) 실행 모드: 'chat' | 'fast' (Simple Retrieval) | 'deep' (Complex RAG)
    # persona field removed
    documents: List[str]  # 검색된 문서 컨텍스트 리스트
    graph_context: List[str]  # (New) 그래프 DB 검색 결과 리스트
    sub_queries: List[str]  # 분해된 하위 질문 리스트
    messages: List[Dict[str, Any]]  # 대화 기록 (Chat History)
    summary: str  # (New) 대화 내용을 요약한 장기 기억 (Long-term Memory)
    persist_documents: List[Any]  # (New) 이전 턴의 문서 컨텍스트 (Reference용)

    # 슈퍼바이저 필드 (계획 및 라우팅)
    plan: List[str]  # 계획된 단계 리스트 (예: ["search", "verify", "answer"])
    next_step: str  # 다음에 실행할 노드 이름
    retrieval_count: int  # 재검색 횟수 카운터
    grade_status: str  # 문서 평가 결과 (success/fail)
    worker_output: Any  # 작업자(Worker)의 실행 결과
    metadata_filters: Dict[str, Any]  # (선택 사항) 통계/검색을 위한 메타데이터 필터

    # 고급 RAG 필드 (Phase 2: 적응형 검색)
    search_query: str  # 재작성된 검색 쿼리 (원본 query와 구분)
    selected_fields: List[str]  # 필드 선택기가 추출한 메타데이터 필드
    selected_fields_cot: List[str]  # 필드 선택 추론 과정 (CoT)
    is_valid: str  # Validator 결과 ("yes"/"no")
    validator_cot: List[str]  # 검증 추론 과정 (CoT)
    analysis_decision: str  # 전략 결정 결과 ("rewrite_query"/"update_fields")
    strategy_decider_cot: List[str]  # 전략 결정 추론 과정 (CoT)

    # Modular RAG Verification (Phase 4)
    is_hallucinated: str  # "yes" or "no"
    is_useful: str  # "yes" or "no"

    # SOP 필드 (Phase 3: 표준감사절차)
    facts: dict  # 추출된 사실 관계
    matched_regulations: list  # 매칭된 관련 법령
    sop_context: str  # (New) 생성기에 제공할 SOP/규정 컨텍스트
    compliance_result: str  # 규정 위반 여부 판정 결과

    # 결과 필드
    answer: str  # 최종 생성된 답변
    reflection_count: int  # 답변 재생성 횟수 (무한 루프 방지)
    feedback: str  # Reflector가 제공하는 피드백 (Generator가 참고)
