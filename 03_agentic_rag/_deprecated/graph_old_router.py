from langgraph.graph import StateGraph, END
from state import AgentState
from modules.router import analyze_query
from modules.retriever import retrieve_documents
from modules.stat_engine import analyze_stats
from modules.generator import generate_answer
from modules.reflector import reflect_answer
from modules.decomposer import decompose_query
from modules.graph_retriever import retrieve_graph_context


from modules.sop import (
    extract_facts,
    match_regulations,
    evaluate_compliance,
    determine_disposition,
)
from modules.adversarial import defense_agent, prosecution_agent, judge_verdict


def route_query(state: AgentState):
    """
    analyze_query 노드의 결과(category)에 따라 다음 경로를 결정하는 조건부 함수입니다.
    """
    category = state["category"]
    if category == "search":
        return "decompose_query"
    elif category == "stats":
        return "analyze_stats"
    elif category == "compare":
        return "decompose_query"
    elif category == "judgment":
        # Judgment도 검색이 필요하므로 Decompose -> Retrieve 경로를 따름
        return "decompose_query"
    else:
        return "decompose_query"


def route_generation(state: AgentState):
    """
    Retrieval 이후, 일반 생성(Generator)으로 갈지 SOP(Audit Logic)로 갈지 결정
    """
    category = state.get("category")
    if category == "judgment":
        print(" -> [Route] SOP(심층 감사) 모드로 진입합니다.")
        return "extract_facts"
    else:
        return "generate_answer"


def should_retry(state: AgentState):
    """
    Reflector의 피드백을 보고 재시도 여부를 결정합니다.
    """
    feedback = state.get("feedback", "")
    count = state.get("reflection_count", 0)

    if count >= 3:
        print(" -> [Stop] 최대 재시도 횟수 초과. 종료합니다.")
        return END

    if feedback.startswith("FAIL:"):
        print(" -> [Retry] 답변 품질 미달. 다시 생성합니다.")
        return "generate_answer"
    else:
        print(" -> [Pass] 답변 통과.")
        return END


# 1. 그래프 빌더 초기화
workflow = StateGraph(AgentState)

# 2. 노드 추가
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("decompose_query", decompose_query)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("retrieve_graph_context", retrieve_graph_context)
workflow.add_node("analyze_stats", analyze_stats)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("reflect_answer", reflect_answer)

# SOP Nodes
workflow.add_node("extract_facts", extract_facts)
workflow.add_node("match_regulations", match_regulations)
workflow.add_node("evaluate_compliance", evaluate_compliance)
workflow.add_node("determine_disposition", determine_disposition)

# 3. 엣지 연결
workflow.set_entry_point("analyze_query")

workflow.add_conditional_edges(
    "analyze_query",
    route_query,
    {"decompose_query": "decompose_query", "analyze_stats": "analyze_stats"},
)

workflow.add_edge("decompose_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "retrieve_graph_context")

# Conditional Edge: Graph Search -> (Gen OR SOP)
workflow.add_conditional_edges(
    "retrieve_graph_context",
    route_generation,
    {"generate_answer": "generate_answer", "extract_facts": "extract_facts"},
)

# SOP Chain
workflow.add_edge("extract_facts", "match_regulations")
workflow.add_edge("match_regulations", "evaluate_compliance")
workflow.add_edge("evaluate_compliance", "determine_disposition")


def route_disposition(state: AgentState):
    """
    SOP 결과가 'Violated'이면 Adversarial Audit(변론-반박-판결)으로 진입.
    아니면 종료.
    """
    compliance = state.get("compliance_result", {})
    # compliance가 dict가 아니거나 status가 없으면 안전하게 종료
    if isinstance(compliance, dict) and compliance.get("status") == "Violated":
        print(" -> [Route] 규정 위반 감지. Adversarial Audit(변론 절차) 개시.")
        return "defense_agent"
    else:
        print(" -> [Route] 위반 아님 또는 판단 불가. 종료.")
        return END


# 1. 그래프 빌더 초기화
workflow = StateGraph(AgentState)

# 2. 노드 추가 (Nodes)
# --- 기본 RAG 노드 ---
workflow.add_node("analyze_query", analyze_query)  # 질문 분석 (Router)
workflow.add_node("decompose_query", decompose_query)  # 질문 분해 (Decomposer)
workflow.add_node("retrieve_documents", retrieve_documents)  # 벡터 검색 (Vector DB)
workflow.add_node(
    "retrieve_graph_context", retrieve_graph_context
)  # 그래프 검색 (Graph DB)
workflow.add_node("analyze_stats", analyze_stats)  # 통계 분석 (Code Interpreter)
workflow.add_node("generate_answer", generate_answer)  # 답변 생성 (Generic)
workflow.add_node("reflect_answer", reflect_answer)  # 자기 성찰 (Reflector)

# --- SOP (표준감사절차) 노드 ---
workflow.add_node("extract_facts", extract_facts)  # 팩트 추출
workflow.add_node("match_regulations", match_regulations)  # 규정 매칭
workflow.add_node("evaluate_compliance", evaluate_compliance)  # 준수 여부 판단
workflow.add_node("determine_disposition", determine_disposition)  # 처분 결정

# --- Adversarial Audit (적대적 감사) 노드 ---
workflow.add_node("defense_agent", defense_agent)  # 피감기관 (변호)
workflow.add_node("prosecution_agent", prosecution_agent)  # 감사관 (검사)
workflow.add_node("judge_verdict", judge_verdict)  # 위원장 (판결)

# 3. 엣지 연결 (Edges)
# [Start] -> [Router]
workflow.set_entry_point("analyze_query")

# [Router] -> [Branch]
workflow.add_conditional_edges(
    "analyze_query",
    route_query,
    {"decompose_query": "decompose_query", "analyze_stats": "analyze_stats"},
)

# [Decomposer] -> [Vector Search]
workflow.add_edge("decompose_query", "retrieve_documents")

# [Vector Search] -> [Graph Search] (Hybrid)
workflow.add_edge("retrieve_documents", "retrieve_graph_context")

# [Graph Search] -> [Conditional: Generic vs SOP]
workflow.add_conditional_edges(
    "retrieve_graph_context",
    route_generation,
    {"generate_answer": "generate_answer", "extract_facts": "extract_facts"},
)

# [SOP Chain]: Fact -> Reg -> Compliance -> Disposition
workflow.add_edge("extract_facts", "match_regulations")
workflow.add_edge("match_regulations", "evaluate_compliance")
workflow.add_edge("evaluate_compliance", "determine_disposition")

# [Disposition] -> [Conditional: End vs Trial]
workflow.add_conditional_edges(
    "determine_disposition",
    route_disposition,
    {"defense_agent": "defense_agent", END: END},
)

# [Adversarial Chain]: Defense -> Prosecution -> Judge -> End
workflow.add_edge("defense_agent", "prosecution_agent")
workflow.add_edge("prosecution_agent", "judge_verdict")
workflow.add_edge("judge_verdict", END)

# [Stats Route]
workflow.add_edge("analyze_stats", "generate_answer")

# [Generation] -> [Reflection]
workflow.add_edge("generate_answer", "reflect_answer")

# [Reflection] -> [Loop or End]
workflow.add_conditional_edges(
    "reflect_answer", should_retry, {"generate_answer": "generate_answer", END: END}
)

# 4. 그래프 컴파일
app = workflow.compile()
