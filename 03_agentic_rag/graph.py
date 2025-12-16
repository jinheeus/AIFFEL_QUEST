from langgraph.graph import StateGraph, END
from state import AgentState
import sys
import os

sys.path.append(
    os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)  # Add root for common import

# 모듈 임포트
from modules.supervisor import supervisor_node
from modules.chat_worker import chat_worker
from modules.retriever import retrieve_documents
from modules.graph_retriever import retrieve_graph_context
from modules.generator import generate_answer
from modules.reflector import reflect_answer

# 고급 적응형 라이브러리 (Field Selector, Validator, Strategy Decider)
from modules.advanced_library import (
    field_selector,
    validator,
    strategy_decider,
)

# SOP 모듈 (감사 절차)
from modules.sop import (
    extract_facts,
    match_regulations,
    evaluate_compliance,
    determine_disposition,
)

# 적대적 감사 모듈 (변호/검사/판사)
from modules.adversarial import defense_agent, prosecution_agent, judge_verdict


# --- Conditional Logic ---


def route_supervisor(state: AgentState):
    """
    Supervisor의 결정(next_step)에 따라 경로를 분기합니다.
    """
    next_step = state.get("next_step")
    print(f" -> [Router] Routing to: {next_step}")

    if next_step == "chat_worker":
        return "chat_worker"
    elif next_step == "research_worker":
        # 리서치 작업은 먼저 필드 선택기(의도 파악)로 시작합니다.
        return "field_selector"
    elif next_step == "audit_worker":
        # 감사 작업은 먼저 사실 관계 추출로 시작합니다.
        return "extract_facts"
    else:
        # 기본값
        return "chat_worker"


def route_generation(state: AgentState):
    """
    Retrieval 이후, 일반 생성(Generator)으로 갈지 SOP(Audit Logic)로 갈지 결정
    (Supervisor 모델에서는 이미 경로가 정해지지만, Hybrid Flow를 위해 유지)
    """
    category = state.get("category")
    if category == "judgment":
        return "extract_facts"
    else:
        return "generate_answer"


def should_retry(state: AgentState):
    feedback = state.get("feedback", "")
    count = state.get("reflection_count", 0)

    if count >= 3:
        print(" -> [Stop] Max retries reached.")
        return END

    if feedback.startswith("FAIL:"):
        print(" -> [Retry] Regenerating answer.")
        return "generate_answer"
    else:
        return END


def route_disposition(state: AgentState):
    compliance = state.get("compliance_result", {})
    if isinstance(compliance, dict) and compliance.get("status") == "Violated":
        print(" -> [Route] Violation Detected. Starting Adversarial Simulation.")
        return "defense_agent"
    else:
        return END


# --- Graph Construction ---

workflow = StateGraph(AgentState)

# 1. 노드 정의 (Nodes)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("chat_worker", chat_worker)

# 리서치 워커 체인 (고급 RAG)
workflow.add_node("field_selector", field_selector)
# workflow.add_node("decompose_query", decompose_query) # 단순화를 위해 생략
workflow.add_node(
    "retrieve_documents", retrieve_documents
)  # state['search_query'] 사용 필수
workflow.add_node(
    "retrieve_graph_context", retrieve_graph_context
)  # 하이브리드 그래프 검색
workflow.add_node("validator", validator)  # 문서 검증기 (구 grade_documents 대체)
workflow.add_node(
    "strategy_decider", strategy_decider
)  # 전략 결정기 (구 rewrite_query 대체)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("reflect_answer", reflect_answer)

# 감사 워커 체인 (SOP: 표준감사절차)
workflow.add_node("extract_facts", extract_facts)
workflow.add_node("match_regulations", match_regulations)
workflow.add_node("evaluate_compliance", evaluate_compliance)
workflow.add_node("determine_disposition", determine_disposition)

# 적대적 감사 체인 (Adversarial)
workflow.add_node("defense_agent", defense_agent)
workflow.add_node("prosecution_agent", prosecution_agent)
workflow.add_node("judge_verdict", judge_verdict)


# 2. 엣지 연결 (Edges)

# 시작 -> 슈퍼바이저
workflow.set_entry_point("supervisor")

# 슈퍼바이저 -> 각 워커별 분기
workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "chat_worker": "chat_worker",
        # "decompose_query": "decompose_query", # 고급 흐름에서는 생략
        "field_selector": "field_selector",
        "extract_facts": "extract_facts",
    },
)

# Chat Worker -> End
workflow.add_edge("chat_worker", END)

# 리서치 워커 흐름 (적응형 루프)
# 필드 선택 -> 검색 -> 검증
workflow.add_edge("field_selector", "retrieve_documents")
workflow.add_edge("retrieve_documents", "validator")


def route_retrieval(state: AgentState):
    """
    검색 결과 검증 후, 생성(Generation)으로 갈지 전략 결정(Strategy Decider)으로 갈지 판단합니다.
    """
    is_valid = state.get("is_valid", "no")
    count = state.get("retrieval_count", 0)
    print(f"DEBUG: validity={is_valid}, retry_count={count}")

    if is_valid == "yes" or count >= 3:
        if count >= 3:
            print(" -> [Stop] Max retrieval retries reached. Proceeding anyway.")
        return "retrieve_graph_context"
    else:
        print(" -> [Retry] Strategy Decision Required.")
        return "strategy_decider"


workflow.add_conditional_edges(
    "validator",
    route_retrieval,
    {
        "retrieve_graph_context": "retrieve_graph_context",
        "strategy_decider": "strategy_decider",
    },
)


def route_strategy(state: AgentState):
    decision = state.get("analysis_decision", "rewrite_query")
    print(f"DEBUG: decision={decision}")

    if decision == "rewrite_query":
        # Go back to retrieve with new query (skipping field selector usually, but field selector resets query?)
        # Field Selector takes state['search_query'] if exists.
        # But maybe we should go back to Field Selector to re-eval fields?
        # Team NB says: "rewrite_query": "field_selector"
        return "field_selector"

    elif decision == "update_fields":
        # Keep query, change fields.
        # Ensure retriever uses these fields (Hybrid Retriever needs update to use state['selected_fields'])
        return "retrieve_documents"

    return "retrieve_documents"  # Fallback


workflow.add_conditional_edges(
    "strategy_decider",
    route_strategy,
    {"field_selector": "field_selector", "retrieve_documents": "retrieve_documents"},
)

workflow.add_conditional_edges(
    "retrieve_graph_context",
    route_generation,
    {"extract_facts": "extract_facts", "generate_answer": "generate_answer"},
)
# workflow.add_edge("retrieve_graph_context", "generate_answer") # Removed
workflow.add_edge("generate_answer", "reflect_answer")
workflow.add_conditional_edges(
    "reflect_answer", should_retry, {"generate_answer": "generate_answer", END: END}
)

# Audit Worker Flow (SOP)
workflow.add_edge("extract_facts", "match_regulations")
workflow.add_edge("match_regulations", "evaluate_compliance")
workflow.add_edge("evaluate_compliance", "determine_disposition")

# Audit -> Adversarial (Conditional)
workflow.add_conditional_edges(
    "determine_disposition",
    route_disposition,
    {"defense_agent": "defense_agent", END: END},
)

# Adversarial Flow
workflow.add_edge("defense_agent", "prosecution_agent")
workflow.add_edge("prosecution_agent", "judge_verdict")
workflow.add_edge("judge_verdict", END)

# 3. Checkpointer (Memory Persistence)
from langgraph.checkpoint.memory import MemorySaver

try:
    from redis import Redis
    from common.memory.redis_checkpointer import RedisSaver

    # Try connecting to Redis (assume localhost:6379 for now)
    # Ideally use Config.REDIS_URL
    redis_client = Redis(host="localhost", port=6379, db=0)
    # Check connection
    redis_client.ping()
    checkpointer = RedisSaver(redis_client)
    print("✅ Redis Memory Enabled")
except Exception as e:
    print(f"⚠️ Redis unavailable ({e}). Using In-Memory Saver (State is ephemeral).")
    checkpointer = MemorySaver()

# Compile
app = workflow.compile(checkpointer=checkpointer)
