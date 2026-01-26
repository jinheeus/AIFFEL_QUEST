from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# Conditional Imports for Redis
try:
    from redis import Redis
    from common.memory.redis_checkpointer import RedisSaver
except ImportError:
    Redis = None
    RedisSaver = None

from state import AgentState
from common.config import Config
from common.model_factory import ModelFactory
from common.logger_config import setup_logger

# --- 모듈형 RAG 컴포넌트 임포트 (Modular RAG Components) ---
from modules.generator import generate_answer
from modules.retriever import retrieve_documents
from modules.grader import grade_documents, grade_hallucination, grade_answer
from modules.rewriter import rewrite_query
from modules.field_selector import field_selector
from modules.sop_retriever import sop_retriever
from modules.memory import summarize_conversation
from modules.sql_retriever import SQLRetriever
from modules.drafting_agent import DraftingAgent

# Fallback / Simple Chat
from modules.chat_worker import chat_worker

logger = setup_logger("GRAPH")


class RouterOutput(BaseModel):
    category: str = Field(
        description="Category: 'chat' (일상 대화), 'fast' (단순 검색/조회), 'deep' (복잡한 추론/분석)."
    )


def node_router(state: AgentState):
    """
    [Node] Router
    사용자의 의도를 'chat', 'fast', 'deep', 'report' 중 하나로 분류합니다.
    새로운 주제(Context Pivot) 여부를 판단하여 이전 맥락을 관리합니다.
    """
    logger.info("--- [Router] Routing ---")
    query = state.get("query", "")

    # [컨텍스트 관리]
    # 이전 턴의 문서는 일단 유지하지만, Router 판단에 따라 persist 여부 결정
    prev_docs = state.get("documents", [])
    persist_docs = state.get("persist_documents", [])

    if prev_docs and "검색 결과가 없습니다" not in str(prev_docs[0]):
        # 이전 턴에 유효한 검색 결과가 있었다면 후보로 등록
        persist_docs = prev_docs

    # 상태 초기화 (Clean Slate)
    state["search_query"] = ""
    state["sub_queries"] = []
    state["documents"] = []
    state["graph_context"] = []
    state["sop_context"] = ""
    state["grade_status"] = ""
    state["retrieval_count"] = 0
    state["reflection_count"] = 0
    state["is_hallucinated"] = "no"
    state["is_useful"] = "yes"
    state["feedback"] = ""

    # 1. 빠른 키워드 체크 (Optimization)
    greetings = ["안녕", "반가워", "누구니", "hello", "hi", "하이", "ㅎㅇ"]
    if any(query.strip().startswith(x) for x in greetings) and len(query) < 10:
        logger.info(" -> [Router] Keyword Hit: Chat")
        return {"mode": "chat", "category": "chat"}

    # 2. 의도 분류 (Intent Classification)
    try:
        # [HyperCLOVA X] Heavy 모델로 의도 분석 (Reasoning Optimized)
        llm = ModelFactory.get_rag_model(level="heavy", temperature=0)

        # [Note] HCX 안정성을 위해 문자열 출력 파싱 방식 사용
        system_prompt = """당신은 감사 RAG 시스템의 의도 분류기(Intent Classifier)입니다.

        사용자의 질문을 다음 4가지 카테고리 중 하나로 분류하십시오:
        1. 'chat': 일상 대화, 인사, 자기소개 또는 감사와 무관한 질문.
        2. 'fast': **특정 감사 사례 데이터 검색**, 통계, 리스트 조회.
           - 예시: "근무태만 사례 3개만 찾아줘", "가스공사 사례는?", "2024년 징계 건수".
           - 사용자가 **"구체적인 개체(회사, 연도, 비위행위)"를 언급하며 사례 리스트를 요구**할 때 사용합니다.
           - **주의**: "감사 절차", "규정", "방법"을 묻는 질문은 'fast'가 아닙니다 ('deep'입니다).
        3. 'deep': **절차, 규정, 방법 설명**, 복잡한 분석, 보고서 작성을 위한 정보 탐색.
           - 예시: "**일반적인 감사 진행 절차가 어떻게 돼?**", "횡령 시 처리 규정은?", "이 두 사례 비교해줘".
           - 사용자가 **지식, 절차, 규정, 방법론**에 대해 물으면 무조건 'deep'입니다.
           - "보고서를 써야 하니까 자료 찾아줘"는 'report'가 아니라 'deep'입니다.
        4. 'report': 사용자가 지금 즉시 실제 감사 보고서를 **작성**, **초안 생성**하라고 명시적으로 요청함.
           - 예시: "방금 찾은 사례로 보고서 써줘", "이대로 보고서 작성해".

        [주제 전환 판단 (Pivot Detection)]
        사용자가 완전히 새로운 주제(새로운 개체, 새로운 회사)로 전환하는지, 아니면 이전 맥락에 대한 후속 질문인지 판단하십시오.
        - "가스공사 사례 3개" -> 새로운 주제: True
        - "첫 번째 사례에 대해 더 말해줘" -> 새로운 주제: False
        - "감사 절차는?" (지식 질문) -> 새로운 주제: True (False여도 상관없으나 보통 독립적 질문)

        [출력 형식]
        다음 형식의 한 줄로 반환하십시오: Category | NewTopic(True/False)
        예시 1: fast | True
        예시 2: deep | False
        예시 3: report | False
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Query: {query}"),
            ]
        )

        chain = prompt | llm | StrOutputParser()
        result_text = chain.invoke({"query": query})

        # 결과 파싱
        cleaned_text = result_text.strip().lower()

        # 안전한 파싱 (Pipe 구분 또는 줄바꿈 처리)
        parts = cleaned_text.split("|")
        category_part = parts[0].strip()
        new_topic_part = parts[1].strip() if len(parts) > 1 else ""

        # Category 결정
        if "report" in category_part:
            category = "report"
        elif "fast" in category_part:
            category = "fast"
        elif "chat" in category_part:
            category = "chat"
        else:
            category = "deep"

        # NewTopic 결정
        is_new_topic = True
        if "false" in new_topic_part or "no" in new_topic_part:
            is_new_topic = False

        # [Safety Net] Context 없이 Report 요청 시 Deep으로 전환
        if category == "report":
            has_context = len(persist_docs) > 0 or len(state["messages"]) > 2
            if not has_context:
                logger.info(
                    " -> [Router] Report requested without context. Fallback to 'deep'."
                )
                category = "deep"
                is_new_topic = True

        mode = category  # category 이름이 곧 mode 키

        logger.info(
            f" -> [Router] Decision: {mode.upper()} | New Topic: {is_new_topic}"
        )

        # [컨텍스트 관리]
        final_persist_docs = []
        if not is_new_topic:
            final_persist_docs = persist_docs
            if final_persist_docs:
                logger.info(
                    f" -> [Router] Persistence: KEEPING {len(final_persist_docs)} docs (Follow-up)"
                )
        else:
            logger.info(" -> [Router] Persistence: CLEARED (New Topic)")

        return {
            "mode": mode,
            "category": category,
            "search_query": "",
            "persist_documents": final_persist_docs,
            "documents": [],
            "command": "",
        }

    except Exception as e:
        logger.error(f" -> [Router] Error ({e}), Defaulting to Deep RAG")
        return {
            "mode": "deep",
            "category": "deep",
            "persist_documents": [],
        }


# --- 조건부 엣지 (Conditional Edges) ---


def route_start(state: AgentState):
    """모드에 따라 Chat, Fast(SQL), Deep(Field Selector) 트랙 분기."""
    mode = state.get("mode", "deep")
    if mode == "chat":
        return "chat_worker"
    elif mode == "fast":
        return "retrieve_sql"
    elif mode == "report":
        return "report_manager"
    else:
        return "field_selector"  # Deep RAG start


def route_post_retrieval(state: AgentState):
    """Fast Track은 채점/SOP 건너뛰기, Deep Track은 채점 단계로 이동."""
    mode = state.get("mode", "deep")
    if mode == "fast":
        logger.info(" -> [Fast Track] Skipping Grading/SOP.")
        return "generate"
    else:
        return "grade_documents"


def route_post_generation(state: AgentState):
    """Fast Track은 검증 건너뛰기, Deep Track은 검증 단계로 이동."""
    mode = state.get("mode", "deep")
    if mode == "fast":
        return "summarize_conversation"
    else:
        return "verify_answer"


def route_retrieval(state: AgentState):
    """CRAG 로직: 문서 품질이 낮으면 쿼리 재작성, 높으면 SOP 검색 진행."""
    is_success = state.get("grade_status", "no")
    retry_count = state.get("retrieval_count", 0)

    # [Fix] Bypass SOP Retriever. Go directly to Generator for better list handling.
    # SOP Retriever aggregates everything into one fact, which is bad for "List N cases".
    if is_success == "yes" or retry_count >= 1:
        if retry_count >= 1:
            logger.info(
                " -> [Stop] Max retries reached. Proceeding to Answer Generation."
            )
        return "generate"
    else:
        logger.info(" -> [Loop] Retrieval Bad. Rewriting Query.")
        return "rewrite_query"


def route_verification(state: AgentState):
    """Self-RAG 로직: 환각 및 유용성 검증 후 재시도 여부 결정."""
    reflection_count = state.get("reflection_count", 0)
    # [Optimization] Max Retries Reduced to 1 for Speed
    if reflection_count >= 1:
        logger.info(" -> [Stop] Max reflection reached.")
        return END

    # 1. 환각 체크
    hallucinated = state.get("is_hallucinated", "no")
    if hallucinated == "yes":
        logger.info(" -> [Loop] Hallucination detected. Regenerating...")
        return "generate"

    # 2. 유용성 체크
    useful = state.get("is_useful", "yes")
    if useful == "yes":
        return END
    else:
        logger.info(" -> [Loop] Answer not useful. Rewriting Query.")
        return "rewrite_query"


# --- 노드 래퍼 (Wrapper) ---


def node_retrieve(state: AgentState):
    """Hybrid Retriever 래퍼."""
    return retrieve_documents(state)


def node_grade_documents(state: AgentState):
    """문서 평가 노드."""
    q = state.get("search_query") or state["query"]
    docs = state.get("documents", [])
    result = grade_documents(q, docs)
    return {
        "documents": result["documents"],
        "grade_status": result["is_retrieval_success"],
    }


def node_rewrite(state: AgentState):
    """쿼리 재작성 노드."""
    q = state.get("search_query") or state["query"]
    new_q = rewrite_query(q)
    return {
        "search_query": new_q,
        "retrieval_count": state.get("retrieval_count", 0) + 1,
    }


def node_generate(state: AgentState):
    """답변 생성 노드."""
    return generate_answer(state)


def node_consistency_check(state: AgentState):
    """환각(Hallucination) 및 유용성(Utility) 검증 노드."""
    ans = state.get("answer", "")
    docs = state.get("documents", [])

    # 문서가 없으면 환각 아님(Grounding 불가), 유용성은 낮음으로 처리
    if not docs or docs == ["검색 결과가 없습니다."]:
        return {"is_hallucinated": "no", "is_useful": "no"}

    is_grounded = grade_hallucination(ans, docs)  # 'yes' or 'no'
    is_hallucinated = "no" if is_grounded == "yes" else "yes"

    q = state.get("search_query") or state["query"]
    is_useful = grade_answer(q, ans)

    return {
        "is_hallucinated": is_hallucinated,
        "is_useful": is_useful,
        "reflection_count": state.get("reflection_count", 0) + 1,
    }


def node_retrieve_sql(state: AgentState):
    """[Node] SQL 검색 (Fast Track)"""
    logger.info("--- [Node] SQL Retrieve ---")
    query = state["query"]
    context = state.get("persist_documents", []) or state.get("documents", [])

    retriever = SQLRetriever()
    documents = retriever.retrieve(query, context=context)
    return {"documents": documents, "retrieval_count": 1}


def node_report_manager(state: AgentState):
    """[Node] Report Manager (Drafting Agent)"""
    logger.info("--- [Node] Report Manager ---")
    agent = DraftingAgent()
    result = agent.analyze_requirements(state["messages"])

    if result.get("status") == "ready":
        logger.info(" -> [Report Manager] Ready. Triggering Frontend.")
        return {
            "answer": "보고서 작성을 위한 정보가 충분합니다. 작성을 시작합니다...",
            "command": "open_report",
        }
    else:
        missing = result.get("missing_fields", [])
        logger.info(f" -> [Report Manager] Missing: {missing}")
        missing_str = ", ".join(missing)
        return {
            "answer": f"보고서 작성을 위해 다음 정보가 더 필요합니다: {missing_str}. \n해당 정보를 말씀해 주시면 바로 초안을 작성해 드리겠습니다.",
            "command": "",
        }


# --- 그래프 구성 (Graph Construction) ---

workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("router", node_router)
workflow.add_node("chat_worker", chat_worker)
workflow.add_node("report_manager", node_report_manager)
workflow.add_node("retrieve_sql", node_retrieve_sql)
workflow.add_node("field_selector", field_selector)
workflow.add_node("hybrid_retriever", node_retrieve)
workflow.add_node("grade_documents", node_grade_documents)
workflow.add_node("sop_retriever", sop_retriever)
workflow.add_node("rewrite_query", node_rewrite)
workflow.add_node("generate", node_generate)
workflow.add_node("verify_answer", node_consistency_check)
workflow.add_node("summarize_conversation", summarize_conversation)

# Edges
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_start,
    {
        "chat_worker": "chat_worker",
        "retrieve_sql": "retrieve_sql",
        "report_manager": "report_manager",
        "field_selector": "field_selector",
    },
)

workflow.add_edge("chat_worker", END)
workflow.add_edge("report_manager", END)

# SQL Track
workflow.add_conditional_edges(
    "retrieve_sql",
    route_post_retrieval,
    {"generate": "generate", "grade_documents": "grade_documents"},
)

# Deep RAG Track
workflow.add_edge("field_selector", "hybrid_retriever")
workflow.add_conditional_edges(
    "hybrid_retriever",
    route_post_retrieval,
    {"generate": "generate", "grade_documents": "grade_documents"},
)
workflow.add_conditional_edges(
    "grade_documents",
    route_retrieval,
    {"sop_retriever": "sop_retriever", "rewrite_query": "rewrite_query"},
)
workflow.add_edge("rewrite_query", "hybrid_retriever")
workflow.add_edge("sop_retriever", "generate")

# Generation & Verification
workflow.add_conditional_edges(
    "generate",
    route_post_generation,
    {
        "summarize_conversation": "summarize_conversation",
        "verify_answer": "verify_answer",
    },
)
workflow.add_conditional_edges(
    "verify_answer",
    route_verification,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
        END: "summarize_conversation",
    },
)

workflow.add_edge("summarize_conversation", END)

# Checkpointer
if Config.ENABLE_REDIS and Redis is not None:
    try:
        redis_client = Redis(host="localhost", port=6379, db=0)
        redis_client.ping()
        checkpointer = RedisSaver(redis_client)
        logger.info("✅ Redis Memory Enabled")
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed ({e}). Fallback to In-Memory.")
        checkpointer = MemorySaver()
else:
    logger.info("ℹ️ Redis disabled or library missing. Using In-Memory.")
    checkpointer = MemorySaver()

# Compile
app = workflow.compile(checkpointer=checkpointer)
