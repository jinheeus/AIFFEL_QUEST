from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from state import AgentState
from common.config import Config
from common.model_factory import ModelFactory
from common.logger_config import setup_logger

# --- 모듈형 RAG 컴포넌트 임포트 ---
from modules.generator import generate_answer
from modules.retriever import retrieve_documents  # Wraps HybridRetriever
from modules.grader import grade_documents, grade_hallucination, grade_answer
from modules.rewriter import rewrite_query
from modules.field_selector import field_selector
from modules.sop_retriever import sop_retriever
from modules.memory import summarize_conversation  # New Memory Node
from modules.sql_retriever import SQLRetriever

# Fallback / Simple Chat
from modules.chat_worker import chat_worker

logger = setup_logger("GRAPH")


class RouterOutput(BaseModel):
    category: str = Field(
        description="Category: 'chat' (일상 대화), 'fast' (단순 검색/조회), 'deep' (복잡한 추론/분석)."
    )


def node_router(state: AgentState):
    """
    [Node] LLM 기반 라우터 (Router)
    사용자의 의도를 'chat', 'fast' (Fast Track), 'deep' (Deep RAG) 중 하나로 분류합니다.
    또한, 새로운 주제(Context)인지 판단하여 이전 맥락을 유지하거나 초기화합니다.
    """
    logger.info("--- [Router] Routing ---")
    query = state.get("query", "")

    # [컨텍스트 유지 로직]
    # 상태를 초기화하기 전에 이전 턴의 문서들을 저장
    prev_docs = state.get("documents", [])

    # 디버그: 이전 턴에서 받은 문서 개수 확인
    logger.debug(f" -> [Router Debug] Prev Docs Count: {len(prev_docs)}")

    persist_docs = state.get("persist_documents", [])
    if prev_docs and "검색 결과가 없습니다" not in str(prev_docs[0]):
        persist_docs = prev_docs
        logger.debug(f" -> [Router Debug] Persisting {len(persist_docs)} documents.")
    else:
        logger.debug(
            f" -> [Router Debug] Keeping existing {len(persist_docs)} persisted docs."
        )

    # 0. 상태 초기화 (Clean Slate): 같은 스레드 내의 이전 턴 임시 데이터를 삭제
    # 이전 'search_query'나 'documents'가 새로운 요청에 간섭하는 것을 방지함
    state["search_query"] = ""  # 검색 쿼리 초기화
    state["sub_queries"] = []
    state["documents"] = []
    state["graph_context"] = []
    state["sop_context"] = ""
    state["grade_status"] = ""
    state["retrieval_count"] = 0
    state["reflection_count"] = 0
    state["is_hallucinated"] = "no"
    state["is_useful"] = "yes"  # 기본값
    state["feedback"] = ""  # 피드백 루프 초기화

    # 1. Quick Keyword Check (Optimization)
    greetings = ["안녕", "반가워", "누구니", "hello", "hi", "하이", "ㅎㅇ"]
    if any(query.strip().startswith(x) for x in greetings) and len(query) < 10:
        logger.info(" -> [Router] Keyword Hit: Chat")
        return {"mode": "chat", "category": "chat"}

    # Intent Classification
    try:
        # Revert to HyperCLOVA X (User Requirement)
        llm = ModelFactory.get_rag_model(level="heavy", temperature=0)

        # [Fix] Abandon JSON Parser for Router because HCX is unstable with JSON.
        # Use String Result and parse it manually.

        system_prompt = """You are an intent classifier for an Audit RAG system.
        
        Classify the query into one of four categories:
        1. 'chat': Casual conversation, greetings, self-introduction, or non-audit questions.
        2. 'fast': Simple information retrieval, fact lookup, list requests.
           - Examples: "근무태만 사례 3개만 찾아줘" (Specific count/entity), "latest 2 cases", "Gas Corp 3 cases".
           - Use 'fast' when the user asks for specific entities or counts.
        3. 'deep': Complex analysis, comparison, cause-effect analysis, or searching for information *to be used in* a report.
           - Examples: "보고서 써야 하니까 횡령 관련 자료 다 가져와봐" (Research phase), "Compare these two cases", "Analyze the root cause".
        4. 'report': User explicitly asks to *WRITE*, *DRAFT*, or *GENERATE* the actual audit report document NOW.
           - Examples: "방금 찾은 2번 사례로 보고서 초안 작성해줘", "내용은 충분한 것 같아. 이제 보고서 뽑아줘", "Open report mode".
           - **Explicit Info Provision**: If the user provides specific details (Date, Amount, Content) and asks to write the report, OR just provides the facts to fill in the missing info, this is 'report'. 
             - e.g., "Here is the info: 2025, 200m won. Write the report." -> 'report'.
             - e.g., "Agency is KOGAS, Period is 2025." (Answering bot's question) -> 'report' | False (Keep context).
           - CRITICAL distinction: "I need to write a report, so find X" -> This is 'deep' (Search), NOT 'report'.

        [Pivot Detection]
        Determine if the user is pivoting to a completely NEW topic (New Entity, New Company, New Search) or asking a FOLLOW-UP question about the previous context.
        - "Gas Corp 3 cases" -> New Topic: TRUE
        - "Tell me more about the first one" -> New Topic: FALSE
        - "Summarize that" -> New Topic: FALSE
        - "Draft a report for this" -> New Topic: FALSE (Usually uses existing context)

        [Output Format]
        Return a single line with: Category | NewTopic(True/False)
        Example 1: fast | True
        Example 2: fast | False
        Example 3: chat | True
        Example 4: report | False
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                # Recent history helps context
                ("human", "Query: {query}"),
            ]
        )

        chain = prompt | llm | StrOutputParser()
        result_text = chain.invoke({"query": query})

        # Robust Parsing
        cleaned_text = result_text.strip().lower()

        # 1. Try splitting by pipe |
        if "|" in cleaned_text:
            parts = cleaned_text.split("|")
            category_part = parts[0].strip()
            new_topic_part = parts[1].strip() if len(parts) > 1 else ""
        else:
            # 2. Try newline or just look for keywords
            category_part = cleaned_text
            new_topic_part = cleaned_text

        # Robust Parsing
        if "report" in category_part:
            category = "report"
        elif "fast" in category_part:
            category = "fast"
        elif "chat" in category_part:
            category = "chat"
        else:
            category = "deep"  # default

        # NewTopic 판단도 유사하게
        # Default to True, but check if user explicitly said false
        is_new_topic = True

        # Check parsing from pipe if available
        if "|" in cleaned_text:
            # Check the second part
            second_part = parts[1].strip().lower() if len(parts) > 1 else ""
            if "false" in second_part:
                is_new_topic = False
        else:
            # If no pipe, check the whole string for explicit 'false' near 'NewTopic' keyword if it existed,
            # but since we stripped it, just rely on content.
            pass

        # [Safety Net]
        # If category is 'report' but we have NO context (no persist_docs and short history),
        # fallback to 'deep' (search) to find info first.
        if category == "report":
            has_context = len(persist_docs) > 0 or len(state["messages"]) > 2
            if not has_context:
                logger.info(
                    " -> [Router] Report intent detected but NO Context. Fallback to 'deep' search."
                )
                category = "deep"  # Fallback to search
                is_new_topic = True  # Treat as new search

        # Determine New Topic (Default to True for safety)
        is_new_topic = True

        # Check specifically for "false" / "no" in the New Topic part or if explicitly labeled
        # Examples: "NewTopic: False", "False", "fast | false"
        if "false" in new_topic_part or "no" in new_topic_part:
            is_new_topic = False

        # Edge Case: If LLM outputs "Category: fast\nNewTopic: False"
        if "newtopic" in cleaned_text and "false" in cleaned_text:
            # Double check: make sure "false" is associated with newtopic
            # Simple heuristic: if "false" appears, it's likely NewTopic=False (since category is rarely 'false')
            is_new_topic = False

        if "fast" in category_part:
            category = "fast"
        elif "chat" in category_part:
            category = "chat"
        elif "report" in category_part:
            category = "report"
        else:
            category = "deep"  # default to deep if unsure

        if "chat" in category:
            mode = "chat"
        elif "fast" in category:
            mode = "fast"
        elif "report" in category:
            mode = "report"
        else:
            mode = "deep"

        logger.info(
            f" -> [Router] LLM Decided: {mode.upper()} | New Topic: {is_new_topic} (Raw: {result_text})"
        )

        # [Context Logic]
        # If New Topic -> Clear Persistence
        # If Follow-up -> Keep Persistence (if available)
        final_persist_docs = []
        if not is_new_topic:
            final_persist_docs = persist_docs
            if final_persist_docs:
                logger.info(
                    f" -> [Router] Persistence: KEEPING {len(final_persist_docs)} docs (Follow-up)"
                )
            else:
                logger.info(" -> [Router] Persistence: None available to keep.")
        else:
            final_persist_docs = []
            logger.info(" -> [Router] Persistence: CLEARED (New Topic)")

        # Explicitly return cleared state to ensure updates propagate in LangGraph
        return {
            "mode": mode,
            "category": category,
            # Clean Slate
            "search_query": "",
            "sub_queries": [],
            "documents": [],
            "persist_documents": final_persist_docs,  # Updated Logic
            "graph_context": [],
            "sop_context": "",
            "grade_status": "",
            "retrieval_count": 0,
            "reflection_count": 0,
            "is_hallucinated": "no",
            "is_useful": "yes",
            "feedback": "",
            "command": "",  # Init
        }

    except Exception as e:
        logger.error(f" -> [Router] Error ({e}), Defaulting to Deep RAG")
        return {
            "mode": "deep",
            "category": "deep",
            # Clean Slate on Error too
            "search_query": "",
            "sub_queries": [],
            "documents": [],
            "persist_documents": [],  # Safe default: Clear on error
            "graph_context": [],
            "sop_context": "",
            "grade_status": "",
            "retrieval_count": 0,
            "reflection_count": 0,
            "is_hallucinated": "no",
            "is_useful": "yes",
            "feedback": "",
        }


# --- 조건부 엣지 (Conditional Edges) 로직 ---


def route_start(state: AgentState):
    """
    모드(Mode)에 따라 Chat, Fast(SQL), Deep(Field Selector) 트랙으로 분기합니다.
    """
    mode = state.get("mode", "deep")
    if mode == "chat":
        return "chat_worker"
    elif mode == "fast":
        return "retrieve_sql"
    elif mode == "report":
        return "report_manager"
    else:
        # Deep RAG starts with Field Selector
        return "field_selector"


def route_post_retrieval(state: AgentState):
    """
    Fast Track: 채점(Grading) 및 SOP 검색을 건너뛰고 바로 답변 생성으로 이동.
    Deep Track: 문서 채점(Grading) 단계로 이동.
    """
    mode = state.get("mode", "deep")
    if mode == "fast":
        logger.info(" -> [Fast Track] Skipping Grading/SOP. Proceeding to Generation.")
        return "generate"
    else:
        return "grade_documents"


def route_post_generation(state: AgentState):
    """
    Fast Track: 검증(Verification) 단계를 건너뛰고 바로 요약(Summary)으로 이동.
    Deep Track: 답변 검증(Verification) 단계로 이동.
    """
    mode = state.get("mode", "deep")
    if mode == "fast":
        logger.info(" -> [Fast Track] Skipping Verification. Proceeding to Summary.")
        return "summarize_conversation"
    else:
        return "verify_answer"


def route_retrieval(state: AgentState):
    """
    CRAG 로직: 검색된 문서가 부적절한 경우 쿼리를 재작성(Reformulate)합니다.
    """
    is_success = state.get("grade_status", "no")
    retry_count = state.get("retrieval_count", 0)

    if is_success == "yes" or retry_count >= 3:
        if retry_count >= 3:
            logger.info(" -> [Stop] Max retries reached. Proceeding to SOP Retrieval.")
        return "sop_retriever"
    else:
        logger.info(" -> [Loop] Retrieval Bad. Rewriting Query.")
        return "rewrite_query"


def route_verification(state: AgentState):
    """
    Self-RAG 로직: 환각(Hallucination) 여부와 답변의 유용성(Utility)을 검증합니다.
    """

    # 무한 루프 방지
    reflection_count = state.get("reflection_count", 0)
    if reflection_count >= 3:
        logger.info(" -> [Stop] Max reflection/regeneration reached. Ending.")
        return END

    # 1. Check Hallucination
    hallucinated = state.get("is_hallucinated", "no")  # 'yes' means hallucinated (bad)

    if hallucinated == "yes":
        logger.info(" -> [Loop] Hallucination detected. Regenerating...")
        return "generate"  # Simple retry (could add instructions)

    # 2. Check Answer Utility
    useful = state.get("is_useful", "yes")

    if useful == "yes":
        logger.info(" -> [End] Answer is useful.")
        return END
    else:
        logger.info(" -> [Loop] Answer not useful. Rewriting Query.")
        return "rewrite_query"


# --- 노드 래퍼 (Wrapper) ---
# 기존 modules의 함수들을 LangGraph State에 맞게 연결해주는 어댑터 역할을 합니다.


def node_retrieve(state: AgentState):
    # 기존 retrieve_documents 로직을 호출하는 래퍼(Wrapper)입니다.
    # 'search_query'가 존재하면 이를 우선적으로 사용합니다.
    return retrieve_documents(state)


def node_grade_documents(state: AgentState):
    # Grader를 호출하여 문서를 평가합니다.
    q = state.get("search_query") or state["query"]
    docs = state.get("documents", [])
    result = grade_documents(
        q, docs
    )  # returns {documents: [], is_retrieval_success: 'yes/no'}

    return {
        "documents": result["documents"],
        "grade_status": result["is_retrieval_success"],
    }


def node_rewrite(state: AgentState):
    # Rewriter를 호출하여 쿼리를 재작성합니다.
    q = state.get("search_query") or state["query"]
    new_q = rewrite_query(q)
    return {
        "search_query": new_q,
        "retrieval_count": state.get("retrieval_count", 0) + 1,
    }


def node_generate(state: AgentState):
    # generate_answer 함수를 호출하는 래퍼입니다.
    return generate_answer(state)


def node_consistency_check(state: AgentState):
    # 환각(Hallucination) 검사
    ans = state.get("answer", "")
    docs = state.get("documents", [])

    # Groundedness(근거 기반 여부) 확인
    if not docs or docs == ["검색 결과가 없습니다."]:
        return {"is_hallucinated": "no", "is_useful": "no"}

    is_grounded = grade_hallucination(ans, docs)  # 'yes' (근거 있음) 또는 'no' (환각)

    # 상태 로직 반전: 근거가 없으면(NOT grounded) 환각(is_hallucinated='yes')으로 간주
    is_hallucinated = "no" if is_grounded == "yes" else "yes"

    # 답변의 유용성(Utility) 확인
    q = state.get("search_query") or state["query"]
    is_useful = grade_answer(q, ans)  # 'yes' or 'no'

    # 루프 방지를 위해 reflection_count 증가
    cur_count = state.get("reflection_count", 0)

    return {
        "is_hallucinated": is_hallucinated,
        "is_useful": is_useful,
        "reflection_count": cur_count + 1,
    }


def node_retrieve_sql(state: AgentState):
    """
    [Node] SQL 기반 메타데이터 검색 (Fast Track)
    사용자의 질문을 SQL로 변환하여 DB에서 직접 검색합니다.
    """
    logger.info("--- [Node] SQL Retrieve ---")
    query = state["query"]

    # Get Context (from previous turn)
    context = state.get("persist_documents", [])
    if not context:
        # Fallback to current documents if exist (rare for initial SQL node)
        context = state.get("documents", [])

    retriever = SQLRetriever()  # Re-init each time or move to global if costly

    # Pass context to retrieve context-aware SQL
    documents = retriever.retrieve(query, context=context)

    return {"documents": documents, "retrieval_count": 1}


def node_report_manager(state: AgentState):
    """
    [Node] 보고서 작성 관리자 (Report Manager)
    사용자의 보고서 작성 요청을 처리하고, 누락된 정보를 확인하거나 Frontend에 작성 모드 진입 명령을 보냅니다.
    """
    logger.info("--- [Node] Report Manager ---")
    from modules.drafting_agent import DraftingAgent

    agent = DraftingAgent()

    # 1. Readiness Check
    result = agent.analyze_requirements(state["messages"])
    status = result.get("status")

    if status == "ready":
        # Report Ready -> Trigger Frontend
        logger.info(" -> [Report Manager] Ready. Triggering Frontend.")
        return {
            "answer": "보고서 작성을 위한 정보가 충분합니다. 작성을 시작합니다...",
            "command": "open_report",  # Signal to Frontend
        }
    else:
        # Missing Info -> Ask User
        missing = result.get("missing_fields", [])
        logger.info(f" -> [Report Manager] Missing: {missing}")

        # Simple Logic: Formulate a question using LLM or Rule-based
        # For now, rule-based approach for speed/reliability
        missing_str = ", ".join(missing)
        return {
            "answer": f"보고서 작성을 위해 다음 정보가 더 필요합니다: {missing_str}. \n해당 정보를 말씀해 주시면 바로 초안을 작성해 드리겠습니다.",
            "command": "",
        }


# --- 그래프 구성 (Graph Construction) ---

workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("router", node_router)  # Real Router Node
workflow.add_node("chat_worker", chat_worker)
workflow.add_node("report_manager", node_report_manager)  # New Node
workflow.add_node("retrieve_sql", node_retrieve_sql)  # SQL Node
workflow.add_node("field_selector", field_selector)
workflow.add_node("hybrid_retriever", node_retrieve)
workflow.add_node("grade_documents", node_grade_documents)
workflow.add_node("sop_retriever", sop_retriever)
workflow.add_node("rewrite_query", node_rewrite)
workflow.add_node("generate", node_generate)
# Removed duplicate 'generate'
workflow.add_node("verify_answer", node_consistency_check)
workflow.add_node("summarize_conversation", summarize_conversation)

# Entry Point
workflow.set_entry_point("router")

# Edges for Router
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
workflow.add_edge("report_manager", END)  # Report Manager is terminal for this turn

# SQL 검색 체인 (검색 후 라우팅 연결)
workflow.add_conditional_edges(
    "retrieve_sql",
    route_post_retrieval,
    {"generate": "generate", "grade_documents": "grade_documents"},
)

# RAG Start
workflow.add_edge("field_selector", "hybrid_retriever")

# 검색 체인 (Adaptive)
workflow.add_conditional_edges(
    "hybrid_retriever",
    route_post_retrieval,
    {"generate": "generate", "grade_documents": "grade_documents"},
)

# CRAG 루프 (Grade -> SOP or Rewrite)
workflow.add_conditional_edges(
    "grade_documents",
    route_retrieval,
    {"sop_retriever": "sop_retriever", "rewrite_query": "rewrite_query"},
)

workflow.add_edge("rewrite_query", "hybrid_retriever")

# SOP -> Generate
workflow.add_edge("sop_retriever", "generate")

# 답변 생성 및 검증 분기 (Adaptive)
workflow.add_conditional_edges(
    "generate",
    route_post_generation,
    {
        "summarize_conversation": "summarize_conversation",
        "verify_answer": "verify_answer",
    },
)

# 검증 루프 (Verification Loop)
workflow.add_conditional_edges(
    "verify_answer",
    route_verification,
    {
        "generate": "generate",  # 답변 재생성 (환각 발생 시)
        "rewrite_query": "rewrite_query",  # 검색 재시도 (답변 유용성 부족 시)
        END: "summarize_conversation",  # 성공 -> 대화 요약 후 종료
    },
)

# 요약 -> 종료 (Summarizer -> END)
workflow.add_edge("summarize_conversation", END)

if Config.ENABLE_REDIS:
    try:
        from redis import Redis
        from common.memory.redis_checkpointer import RedisSaver

        redis_client = Redis(host="localhost", port=6379, db=0)
        redis_client.ping()
        checkpointer = RedisSaver(redis_client)
        logger.info("✅ Redis Memory Enabled")
    except Exception as e:
        logger.warning(f"⚠️ Redis unavailable ({e}). Fallback to In-Memory.")
        checkpointer = MemorySaver()
else:
    checkpointer = MemorySaver()

# Compile
app = workflow.compile(checkpointer=checkpointer)
