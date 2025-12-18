from langgraph.graph import StateGraph, END
from state import AgentState
from config import Config
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# --- Import Modular RAG Components ---
# --- Import Modular RAG Components ---
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


# --- Router Logic ---
# --- Router Logic ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from model_factory import ModelFactory


class RouterOutput(BaseModel):
    category: str = Field(
        description="Category: 'chat' (casual/greeting), 'fast' (simple retrieval/lookup), 'deep' (complex reasoning/analysis)."
    )


def node_router(state: AgentState):
    """
    [Node] LLM-based Router.
    Classifies user intent into 'chat', 'fast' (Fast Track), or 'deep' (Deep RAG).
    Updates state["mode"].
    """
    print("--- [Router] Routing ---")
    query = state.get("query", "")

    # [Context Persistence]
    # Save documents from the previous turn BEFORE clearing state
    prev_docs = state.get("documents", [])

    # DEBUG: Check what we received from previous turn
    print(f" -> [Router Debug] Prev Docs Count: {len(prev_docs)}")

    persist_docs = state.get("persist_documents", [])
    if prev_docs and "검색 결과가 없습니다" not in str(prev_docs[0]):
        persist_docs = prev_docs
        print(f" -> [Router Debug] Persisting {len(persist_docs)} documents.")
    else:
        print(
            f" -> [Router Debug] Keeping existing {len(persist_docs)} persisted docs."
        )

    # 0. Clean Slate: Clear ephemeral state from previous turns in the same thread
    # This prevents stale 'search_query' or 'documents' from interfering with the new request.
    state["search_query"] = ""  # Reset search query
    state["sub_queries"] = []
    state["documents"] = []
    state["graph_context"] = []
    state["sop_context"] = ""
    state["grade_status"] = ""
    state["retrieval_count"] = 0
    state["reflection_count"] = 0
    state["is_hallucinated"] = "no"
    state["is_useful"] = "yes"  # Default
    state["feedback"] = ""  # Clear previous feedback loop

    # 1. Quick Keyword Check (Optimization)
    greetings = ["안녕", "반가워", "누구니", "hello", "hi", "하이", "ㅎㅇ"]
    if any(query.strip().startswith(x) for x in greetings) and len(query) < 10:
        print(" -> [Router] Keyword Hit: Chat")
        return {"mode": "chat", "category": "chat"}

    # Intent Classification
    try:
        # Revert to HyperCLOVA X (User Requirement)
        llm = ModelFactory.get_rag_model(level="heavy", temperature=0)

        # [Fix] Abandon JSON Parser for Router because HCX is unstable with JSON.
        # Use String Result and parse it manually.

        system_prompt = """You are an intent classifier for an Audit RAG system.
        
        Classify the query into one of three categories:
        1. 'chat': Casual conversation, greetings, self-introduction, or non-audit questions.
        2. 'fast': Simple information retrieval, fact lookup, list requests (e.g., 'latest 2 cases'), OR specific follow-up questions about the previous answer (e.g., 'tell me more about #2', 'give me the file for #1'). Requires NO complex reasoning/planning.
        3. 'deep': Complex analysis, comparison, cause-effect analysis, or drafting reports. Requires multi-step reasoning.
        
        [Output Format]
        Return ONLY the category name: "chat", "fast", or "deep".
        Do not add any explanation or punctuation.
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
        if "fast" in cleaned_text:
            category = "fast"
        elif "chat" in cleaned_text:
            category = "chat"
        else:
            category = "deep"  # default to deep if unsure

        if "chat" in category:
            mode = "chat"
        elif "fast" in category:
            mode = "fast"
        else:
            mode = "deep"

        print(f" -> [Router] LLM Decided: {mode.upper()} (Raw: {result_text})")

        # Explicitly return cleared state to ensure updates propagate in LangGraph
        return {
            "mode": mode,
            "category": category,
            # Clean Slate
            "search_query": "",
            "sub_queries": [],
            "documents": [],
            "persist_documents": persist_docs,  # Pass forward the history
            "graph_context": [],
            "sop_context": "",
            "grade_status": "",
            "retrieval_count": 0,
            "reflection_count": 0,
            "is_hallucinated": "no",
            "is_useful": "yes",
            "feedback": "",
        }

    except Exception as e:
        print(f" -> [Router] Error ({e}), Defaulting to Deep RAG")
        return {
            "mode": "deep",
            "category": "deep",
            # Clean Slate on Error too
            "search_query": "",
            "sub_queries": [],
            "documents": [],
            "persist_documents": state.get("persist_documents", []),  # Keep Persistence
            "graph_context": [],
            "sop_context": "",
            "grade_status": "",
            "retrieval_count": 0,
            "reflection_count": 0,
            "is_hallucinated": "no",
            "is_useful": "yes",
            "feedback": "",
        }


# --- Conditional Edges Logic ---


def route_start(state: AgentState):
    """
    Routes to Chat or RAG (Fast/Deep) based on 'mode'.
    """
    mode = state.get("mode", "deep")
    if mode == "chat":
        return "chat_worker"
    elif mode == "fast":
        return "retrieve_sql"
    else:
        # Deep RAG starts with Field Selector
        return "field_selector"


def route_post_retrieval(state: AgentState):
    """
    Fast Track: Skip grading/SOP.
    Deep Track: Proceed to grading.
    """
    mode = state.get("mode", "deep")
    if mode == "fast":
        print(" -> [Fast Track] Skipping Grading/SOP. Proceeding to Generation.")
        return "generate"
    else:
        return "grade_documents"


def route_post_generation(state: AgentState):
    """
    Fast Track: Skip Verification.
    Deep Track: Proceed to Verification.
    """
    mode = state.get("mode", "deep")
    if mode == "fast":
        print(" -> [Fast Track] Skipping Verification. Proceeding to Summary.")
        return "summarize_conversation"
    else:
        return "verify_answer"


def route_retrieval(state: AgentState):
    """
    CRAG Logic: If retrieval failed (irrelevant docs), rewrite query.
    """
    is_success = state.get("grade_status", "no")
    retry_count = state.get("retrieval_count", 0)

    if is_success == "yes" or retry_count >= 3:
        if retry_count >= 3:
            print(" -> [Stop] Max retries reached. Proceeding to SOP Retrieval.")
        return "sop_retriever"
    else:
        print(" -> [Loop] Retrieval Bad. Rewriting Query.")
        return "rewrite_query"


def route_verification(state: AgentState):
    """
    Self-RAG Logic: Check hallucination and answer quality.
    """

    # [Fix] Loop Prevention (Check FIRST)
    reflection_count = state.get("reflection_count", 0)
    if reflection_count >= 3:
        print(" -> [Stop] Max reflection/regeneration reached. Ending.")
        return END

    # 1. Check Hallucination
    hallucinated = state.get("is_hallucinated", "no")  # 'yes' means hallucinated (bad)

    if hallucinated == "yes":
        print(" -> [Loop] Hallucination detected. Regenerating...")
        return "generate"  # Simple retry (could add instructions)

    # 2. Check Answer Utility
    useful = state.get("is_useful", "yes")

    if useful == "yes":
        print(" -> [End] Answer is useful.")
        return END
    else:
        print(" -> [Loop] Answer not useful. Rewriting Query.")
        return "rewrite_query"


# --- Node Wrappers (Adopting new modules to State) ---


def node_retrieve(state: AgentState):
    # Wrapper to call existing retrieve_documents logic
    # Ensure it uses 'search_query' if available
    return retrieve_documents(state)


def node_grade_documents(state: AgentState):
    # Call grader
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
    # Call rewriter
    q = state.get("search_query") or state["query"]
    new_q = rewrite_query(q)
    return {
        "search_query": new_q,
        "retrieval_count": state.get("retrieval_count", 0) + 1,
    }


def node_generate(state: AgentState):
    # wrapper for generate_answer
    return generate_answer(state)


def node_consistency_check(state: AgentState):
    # Hallucination Check
    ans = state.get("answer", "")
    docs = state.get("documents", [])

    # Check groundedness
    if not docs or docs == ["검색 결과가 없습니다."]:
        return {"is_hallucinated": "no", "is_useful": "no"}

    is_grounded = grade_hallucination(
        ans, docs
    )  # 'yes' (grounded) or 'no' (hallucination)

    # Invert logic for state: is_hallucinated 'yes' if NOT grounded
    is_hallucinated = "no" if is_grounded == "yes" else "yes"

    # Check utility
    q = state.get("search_query") or state["query"]
    is_useful = grade_answer(q, ans)  # 'yes' or 'no'

    # Increment reflection count to track cycles
    cur_count = state.get("reflection_count", 0)

    return {
        "is_hallucinated": is_hallucinated,
        "is_useful": is_useful,
        "reflection_count": cur_count + 1,
    }

    return {
        "is_hallucinated": is_hallucinated,
        "is_useful": is_useful,
        "reflection_count": cur_count + 1,
    }


def node_retrieve_sql(state: AgentState):
    """
    [Node] SQL-based Retrieval for Metadata Queries (Fast Track).
    """
    print("--- [Node] SQL Retrieve ---")
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


# --- Graph Construction ---

workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("router", node_router)  # Real Router Node
workflow.add_node("chat_worker", chat_worker)
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
        "field_selector": "field_selector",
    },
)

workflow.add_edge("chat_worker", END)

# SQL Retrieval Chain (Connect to Post-Retrieval Routing)
workflow.add_conditional_edges(
    "retrieve_sql",
    route_post_retrieval,
    {"generate": "generate", "grade_documents": "grade_documents"},
)

# RAG Start
workflow.add_edge("field_selector", "hybrid_retriever")

# Retrieval Chain (Adaptive)
workflow.add_conditional_edges(
    "hybrid_retriever",
    route_post_retrieval,
    {"generate": "generate", "grade_documents": "grade_documents"},
)

# CRAG Loop (Grade -> SOP or Rewrite)
workflow.add_conditional_edges(
    "grade_documents",
    route_retrieval,
    {"sop_retriever": "sop_retriever", "rewrite_query": "rewrite_query"},
)

workflow.add_edge("rewrite_query", "hybrid_retriever")

# SOP -> Generate
workflow.add_edge("sop_retriever", "generate")

# Generation & Verification (Adaptive)
workflow.add_conditional_edges(
    "generate",
    route_post_generation,
    {
        "summarize_conversation": "summarize_conversation",
        "verify_answer": "verify_answer",
    },
)

# Verification Loop
workflow.add_conditional_edges(
    "verify_answer",
    route_verification,
    {
        "generate": "generate",  # Retry generation (Hallucination)
        "rewrite_query": "rewrite_query",  # Retry retrieval (Not Useful)
        END: "summarize_conversation",  # Success -> Summarize before END
    },
)

# Summarizer -> END
workflow.add_edge("summarize_conversation", END)

# --- Checkpointer ---
from langgraph.checkpoint.memory import MemorySaver

if Config.ENABLE_REDIS:
    try:
        from redis import Redis
        from common.memory.redis_checkpointer import RedisSaver

        redis_client = Redis(host="localhost", port=6379, db=0)
        redis_client.ping()
        checkpointer = RedisSaver(redis_client)
        print("✅ Redis Memory Enabled")
    except Exception as e:
        print(f"⚠️ Redis unavailable ({e}). Fallback to In-Memory.")
        checkpointer = MemorySaver()
else:
    checkpointer = MemorySaver()

# Compile
app = workflow.compile(checkpointer=checkpointer)
