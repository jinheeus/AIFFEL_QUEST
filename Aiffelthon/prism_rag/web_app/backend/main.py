import sys
import os
import json
import asyncio
from typing import AsyncGenerator

# Add parent directory to path to import agentic_rag_v2 modules
# web_app/backend -> web_app -> project_root
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)  # Load project root first to import common.config

# Import Config to get the active RAG version
from common.config import Config

rag_dir = os.path.join(project_root, "rag", Config.ACTIVE_RAG_DIR)

# [Fix] Windows CP949 Encoding Issue
# Reconfigure stdout to use utf-8 so emojis don't crash the app
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print(f"🔹 [Backend] Active RAG Module: {rag_dir}")
sys.path.append(rag_dir)  # Add dynamic RAG module to path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import app as rag_app  # The compiled LangGraph app

# DraftingAgent depends on agentic_rag_v2 modules.
# If running v1, this might not exist.
try:
    from modules.drafting_agent import DraftingAgent
except ImportError:
    print(
        "⚠️ DraftingAgent not found (Likely running V1). Report generation will be disabled."
    )
    DraftingAgent = None

app = FastAPI(title="Agentic RAG API")

# Allow CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    print("🔹 [Startup] Warming up VectorRetriever (Loading BM25 Index)...")
    # Initialize singleton to trigger BM25 build/load
    try:
        from modules.vector_retriever import get_retriever

        get_retriever()
        print("🔹 [Startup] VectorRetriever Ready!")
    except ImportError:
        print(
            "⚠️ modules.vector_retriever not found (Likely running V1). Skipping warmup."
        )


class ChatRequest(BaseModel):
    query: str
    history: list = []
    session_id: str = "default_session"
    additional_info: dict = {}
    dashboard_context: dict = {}  # 대시보드 필터 현황


NODE_NAMES = {
    # V2 Nodes (Updated)
    "router": "질문 의도 분석 및 라우팅",
    "chat_worker": "일상 대화 처리",
    "retrieve_sql": "SQL 기반 메타데이터 검색",
    "report_manager": "보고서 작성 모드",
    "hybrid_retriever": "규정 및 사례 검색 (Hybrid)",
    "grade_documents": "문서 적합성 평가",
    "sop_retriever": "업무 편람(SOP) 검색",
    "rewrite_query": "검색 쿼리 재작성",
    "generate": "답변 생성",
    "verify_answer": "답변 정합성 검증",
    "summarize_conversation": "대화 요약",
    # V1 Nodes
    "date_extract": "날짜 정보 추출 및 정규화",
    "field_select": "검색 필드 선택",
    "field_selector": "검색 필드 선택",
    "retrieve": "관련 문서 검색",
    "rerank": "검색 결과 재순위화 (Rerank)",
    "validate": "검색 결과 검증",
    "rewrite": "검색 쿼리 재작성",
}


async def event_generator(
    query: str, history: list, session_id: str
) -> AsyncGenerator[str, None]:
    """
    Yields Server-Sent Events (SSE) for the frontend.
    """
    inputs = {
        # Do NOT initialize documents=[], or it wipes previous state before Router can save it!
        "messages": history,  # Pass history to graph state
        "reflection_count": 0,
    }

    # Input Key Mapping (V1 vs V2)
    # V2 uses "query", V1 uses "question"
    if "v1" in Config.ACTIVE_RAG_DIR:
        inputs["question"] = query
        inputs["original_question"] = query  # V1 often needs this initialized
    else:
        inputs["query"] = query

    # Thread Config for Redis Memory
    config = {"configurable": {"thread_id": session_id}}

    # Initial Event
    yield f"data: {json.dumps({'type': 'status', 'content': '분석 시작...'})}\n\n"

    try:
        # Use astream to get async node updates
        # [Fix] Increase recursion limit for complex RAG flows
        config["recursion_limit"] = 50
        
        import time
        start_total = time.time()
        async for output in rag_app.astream(inputs, config=config):
            for key, value in output.items():
                print(f"[API Log] Node Completed: {key}")

                # 1. Send Status Update (Thought Process)
                status_msg = NODE_NAMES.get(key, f"{key} 단계 완료")
                yield f"data: {json.dumps({'type': 'status', 'node': key, 'content': status_msg})}\n\n"

                # 2. If Final Answer is ready
                safe_answer_nodes = [
                    "chat_worker",  # Simple chat, always final
                    "summarize_conversation",  # End of RAG flow
                    "report_manager",  # Report ready signal
                ]

                if key in safe_answer_nodes:
                    # We need to dig the answer from the state.
                    if "answer" in value and value["answer"]:
                        yield f"data: {json.dumps({'type': 'answer', 'content': value['answer']})}\n\n"
                        # 출처 문서 함께 전송
                        docs = value.get("persist_documents") or value.get("documents") or []
                        if docs:
                            refs = []
                            for doc in docs[:5]:
                                if isinstance(doc, dict):
                                    refs.append({
                                        "title": doc.get("summary_title") or doc.get("title", ""),
                                        "agency": doc.get("site") or doc.get("agency_category", ""),
                                        "date": str(doc.get("date", ""))[:10],
                                        "risk": doc.get("risk_category", ""),
                                        "disposition": doc.get("disposition_level", ""),
                                        "url": doc.get("download_url", ""),
                                    })
                            if refs:
                                yield f"data: {json.dumps({'type': 'references', 'content': refs})}\n\n"

                    if "command" in value and value["command"]:
                        yield f"data: {json.dumps({'type': 'command', 'content': value['command']})}\n\n"
        
        end_total = time.time()
        print("🔥 TOTAL RAG TIME:", end_total - start_total)
        
        yield "data: [DONE]\n\n"

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f" -> [API] Incoming Session ID: {request.session_id}")

    # [DEBUG] Session ID
    session_id = request.session_id or "default_session"
    print(f" -> [API] Session ID: {session_id}")

    return StreamingResponse(
        event_generator(request.query, request.history, session_id),
        media_type="text/event-stream",
    )


@app.post("/check_report_readiness")
async def check_report_readiness_endpoint(request: ChatRequest):
    """
    Checks if there is enough information to generate a report.
    """
    print(f"🔹 [API] Checking Readiness for Session: {request.session_id}")
    if not DraftingAgent:
        return {
            "status": "error",
            "message": "DraftingAgent not supported in this RAG version.",
        }

    agent = DraftingAgent()
    result = agent.analyze_requirements(request.history)
    return result


@app.post("/generate_report")
async def generate_report_endpoint(request: ChatRequest):
    """
    Generates a formal audit report based on conversation history.
    """
    print(f"🔹 [API] Generating Report for Session: {request.session_id}")
    print(f"   -> Additional Info: {request.additional_info}")

    # 1. Initialize Components
    if not DraftingAgent:
        return {
            "report": "오류: 현재 RAG 버전에서는 보고서 생성 기능을 지원하지 않습니다."
        }

    agent = DraftingAgent()
    from modules.vector_retriever import get_retriever

    retriever = get_retriever()

    # 2. Construct Search Query for Context (Source B)
    search_query = ""
    if request.additional_info:
        subjects = [
            request.additional_info.get("대상 기관", ""),
            request.additional_info.get("사건 제목", ""),
            request.additional_info.get("문제점", ""),
        ]
        search_query = " ".join([s for s in subjects if s]).strip()

    if not search_query and request.history:
        for msg in reversed(request.history):
            if msg["role"] == "user":
                search_query = msg["content"]
                break

    if not search_query:
        search_query = "감사 보고서 작성 일반 규정"

    print(f"   -> Retrieval Query: '{search_query}'")

    # 3. Retrieve Documents (Source B)
    try:
        retrieved_docs = retriever.search_and_merge(search_query, top_k=3)
        print(f"   -> Retrieved {len(retrieved_docs)} documents for context.")
    except Exception as e:
        print(f"   -> ⚠️ Retrieval Failed: {e}")
        retrieved_docs = []

    # 4. Generate Report
    report_content = agent.generate_report(
        messages=request.history,
        retrieved_docs=retrieved_docs,
        additional_info=request.additional_info,
        dashboard_context=request.dashboard_context,
    )

    return {"report": report_content}


@app.get("/health")
def health_check():
    return {"status": "ok", "model": Config.LLM_MODEL}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)