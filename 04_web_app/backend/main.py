import sys
import os
import json
import asyncio
from typing import AsyncGenerator

# Add parent directory to path to import 03_agentic_rag modules
current_dir = os.path.dirname(os.path.abspath(__file__))
# 04_web_app/backend -> 04_web_app -> project_root
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
rag_dir = os.path.join(project_root, "03_agentic_rag")
sys.path.append(project_root)  # For config.py
sys.path.append(rag_dir)  # For graph.py and modules

# Ensure Environment Variables are loaded (if needed)
from config import Config

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import app as rag_app  # The compiled LangGraph app

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
    from modules.vector_retriever import get_retriever

    get_retriever()
    print("🔹 [Startup] VectorRetriever Ready!")


class ChatRequest(BaseModel):
    query: str
    # persona field removed
    history: list = []  # (New) history input
    session_id: str = "default_session"  # New: for persistent memory


NODE_NAMES = {
    "supervisor": "감사 계획 수립 (Supervisor)",
    "chat_worker": "일상 대화 처리",
    "analyze_query": "질문 의도 분석",
    "decompose_query": "복합 질문 분해 및 계획 수립",
    "retrieve_documents": "규정 및 사례 검색",
    "grade_documents": "문서 적합성 평가 (Adaptive)",
    "rewrite_query": "질문 재작성 (Rewrite)",
    "retrieve_graph_context": "관련 규정 연결 관계 분석",
    "analyze_stats": "통계 데이터 집계 및 분석",
    "extract_facts": "핵심 사실관계 추출",
    "match_regulations": "적용 법령 검토",
    "evaluate_compliance": "위반 여부 판정",
    "determine_disposition": "처분 기준 검토",
    "defense_agent": "소명 논리 시뮬레이션",
    "prosecution_agent": "감사 취약점 점검",
    "judge_verdict": "최종 판단 도출",
    "generate": "답변 생성",
    "generate_answer": "답변 생성",
    "reflect_answer": "답변 정합성 검증",
}


async def event_generator(
    query: str, history: list, session_id: str
) -> AsyncGenerator[str, None]:
    """
    Yields Server-Sent Events (SSE) for the frontend.
    """
    inputs = {
        "query": query,
        # persona field removed
        # Do NOT initialize documents=[], or it wipes previous state before Router can save it!
        # "documents": [],
        "messages": history,  # Pass history to graph state
        "reflection_count": 0,
    }

    # Thread Config for Redis Memory
    config = {"configurable": {"thread_id": session_id}}

    # Initial Event
    yield f"data: {json.dumps({'type': 'status', 'content': '분석 시작...'})}\n\n"

    try:
        # Use astream to get async node updates
        # [Fix] Increase recursion limit for complex RAG flows
        config["recursion_limit"] = 50
        async for output in rag_app.astream(inputs, config=config):
            for key, value in output.items():
                print(f"[API Log] Node Completed: {key}")

                # 1. Send Status Update (Thought Process)
                status_msg = NODE_NAMES.get(key, f"{key} 단계 완료")
                yield f"data: {json.dumps({'type': 'status', 'node': key, 'content': status_msg})}\n\n"

                # 2. If Final Answer is ready
                # We need to capture the answer from any node that produces a final output.
                # - 'generate_answer' / 'reflect_answer' (Standard RAG)
                # - 'judge_verdict' (Adversarial Audit)
                # - 'determine_disposition' (SOP - No Violation case)
                final_answer_nodes = [
                    "chat_worker",
                    "generate",
                    "generate_answer",
                    "reflect_answer",
                    "judge_verdict",
                    "determine_disposition",
                ]

                if key in final_answer_nodes:
                    if "answer" in value and value["answer"]:
                        yield f"data: {json.dumps({'type': 'answer', 'content': value['answer']})}\n\n"

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


@app.get("/health")
def health_check():
    return {"status": "ok", "model": Config.LLM_MODEL}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
