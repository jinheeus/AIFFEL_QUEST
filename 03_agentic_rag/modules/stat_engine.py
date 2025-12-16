from state import AgentState
from .shared import rag_pipeline
from .stat_engine_v2 import StatEngineV2

# Instantiate V2 Engine (Singleton-like)
stat_engine = StatEngineV2(milvus_client=rag_pipeline.milvus_client)


def analyze_stats(state: AgentState) -> AgentState:
    """
    [Node] 'stats' 카테고리일 때 실행.
    StatEngineV2 (Code Interpreter)를 사용하여 복잡한 통계 분석을 수행합니다.
    """
    print(f"\n[Node] analyze_stats: 통계 분석 수행 중... (V2 Code Interpreter)")

    query = state.get("query", "")

    # Analyze using Code Interpreter
    result = stat_engine.analyze(query)

    # Format Answer
    stats_summary = f"[통계 분석 결과 (Code Interpreter)]\n{result}"

    print(f" -> 분석 완료: {result[:100].replace(chr(10), ' ')}...")  # Preview result

    state["documents"] = [stats_summary]

    return state
