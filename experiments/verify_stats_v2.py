import sys
import os

# Add root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from modules.stat_engine_v2 import StatEngineV2
from modules.vector_retriever import get_retriever


def test_stats_v2():
    print("Initializing StatEngineV2...")
    rag_pipeline = get_retriever()
    # Assuming vector_store has .client or we need to access it.
    # Langchain Milvus store usually has 'milvus_client' or we can init StatEngine differently.
    # checking StatEngineV2 usage: it takes milvus_client.
    # Let's assume rag_pipeline.vector_store.client works or I might need to add a property.
    engine = StatEngineV2(milvus_client=rag_pipeline.vector_store.client)

    # Test Query 1: Simple Aggregation
    query = "징계 처분(파면, 해임)을 받은 건수는 총 몇 건이야?"
    print(f"\n[Test Query 1] {query}")
    answer = engine.analyze(query)
    print(f"[Answer]\n{answer}")

    # Test Query 2: Trend Analysis (Code Interpreter's strength)
    query = "연도별 적발 건수의 추이를 알려줘."
    print(f"\n[Test Query 2] {query}")
    answer = engine.analyze(query)
    print(f"[Answer]\n{answer}")


if __name__ == "__main__":
    test_stats_v2()
