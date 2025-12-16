import sys
import os

# Add root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from modules.stat_engine_v2 import StatEngineV2
from modules.shared import rag_pipeline


def test_stats_v2():
    print("Initializing StatEngineV2...")
    engine = StatEngineV2(milvus_client=rag_pipeline.milvus_client)

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
