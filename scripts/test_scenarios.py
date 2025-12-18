import sys
import os
import uuid
from dotenv import load_dotenv

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rag_path = os.path.join(project_root, "03_agentic_rag")
sys.path.append(rag_path)
sys.path.append(project_root)

# Load env
load_dotenv(os.path.join(project_root, ".env"))

# Import Graph
try:
    from graph import app
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def run_turn(thread_id, query):
    print(f"\n🔹 User: {query}")
    print("-" * 50)

    inputs = {"query": query}
    config = {"configurable": {"thread_id": thread_id}}

    final_state = None
    # We use stream to see intermediate steps if needed, but for this summary we just need final answer
    # But app.invoke is cleaner for state inspection
    try:
        final_state = app.invoke(inputs, config=config)

        # Check components
        docs = final_state.get("documents", [])
        answer = final_state.get("answer", "")
        mode = final_state.get("mode", "unknown")

        print(f" -> Mode: {mode.upper()}")
        print(f" -> Retrieval Count: {len(docs)}")
        if docs:
            # Print first doc metadata for verification
            first_meta = docs[0].metadata
            print(
                f" -> First Doc: {first_meta.get('title')} ({first_meta.get('date')})"
            )

        print(f" -> Answer: {answer[:300]}...")  # Truncate for readability
        print("-" * 50)

    except Exception as e:
        print(f"❌ Error: {e}")


def run_scenario_1():
    """Scenario 1: Typo Handling + Limit + Follow-up Link"""
    tid = f"test_scenario_1_{uuid.uuid4().hex[:6]}"
    print(f"\n\n🚀 === Scenario 1: Typo Handling & Context Link ({tid}) ===")

    # Turn 1: Typo "인쳔" + Limit 2
    run_turn(tid, "인쳔공항 최신 2건 알려줘")

    # Turn 2: Context Link
    run_turn(tid, "첫번째 사례 원본 파일 줘")


def run_scenario_2():
    """Scenario 2: Year Filtering (SQLite Fix Verification)"""
    tid = f"test_scenario_2_{uuid.uuid4().hex[:6]}"
    print(f"\n\n🚀 === Scenario 2: Year Filtering ({tid}) ===")

    # Turn 1: 2022 Data (Checking if it finds > 0 records)
    run_turn(tid, "가스공사 2022년 감사 사례 3개만")


def run_scenario_3():
    """Scenario 3: Specific ID/Item Context"""
    tid = f"test_scenario_3_{uuid.uuid4().hex[:6]}"
    print(f"\n\n🚀 === Scenario 3: Item Reference ({tid}) ===")

    # Turn 1: Broad search
    run_turn(tid, "한전KDN 최신 사례")

    # Turn 2: Specific Item Detail
    run_turn(tid, "2번 자료 더 자세히 설명해줘")

    # Turn 3: Link
    run_turn(tid, "그거 다운로드 링크")


if __name__ == "__main__":
    print("Running Agentic RAG Scenarios...")
    run_scenario_1()
    run_scenario_2()
    run_scenario_3()
