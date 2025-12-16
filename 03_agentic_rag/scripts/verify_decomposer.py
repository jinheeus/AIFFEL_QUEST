import sys
import os

# Add root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from modules.decomposer import decompose_query
from state import AgentState
from graph import app  # The compiled graph


def test_decomposer_module():
    print("\n=== [Test 1] Decomposer Module Unit Test ===")

    # 1. Simple Query
    state1 = {"query": "직원 횡령 시 징계 기준은?"}
    res1 = decompose_query(state1)
    print(f"Simple Input: {state1['query']}\nOutput: {res1['sub_queries']}")

    # 2. Complex Query
    state2 = {"query": "음주운전 징계 기준과 금품수수 징계 기준 비교해줘"}
    res2 = decompose_query(state2)
    print(f"Complex Input: {state2['query']}\nOutput: {res2['sub_queries']}")


def test_full_workflow():
    print("\n=== [Test 2] Full Workflow Test (Graph) ===")
    query = "음주운전 징계 기준과 금품수수 징계 기준 비교해줘"
    print(f"Query: {query}")

    try:
        final_state = app.invoke({"query": query})

        print("\n[Final Sub-Queries]")
        print(final_state.get("sub_queries"))

        print(f"\n[Retrieved Documents Count]: {len(final_state.get('documents', []))}")

        print("\n[Final Answer Preview]")
        print(final_state["answer"][:200] + "...")

    except Exception as e:
        print(f"Workflow Error: {e}")


if __name__ == "__main__":
    test_decomposer_module()
    test_full_workflow()
