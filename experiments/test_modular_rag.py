import sys
import os
from dotenv import load_dotenv

# Load env
load_dotenv()

# Add 03_agentic_rag to sys.path to allow imports from numbered folder
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
agent_dir = os.path.join(root_dir, "03_agentic_rag")
sys.path.append(agent_dir)

# Also add root for config
sys.path.append(root_dir)

from graph import app
from state import AgentState


def run_test():
    print("🚀 Starting Modular RAG Test...")

    # Test Query
    query = "안뇽"
    # query = "인사 채용 관련 감사 기준과 주요 위반 사례를 알려줘"
    print(f"👉 Query: {query}")

    initial_state = AgentState(
        query=query,
        persona="common",
        messages=[],
        retrieval_count=0,
        reflection_count=0,
    )

    # Stream the graph
    print("\n--- Graph Execution Trace ---")
    captured_answer = None

    try:
        final_state = None
        # Add thread_id for Checkpointer
        config = {
            "configurable": {"thread_id": "test_thread_v1"},
            "recursion_limit": 20,
        }

        for output in app.stream(initial_state, config):
            for node_name, state_update in output.items():
                print(f"✅ Node Finished: {node_name}")
                if node_name == "field_selector":
                    print(f"   -> Filters: {state_update.get('metadata_filters')}")
                    print(f"   -> Fields: {state_update.get('selected_fields')}")
                elif node_name == "grade_documents":
                    print(f"   -> Grade: {state_update.get('grade_status')}")
                    print(f"   -> Docs: {len(state_update.get('documents', []))}")
                elif node_name == "sop_retriever":
                    sop_ctx = state_update.get("sop_context", "")
                    print(f"   -> SOP Retrieved: {len(sop_ctx)} chars")
                    print(f"   -> SOP Snippet: {sop_ctx[:50]}...")
                elif node_name == "rewrite_query":
                    print(f"   -> New Query: {state_update.get('search_query')}")
                elif node_name == "verify_answer":
                    print(f"   -> Hallucinated: {state_update.get('is_hallucinated')}")
                    print(f"   -> Useful: {state_update.get('is_useful')}")
                elif node_name == "generate":
                    # Capture answer when available
                    answer_text = state_update.get("answer", "")
                    print(f"   -> Generated Answer Check: {len(answer_text)} chars")
                    captured_answer = answer_text

            # Keep track of final state
            final_state = output

        print("\n--- Final Answer ---")
        if captured_answer:
            print(captured_answer)
        else:
            print("No answer captured (check 'generate' node output).")

    except Exception as e:
        print(f"❌ Execution Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_test()
