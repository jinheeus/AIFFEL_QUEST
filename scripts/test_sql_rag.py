import sys
import os
from dotenv import load_dotenv

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rag_path = os.path.join(project_root, "03_agentic_rag")
sys.path.append(rag_path)
sys.path.append(project_root)

# Load env
load_dotenv(os.path.join(project_root, ".env"))

# Import from inside 03_agentic_rag
try:
    from graph import app
    from state import AgentState
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)


def run_test():
    query = "인천국제공항공사 최신 2건 알려줘"
    print(f"Testing Query: {query}")

    # Run the graph
    inputs = {"query": query}
    config = {"recursion_limit": 50, "configurable": {"thread_id": "test_sql_2"}}

    print("--- Starting Graph ---")
    try:
        for event in app.stream(inputs, config=config):
            for k, v in event.items():
                print(f" -> Node: {k}")
                if k == "router":
                    print(f"    Mode: {v.get('mode')}, Category: {v.get('category')}")
                if k == "retrieve_sql":
                    docs = v.get("documents", [])
                    print(f"    Docs Found: {len(docs)}")
                    if docs:
                        print(f"    First Doc: {docs[0].page_content[:100]}...")
                if k == "generate":
                    ans = v.get("answer", "")
                    print(f"    Answer: {ans[:200]}...")
    except Exception as e:
        print(f"Graph Execution Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_test()
