import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../03_agentic_rag")))

from modules.vector_retriever import get_retriever


def test_top_k_override():
    print("--- Testing Top-K Override Logic ---")
    retriever = get_retriever()

    query = "안전사고"

    # 1. Default (k=5)
    print("\n[Test 1] Default Search (k=5)")
    docs_default = retriever.search_and_merge(query, top_k=5)
    print(f" -> Retrieved: {len(docs_default)}")

    # 2. Override (k=2)
    print("\n[Test 2] Filtered Search (k=2)")
    docs_override = retriever.search_and_merge(query, top_k=5, filters={"k": 2})
    print(f" -> Retrieved: {len(docs_override)}")

    if len(docs_override) == 2:
        print("\n✅ SUCCESS: Top-K override working!")
    else:
        print(f"\n❌ FAIL: Expected 2, got {len(docs_override)}")


if __name__ == "__main__":
    test_top_k_override()
