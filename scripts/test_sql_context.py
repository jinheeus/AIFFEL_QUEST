import sys
import os
from dotenv import load_dotenv
from langchain_core.documents import Document

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rag_path = os.path.join(project_root, "03_agentic_rag")
sys.path.append(rag_path)
sys.path.append(project_root)

# Load env
load_dotenv(os.path.join(project_root, ".env"))

# Import from inside 03_agentic_rag
try:
    from modules.sql_retriever import SQLRetriever
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def run_test():
    retriever = SQLRetriever()

    # Simulate Context: 3 items
    context_docs = [
        Document(page_content="Content 1", metadata={"idx": 1001, "title": "Doc One"}),
        Document(page_content="Content 2", metadata={"idx": 1002, "title": "Doc Two"}),
        Document(
            page_content="Content 3",
            metadata={"idx": 838, "title": "Incheon Airport Case"},
        ),
    ]

    # Query referencing context
    query = "3번 사례 링크 알려줘"
    print(f"\n--- Test: Context Resolution ({query}) ---")

    docs = retriever.retrieve(query, context=context_docs)

    if docs:
        print(f"✅ Found {len(docs)} docs.")
        print(f"   First Doc Title: {docs[0].metadata.get('title')}")
        # Verify if it picked idx 838
        if docs[0].metadata.get("idx") == 838:
            print("   ✅ Correctly targeted IDX 838!")
        else:
            print(f"   ❌ Wrong IDX: {docs[0].metadata.get('idx')}")
    else:
        print("❌ No docs found.")


if __name__ == "__main__":
    run_test()
