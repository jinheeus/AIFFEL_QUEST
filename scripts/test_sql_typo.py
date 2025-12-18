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
    from modules.sql_retriever import SQLRetriever
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def run_test():
    retriever = SQLRetriever()

    # Test 1: Date Function (Checking for '2023년 거')
    query1 = "인천국제공항공사 2023년 감사 결과 2개 알려줘"
    print(f"\n--- Test 1: Date Function ({query1}) ---")
    docs1 = retriever.retrieve(query1)
    if docs1:
        print(f"✅ Found {len(docs1)} docs.")
        print(f"   Date: {docs1[0].metadata['date']}")
    else:
        print("❌ No docs found for date query.")

    # Test 2: Typo Handling ('인쳔공항')
    query2 = "인쳔공항 최신 2건"
    print(f"\n--- Test 2: Typo Handling ({query2}) ---")
    docs2 = retriever.retrieve(query2)
    if docs2:
        print(f"✅ Found {len(docs2)} docs.")
        print(f"   Company: {docs2[0].metadata['company']}")
    else:
        print("❌ No docs found for typo query.")


if __name__ == "__main__":
    run_test()
