import sys
import os

# Add module path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../03_agentic_rag")))

# from common.utils import load_env
# load_env()

from modules.field_selector import field_selector


def test_field_selector():
    print("--- [Test] Field Selector Verification ---")

    # Test Query: Requesting Latest + Limit 2
    state = {
        "query": "최신 감사 사례 2개만 보여줘",
        # "search_query": "감사 사례", # REMOVED: In proper flow, Field Selector runs on original query first.
        "messages": [],
    }

    print(f"Input Query: {state['query']}")
    result = field_selector(state)

    print("\n[Result]")
    print(f"Fields: {result.get('selected_fields')}")
    print(f"Filters: {result.get('metadata_filters')}")

    filters = result.get("metadata_filters", {})

    # Assertions
    if filters.get("sort") == "date_desc":
        print("✅ Sort 'date_desc' extracted successfully.")
    else:
        print("❌ Failed to extract sort 'date_desc'.")

    if filters.get("k") == 2:
        print("✅ Limit 'k=2' extracted successfully.")
    else:
        print(f"❌ Failed to extract limit 'k=2'. Got: {filters.get('k')}")


if __name__ == "__main__":
    test_field_selector()
