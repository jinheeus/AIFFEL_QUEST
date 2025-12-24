import os
import sys
from dotenv import load_dotenv

# Add project root to path
# __file__ is agentic_rag_v2/test_sop.py
# dirname is agentic_rag_v2
# dirname(dirname) is root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import app

load_dotenv()


def test_sop_flow():
    print("🚀 Starting SOP Graph Test...")

    # Test Query: Clearly implies a judgment/violation check
    # Scenario: Splitting a 30M KRW contract into two 15M KRW private contracts (Classic violation)
    query = "정보시스템 유지보수 용역을 1500만원씩 2회로 나누어 수의계약 체결한 것이 규정 위반이야?"

    print(f"Query: {query}")

    inputs = {"query": query}

    events = app.stream(inputs)

    print(f"\n--- Graph Execution Trace ---")
    final_output = None

    for event in events:
        for key, value in event.items():
            print(f"\n✅ Finished Node: {key}")
            if key == "analyze_query":
                print(f"   Category: {value.get('category')}")
                print(f"   Persona: {value.get('persona')}")

            if key == "extract_facts":
                print(f"   Facts: {value.get('facts')}")

            if key == "matched_regulations":
                print(
                    f"   Regs Found: {len(str(value.get('matched_regulations', [])))} chars"
                )

            if key == "evaluate_compliance":
                print(f"   Compliance: {value.get('compliance_result')}")

            if key == "determine_disposition":
                final_output = value.get("answer")
                print(f"   Disposition Answer Generated.")

            if key == "defense_agent":
                print(f"   [Shield] Defense: {value.get('defense_argument')[:100]}...")

            if key == "prosecution_agent":
                print(
                    f"   [Sword] Prosecution: {value.get('prosecution_argument')[:100]}..."
                )

            if key == "judge_verdict":
                final_output = value.get("answer")
                print(
                    f"   [Gavel] Final Verdict: {value.get('final_judgment')[:100]}..."
                )

    print(f"\n--- Final Answer ---\n{final_output}")


if __name__ == "__main__":
    test_sop_flow()
