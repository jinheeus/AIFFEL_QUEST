import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph import app
from pprint import pprint


def run_test(query: str):
    print(f"\n\n>>> Testing Query: {query}")
    inputs = {
        "query": query,
        "messages": [],
        "reflection_count": 0,
        "persona": "common",
    }
    try:
        # Stream the output to see the path
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"--- Node: {key} ---")
                if "plan" in value:
                    print(f"Plan: {value['plan']}")
                if "answer" in value:
                    print(f"Answer: {value['answer']}")

    except Exception as e:
        print(f"Graph Execution Error: {e}")


if __name__ == "__main__":
    # 1. Chit-chat Test
    run_test("안녕? 너는 누구니?")

    # 2. Research Test (General)
    run_test("야근 수당 지급 기준에 대해 알려줘")

    # 3. Audit Test (Specific Judgment)
    # run_test("계약서에 서명이 누락된 경우 규정 위반이야?")

    # 4. Filtering Test (Field Focus)
    run_test("감사보고서의 조치사항(Action)만 검색해서 알려줘")
