import requests
import json
import time

BASE_URL = "http://localhost:8000"
SESSION_ID = f"test_session_v2_{int(time.time())}"


def print_step(step_name):
    print(f"\n{'=' * 20} {step_name} {'=' * 20}")


HISTORY = []


def chat(query, session_id):
    global HISTORY
    url = f"{BASE_URL}/chat"

    # 1. Update History with User Query
    HISTORY.append({"role": "user", "content": query})

    payload = {
        "query": query,
        "session_id": session_id,
        "history": HISTORY,
    }
    print(f"User: {query}")

    # Simple streaming handler
    response = requests.post(url, json=payload, stream=True)
    full_answer = ""
    command_triggered = None

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data:"):
                try:
                    data = json.loads(decoded_line[5:])
                    if data.get("type") == "answer":
                        content = data.get("content")
                        print(content, end="", flush=True)
                        full_answer += content
                    elif data.get("type") == "command":
                        cmd = data.get("content")
                        print(f"\n[COMMAND RECEIVED]: {cmd}")
                        command_triggered = cmd
                except:
                    pass
    print("\n")

    # 2. Update History with Assistant Answer
    HISTORY.append({"role": "assistant", "content": full_answer})

    # Simulate Frontend triggering generation
    if command_triggered == "open_report":
        print_step("Step 4b: Triggering Report Generation (Frontend Simulation)")
        gen_url = f"{BASE_URL}/generate_report"

        # USE REAL HISTORY NOW
        print(f"[Test Debug] Using Real History ({len(HISTORY)} turns)")

        gen_payload = {
            "query": "make report",
            "session_id": session_id,
            "history": HISTORY,
            "additional_info": {
                "대상 기관": "테스트 공사",
                "사건 유형": "법인카드 유용",
                # Note: The agent might auto-extract this, but for simulation we pass what we 'know' or let the agent infer everything from history.
                # Currently generator.py passes empty additional_info usually, or minimal.
                # Let's see what happens if we pass 'None' or minimal, relying on HISTORY being the source of truth for DraftingAgent.
                "debug_note": "User provided detailed info in chat.",
            },
        }
        gen_resp = requests.post(gen_url, json=gen_payload)

        # Pretty print the final report JSON
        try:
            report_json = gen_resp.json()
            report_content = report_json.get("report", "No report content found")
            print(f"\n{'=' * 20} [FINAL GENERATED REPORT] {'=' * 20}\n")
            print(report_content)
            print(f"\n{'=' * 60}\n")
        except:
            print(f"Report Generation Response: {gen_resp.text}")

    return full_answer


print(f"Starting Test Session: {SESSION_ID}")

# 1. Greeting
print_step("Step 1: Greeting")
chat("안녕", SESSION_ID)

# 2. Search
print_step("Step 2: Search Cases")
chat("법인카드 관련 사례 3개만 알려줘", SESSION_ID)

# 3. Drill Down (Context Pivot)
# User asks for advice on writing a report based on case #2
print_step("Step 3: Drill Down & Advice")
chat(
    "1번 사례랑 비슷한 사건이 있어서 보고서를 쓴다면 내가 어떤걸 알려주면 될까?",
    SESSION_ID,
)

# 4. Report Request (The Core Test)
# User provides details requested in step 3
print_step("Step 4: Request Report with Details")
user_details = """
모두의연구소의 연구개발1팀에서 2024.12.25 경 법인카드로 주류 품목에서 위스키를 300만원 구매 → 업무 연관성 없음 확인됨.  
이에 따라 환수조치 완료, 교육 등 대응 방안 수립 중.
"""
chat(user_details, SESSION_ID)

# 5. Check Log
print(
    "\n[Test Complete] Check the backend logs to see if Drafting Agent was triggered and Report generated."
)
