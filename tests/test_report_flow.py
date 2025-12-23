import requests
import json
import time

BASE_URL = "http://localhost:8000"
SESSION_ID = f"test_session_{int(time.time())}"


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
            "additional_info": {"대상 기관": "테스트 공사", "감사 기간": "2024년 1월"},
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
chat("안녕, 난 감사관이야.", SESSION_ID)

# 2. Search
print_step("Step 2: Search Cases")
chat("횡령 관련 사례 2개만 찾아줘.", SESSION_ID)

# 3. Drill Down (Context Pivot)
print_step("Step 3: Drill Down")
chat("2번 사례가 내가 찾던거랑 비슷하네. 좀 더 자세히 알려줘.", SESSION_ID)

# 4. Report Request (The Core Test)
print_step("Step 4: Request Report")
user_prompt = """
야 저 내용으로 일단 초안한번 적어볼래? 
우리는 '김과장'이 자가차량 이용한 것 처럼 출장 다녀왔는데 
알고보니 전철로 다녀와서 유류비로 과다 지급 되었어 약 90만원 가량.
"""
chat(user_prompt, SESSION_ID)

# 5. Check Log for explicit extraction
print(
    "\n[Test Complete] Check the backend logs to see if 'Source A' and 'Source B' were correctly separated."
)
