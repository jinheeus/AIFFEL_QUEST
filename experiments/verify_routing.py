import sys
import os

# Add module path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../03_agentic_rag")))

# from common.utils import load_env
# load_env()

from graph import node_router, route_start, route_post_retrieval, route_post_generation


# Mock State
def test_router():
    print("--- [Test] Router Verification ---")

    # Test 1: Chat
    state_chat = {"query": "안녕, 반갑다", "mode": ""}
    res_chat = node_router(state_chat)
    print(f"\n[Q1] 안녕 -> Mode: {res_chat.get('mode')}")
    assert res_chat["mode"] == "chat"
    assert route_start(res_chat) == "chat_worker"

    # Test 2: Fast Track
    state_fast = {"query": "최신 감사 사례 2개만 보여줘", "mode": ""}
    # Assuming LLM works correctly (We can't easily assert LLM output deterministically without mocking, but we trust the new prompt)
    # Let's run it and see. If LLM is nondeterministic, we manually set mode for edge tests.
    res_fast = node_router(state_fast)
    print(f"\n[Q2] 최신 사례 -> Mode: {res_fast.get('mode')}")

    # Edge Checks for Fast
    state_fast_simulated = {"mode": "fast"}
    next_node = route_post_retrieval(state_fast_simulated)
    print(f"   [Edge] Post-Retrieval for Fast: {next_node}")
    assert next_node == "generate"

    next_node_gen = route_post_generation(state_fast_simulated)
    print(f"   [Edge] Post-Generation for Fast: {next_node_gen}")
    assert next_node_gen == "summarize_conversation"

    # Test 3: Deep Track
    state_deep = {
        "query": "내부통제 실패 원인을 A와 B 사례를 비교해서 분석해줘",
        "mode": "",
    }
    res_deep = node_router(state_deep)
    print(f"\n[Q3] 분석 요청 -> Mode: {res_deep.get('mode')}")

    # Edge Checks for Deep
    state_deep_simulated = {"mode": "deep"}
    next_node_deep = route_post_retrieval(state_deep_simulated)
    print(f"   [Edge] Post-Retrieval for Deep: {next_node_deep}")
    assert next_node_deep == "grade_documents"

    next_node_gen_deep = route_post_generation(state_deep_simulated)
    print(f"   [Edge] Post-Generation for Deep: {next_node_gen_deep}")
    assert next_node_gen_deep == "verify_answer"

    print("\n✅ Routing Logic Verified!")


if __name__ == "__main__":
    test_router()
