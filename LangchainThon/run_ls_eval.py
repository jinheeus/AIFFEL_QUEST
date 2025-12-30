# run_ls_eval.py

import os
from dotenv import load_dotenv

from langsmith import Client
from rag.pipeline import build_all
from utils.logging import run_langsmith_eval

# ----------------------------------------------------
# 1) 환경 변수 로드 및 LangSmith Dataset 불러오기
# ----------------------------------------------------
load_dotenv()
client = Client()

dataset_id = os.getenv("LANGSMITH_DATASET_ID", "")
if not dataset_id:
    raise ValueError("LANGSMITH_DATASET_ID 가 .env에 설정되어 있지 않습니다.")

# ❌ get_dataset(...) 이 아니라
# ✅ read_dataset(dataset_id=...) 를 써야 합니다.
dataset = client.read_dataset(dataset_id=dataset_id)


# ----------------------------------------------------
# 2) LangSmith evaluate() 에 넘길 target 함수 정의
#    - Dataset 안의 inputs 형식과 맞춰서 작성
# ----------------------------------------------------
def app_fn(inputs: dict):
    """
    LangSmith evaluate()가 각 example.inputs 를 넣어서 호출하는 함수.
    우리가 실제 서비스에서 사용하는 build_all()을 그대로 호출해서
    FullReport를 만든 뒤, dict 형태로 반환한다.
    """
    jd_text = inputs.get("jd_text", "")
    essays = inputs.get("essays", [])
    user_job = inputs.get("user_job", "")
    user_stack = inputs.get("user_stack", "")

    report = build_all(
        jd_text=jd_text,
        essays=essays,
        user_job=user_job,
        user_stack=user_stack,
    )

    # evaluator들이 참고할 수 있도록 전체 리포트를 dict로 반환
    return {
        "report": report.model_dump()
    }


# ----------------------------------------------------
# 3) LangSmith 평가 실행 (LLM-as-judge + groundedness)
# ----------------------------------------------------
if __name__ == "__main__":
    results = run_langsmith_eval(
        runs=app_fn,   # 위에서 정의한 target 함수
        dataset=dataset,
    )

    exp_name = getattr(results, "experiment_name", None)
    exp_url = getattr(results, "url", None)

    print("✅ LangSmith 평가 실행 완료")
    if exp_name:
        print("  - 실험 이름:", exp_name)
    if exp_url:
        print("  - LangSmith에서 보기:", exp_url)
