# eval_langsmith.py
"""
JD 기반 자소서 코칭 파이프라인을 LangSmith로 평가하는 예시.
- rag.pipeline.build_all()을 그대로 호출해서
- LangSmith Dataset + Experiment 구조로 평가 실행.
"""

from langsmith import Client, evaluate
from rag.pipeline import build_all


# 1) 우리가 평가할 "앱" 함수 정의
def app_fn(inputs: dict) -> dict:
    """
    LangSmith가 호출하는 대상 함수.
    inputs 형식은 아래 EXAMPLES에 맞춘다.
    """
    jd_text = inputs["jd_text"]
    essays = inputs["essays"]
    user_job = inputs.get("user_job", "")
    user_stack = inputs.get("user_stack", "")

    # 너희가 이미 만든 파이프라인 그대로 사용
    report = build_all(
        jd_text=jd_text,
        essays=essays,
        user_job=user_job,
        user_stack=user_stack,
    )

    return {
        "overall": report.overall_scores.overall,
        "jd_match": report.overall_scores.jd_match,
        "clarity": report.overall_scores.clarity,
        "raw_report": report.model_dump(),
    }


# 2) 테스트용 예시들 (inputs / reference_outputs)
#    → 이 부분에 실제 JD/자소서 예시를 채우면 됨.
EXAMPLES = [
    {
        "inputs": {
            "jd_text": """[인텔리전스랩스그룹] 데이터 분석가

넥슨코리아채용 완료 시

조직 소개
- 인텔리전스랩스그룹은 다양한 게임 정보를 활용해 ‘빅데이터’, ‘머신러닝·딥러닝’, ‘인공지능(AI)’ 기술과 공학적 사고를 통해 솔루션을 만들고 게임 사용자와 넥슨 구성원이 사용할 서비스를 제공하는 조직입니다.
- [플랫폼본부(인텔리전스랩스) 테크블로그 바로가기]

팀 소개
- 플랫폼분석팀은 넥슨 플랫폼의 유저 경험 개선과 비즈니스 성과 향상을 위해 데이터를 기반으로 문제를 해결하는 팀입니다.
- Data Scientist, Data Engineer, AI Engineer 등 다양한 전문가들이 협업하며, 데이터 수집부터 분석, 인사이트 도출, 모델 실험까지 폭넓은 역할을 수행합니다.
- 분석가의 인사이트 탐색과 개선 활동이 AI 모델 개발 및 머신러닝 인프라와 유기적으로 연결되어 실질적인 성과로 이어질 수 있도록 돕고, 이를 정량적으로 평가할 수 있는 지표 체계를 구축합니다.

주요 업무
- 게임 및 플랫폼 데이터를 기반으로 유저 행동 분석, 서비스 개선을 위한 인사이트 도출
- 비즈니스 지표 설계 및 성과 분석을 통한 전략적 의사결정 지원
- AI/ML 실험 결과 분석 및 성능 평가, 개선 방향 제안
- 데이터 시각화 및 리포트 자동화 시스템 구축

[분석 환경]
- 기술 스택 : Spark, Databricks, Snowflake, AWS Console
- 사용 언어 : SQL, Python
- 업무 도구 : JupyterHub, Tableau, Power BI, Airflow, Notion, Confluence, GitLab
- AI 도구 : Azure OpenAI, Cursor, Microsoft Copilot

지원 자격
- 데이터 기반으로 문제를 정의하고 해결하는 데 흥미를 느끼시는 분
- 다양한 직군과 협업하며, 커뮤니케이션에 능숙하신 분
- 복잡한 데이터를 구조화하고 인사이트를 도출하는 데 강점을 가지신 분
- 겸손하고 배우는 자세로 동료들과 즐겁게 일할 수 있으신 분

[필수 자격 요건]
- SQL 및 Python을 활용한 데이터 분석 실무 경험이 있으신 분
- 클라우드 환경(GCP, AWS 등)에서 데이터 분석 및 시각화 경험이 있으신 분
- 통계적 사고 및 실험 설계 역량을 보유하신 분

우대 사항
- 게임 또는 플랫폼 서비스 관련 데이터 분석 경험이 있으신 분
- AI/ML 모델 성능 분석 및 개선 경험이 있으신 분
- 데이터 기반 비즈니스 지표 개선을 주도한 경험이 있으신 분

[선택 제출] 포트폴리오
- 데이터 분석 프로젝트 경험을 자유롭게 공유해주세요.
- 문제 정의, 분석 과정, 해결책, 정량적 성과 및 배운 점을 포함하면 더욱 좋습니다.

전형 절차
서류전형 → 과제전형 → 면접전형 → 최종합격
""",
            "essays": [
                """[본인의 데이터 분석 경험과 구체적인 사례를 작성해주세요.]
저는 데이터 분석 업무에 성실하게 임하며, 구체적으로는 [프로젝트명]에서 [특정 데이터 분석]을 통해 [구체적인 결과]를 도출했습니다. 다양한 데이터를 분석하며 문제를 창의적으로 해결하는 경험을 쌓았고, 이를 통해 조직에 긍정적인 영향을 주었습니다. SQL과 Python을 활용하여 [특정 분석 내용]을 수행하고, Tableau를 통해 [구체적인 인사이트]를 시각화하여 전달했습니다. 이러한 과정에서 비즈니스 성과 향상에 기여했고, 데이터 분석가로서의 역량을 지속적으로 발전시켰습니다. 앞으로도 데이터 기반 사고를 통해 더 나은 결과를 만들어내는 분석가가 되고 싶습니다.""",
                """[데이터 분석 업무를 수행하며 기여하고 싶은 분야의 목표는 무엇인가요]
귀사는 글로벌 게임 산업을 선도하는 기업으로, 데이터 분석을 통해 더 큰 성장을 이루고 있다고 생각합니다. 저는 이러한 환경에서 데이터 분석가로서 [구체적인 기여 방안]을 통해 회사의 발전에 기여하고 싶습니다. 유저 경험 개선과 비즈니스 성과 향상을 위한 다양한 분석을 수행하며, 조직과 함께 성장하는 것이 목표입니다. 특히 데이터와 기술을 활용해 더 나은 의사결정을 지원하고, 회사의 비전에 부합하는 분석가가 되고자 합니다. 이를 통해 넥슨코리아의 미래에 긍정적인 영향을 미치고 싶습니다.""",
                """[데이터 분석을 위해 사용했던 도구와 기술, 그리고 그 과정에서 직면했던 어려움과 해결 방안에 대해 설명해 주세요.]
데이터 분석을 위해 SQL, Python과 같은 다양한 도구를 사용해 왔으며, 클라우드 환경에서도 분석 경험을 쌓았습니다. 분석 과정에서 [구체적인 데이터 종류]를 다루며 복잡함을 느꼈지만, [구체적인 학습 방법]을 통해 이를 극복했습니다. 문제 상황에서도 긍정적인 태도로 임하며 해결책을 찾았고, 그 결과 분석 역량을 한층 더 강화할 수 있었습니다. 이러한 경험을 통해 어떤 환경에서도 유연하게 대응할 수 있는 분석가로 성장했다고 생각합니다.""",
                """[팀 내 협업 경험과 이를 통해 얻은 교훈 또는 성과를 구체적으로 서술해 주세요.]
저는 팀 내에서 원활한 협업을 중요하게 생각하며, [구체적인 협업 사례]를 통해 열린 자세로 소통하려 노력해 왔습니다. 다양한 직군의 동료들과 협업하며 서로의 의견을 존중했고, 이를 통해 더 나은 결과를 도출할 수 있었습니다. 팀워크를 바탕으로 프로젝트를 성공적으로 마무리했으며, 협업의 중요성을 다시 한번 느끼게 되었습니다. 이러한 경험은 앞으로 어떤 조직에서도 잘 적응할 수 있는 밑거름이 되었다고 생각합니다."""
            ],
            "user_job": "데이터 분석가",
            "user_stack": "SQL, Python, Tableau, 클라우드",
        },
        "outputs": {
            # 사람이 보기엔 대략 이 정도 점수라고 가정
            "target_overall": 90,
        },
        "metadata": {
            "case_name": "넥슨 인텔리전스랩스 데이터 분석가 1",
        },
    },
]


# 3) 간단한 코드 평가 함수 (공식 문서 스타일에 맞춤)
#    signature: (outputs, reference_outputs) 만 받는 형태
def overall_gap(outputs: dict, reference_outputs: dict) -> dict:
    """
    - outputs["overall"]        : AI가 낸 overall 점수
    - reference_outputs[...]    : 사람이 기대한 점수

    둘의 차이가 작을수록 score 1에 가깝게, 크면 0에 가깝게.
    """
    predicted = outputs["overall"]
    target = reference_outputs["target_overall"]

    diff = abs(predicted - target)
    # diff가 0이면 score=1, diff가 50 이상이면 score=0
    score = max(0.0, 1.0 - diff / 50.0)

    return {
        "key": "overall_gap",
        "score": score,
        "comment": f"pred={predicted}, target={target}, diff={diff}",
    }


if __name__ == "__main__":
    # 4) LangSmith Client로 Dataset 만들기/채우기
    client = Client()

    dataset_name = "jd-coverletter-eval"

    # 이미 같은 이름의 데이터셋이 있을 수도 있으니, 있으면 재사용
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="JD 기반 자소서 첨삭 모델 평가용 데이터셋",
        )

    # 예제들 추가 (여러 번 실행하면 중복될 수 있으니, 실제론 체크 로직을 넣어도 됨)
    client.create_examples(
        dataset_id=dataset.id,
        examples=EXAMPLES,
    )

    # 5) LangSmith 평가 실행
    results = evaluate(
        app_fn,                    # 평가 대상 함수
        data=dataset_name,         # 방금 만든 Dataset 이름
        evaluators=[overall_gap],  # 커스텀 metric
        experiment_prefix="jd-coach-v1",
        # 발표용이니 결과를 LangSmith에 업로드해서 UI로 보는 걸 추천
        upload_results=True,
    )

    print("실험 이름:", results.experiment_name)
    print("평균 metric:", results.metrics)
    print("LangSmith Datasets & Experiments 페이지에서 결과를 확인하세요.")
