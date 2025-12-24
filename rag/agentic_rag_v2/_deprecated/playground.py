import os
import sys

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from graph import app


def test_query(query: str):
    print(f"\n{'=' * 50}")
    print(f"User Query: {query}")
    print(f"{'=' * 50}")

    # 그래프 실행
    inputs = {"query": query}
    # invoke는 최종 state를 반환합니다.
    result = app.invoke(inputs)

    # 결과 출력
    print("\n[Result]")
    print(f"- Category: {result.get('category')}")
    print(f"- Persona: {result.get('persona')}")
    print(f"- Answer:\n{result.get('answer')}")


if __name__ == "__main__":
    queries = [
        # 1. 일반 검색 (Audit)
        "출장비 횡령 시 처분 기준은 어떻게 돼?",
        # 2. 통계 질문 (Stats - 2021년)
        "2021년도에 관련해서 가장 많이 지적된 사항이 뭐야?",
        # 3. 경영진 페르소나 테스트 (Manager)
        # 3. 경영진 페르소나 테스트 (Manager)
        "경영진 보고용으로 정리해줘: 출장비 규정 위반 현황이 어때?",
        # 4. GraphRAG 테스트 (Regulation Search)
        "국가를 당사자로 하는 계약에 관한 법률을 위반한 사례 찾아줘",
    ]

    for q in queries:
        test_query(q)
