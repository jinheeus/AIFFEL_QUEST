import sys
import os

# Add root directory for imports
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import pandas as pd
import ast
import json
from tqdm import tqdm
from time import time
from statistics import mean

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import Config

# Import Pipeline
# Add 02_advanced_rag to path
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_advanced_rag"
    )
)
try:
    from pipeline import HighContextRAGPipeline
except ImportError:
    # Fallback if run from different dir
    from .pipeline import HighContextRAGPipeline


def convert_matches_to_score(match_dict):
    count = sum(match_dict.values())
    if count >= 4:
        return 1.0
    elif count >= 2:
        return 0.6
    elif count == 1:
        return 0.2
    else:
        return 0.0


def run_local_evaluation(csv_path, output_path, limit=None, run_count=1):
    print("🚀 Starting Local Hybrid Evaluation...")

    # 1. Init Pipeline
    try:
        pipeline = HighContextRAGPipeline()
        print("✅ HighContextRAGPipeline Initialized (Hybrid Target)")
    except Exception as e:
        print(f"❌ Failed to init pipeline: {e}")
        return

    # 2. Init Judge LLM (GPT-4o)
    api_key = os.getenv("OPENAI_API_KEY") or Config.OPENAI_API_KEY
    if not api_key:
        print("⚠️ Warning: OPENAI_API_KEY not found. Scoring will be skipped.")
        judge_llm = None
    else:
        judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        print("✅ Judge LLM (GPT-4o-mini) Ready")

    # 3. Load Data
    try:
        df = pd.read_csv(csv_path)
        print(f"📄 Loaded {len(df)} queries from {csv_path}")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return

    if limit:
        df = df.head(limit)

    # 4. Evaluation Loop (Multi-Run)
    all_runs_results = []

    # Scoring Prompt
    system_prompt = """
당신은 감사문서 기반 질문 문서 유사도 평가 전문가입니다.
아래 다섯 가지 기준을 사용해 질문(question)과 문서(document)가 유사한지 판단하십시오.
판정은 매우 엄격하게 수행하며 True는 기준을 명확히 충족할 때만 선택하십시오.

[평가 기준]

1. 주제 일치(Topic Match)
   질문과 문서가 다루는 감사 분야가 세부 분야 수준에서 완전히 동일할 때만 True로 판단합니다.
   출장비 계약 수의계약 등 동일한 분야여야 하며 큰 범주가 비슷하거나 내부통제 같은 일반 단어만 겹치는 경우는 False입니다.

2. 세부쟁점 일치(Sub Issue Match)
   질문이 요구하는 핵심 쟁점이 문서에서 다루는 구체적 문제와 직접적으로 일치할 때만 True입니다.
   예를 들어 질문이 출장비 증빙 누락 문제를 묻는 경우 문서에도 증빙 누락 문제나 관련 부적정 지급 쟁점이 포함되어 있어야 합니다.
   같은 분야라도 문제 포인트가 다르면 False입니다.

3. 사건 메커니즘 유사(Case Mechanism Match)
   문제 발생 과정과 사건 전개 방식이 질문과 문서에서 동일한 경우에만 True입니다.
   절차 미준수로 인한 부당 지급과 같은 단계적 구조가 일치해야 합니다.
   결과만 비슷하거나 원리 구조가 다르면 False입니다.

4. 위반행위 패턴 유사(Violation Pattern Match)
   부정행위의 유형이 질문과 문서 양쪽에서 동일할 때 True입니다.
   허위 청구 부당 지급 규정 미준수 등 위반 패턴이 완전히 일치해야 합니다.
   유형이 다르면 False입니다.

5. 원인 구조 유사(Cause Pattern Match)
   문제의 근본 원인이 질문과 문서에서 동일할 때만 True입니다.
   내부통제 미흡 관리 감독 소홀 규정 미비 등 원인 체계가 같아야 합니다.
   개인 일탈이나 고의적 비위처럼 다른 구조라면 False입니다.

[출력 형식]
아래 JSON 형식으로만 출력하십시오.

{{
  "topic_match": true/false,
  "subtopic_match": true/false,
  "case_structure_match": true/false,
  "violation_pattern_match": true/false,
  "cause_pattern_match": true/false
}}

추가 설명 이유 해석 문장은 절대 포함하지 마십시오.
"""
    eval_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "[Question]\n{question}\n\n[Document]\n{document}"),
        ]
    )

    for run_idx in range(run_count):
        print(f"\n▶️ Running Retrieval Eval Batch {run_idx + 1}/{run_count}...")
        results = []

        for i, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Evaluating Run {run_idx + 1}"
        ):
            query = row.get("question") or row.get("query")
            if not query:
                continue

            # A. Retrieval (Hybrid)
            start_t = time()
            # Note: HighContextRAGPipeline.search_and_merge uses vector search, so result may vary slightly if HNSW params differ,
            # but usually deterministic for same query unless index changes.
            # However, running multiple times helps catch any underlying instability or API flakes.
            retrieved_docs_text = pipeline.search_and_merge(query, top_k=5)
            elapsed = time() - start_t

            # B. Scoring (LLM Judge)
            doc_scores = []
            if judge_llm and retrieved_docs_text:
                for doc_text in retrieved_docs_text:
                    try:
                        chain = eval_prompt | judge_llm
                        res_json = chain.invoke(
                            {"question": query, "document": doc_text}
                        )
                        content = (
                            res_json.content.replace("```json", "")
                            .replace("```", "")
                            .strip()
                        )
                        score_dict = json.loads(content)
                        final_score = convert_matches_to_score(score_dict)
                        doc_scores.append(final_score)
                    except Exception as e:
                        # print(f"Scoring Error: {e}")
                        doc_scores.append(0.0)

            mean_score = mean(doc_scores) if doc_scores else 0.0

            results.append(
                {
                    "run_id": run_idx + 1,
                    "query": query,
                    "retrieved_docs": retrieved_docs_text,
                    "doc_scores": doc_scores,
                    "mean_score": mean_score,
                    "latency": elapsed,
                }
            )

        all_runs_results.extend(results)

    # 5. Save Results
    res_df = pd.DataFrame(all_runs_results)
    res_df.to_csv(output_path, index=False)

    avg_score = res_df["mean_score"].mean()
    std_dev = (
        res_df.groupby("run_id")["mean_score"].mean().std() if run_count > 1 else 0.0
    )

    print(f"\n🏆 Evaluation Complete (x{run_count} Runs)!")
    print(f"   Overall Mean Score: {avg_score:.4f}")
    if run_count > 1:
        print(f"   Standard Deviation: ± {std_dev:.4f}")
    print(f"   Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="./99_archive/experiments/retrieval.csv")
    parser.add_argument("--output_path", default="./evaluate_hybrid_results_v2.csv")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of queries for test"
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")

    args = parser.parse_args()
    run_local_evaluation(args.csv_path, args.output_path, args.limit, args.runs)
