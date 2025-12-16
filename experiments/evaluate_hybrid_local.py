import os
import sys
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
# Add 02_high_context_rag to path to import pipeline module
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_high_context_rag")
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


def run_local_evaluation(csv_path, output_path, limit=None):
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

    # 4. Evaluation Loop
    results = []

    # Scoring Prompt
    system_prompt = """
당신은 감사문서 기반 질문 문서 유사도 평가 전문가입니다.
[평가 기준]
1. 주제 일치(Topic Match)
2. 세부쟁점 일치(Sub Issue Match)
3. 사건 메커니즘 유사(Case Mechanism Match)
4. 위반행위 패턴 유사(Violation Pattern Match)
5. 원인 구조 유사(Cause Pattern Match)

[출력 형식]
JSON으로만 출력:
{{
  "topic_match": true/false,
  "subtopic_match": true/false,
  "case_structure_match": true/false,
  "violation_pattern_match": true/false,
  "cause_pattern_match": true/false
}}
"""
    eval_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "[Question]\n{question}\n\n[Document]\n{document}"),
        ]
    )

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        query = row.get("question") or row.get("query")
        if not query:
            continue

        # A. Retrieval (Hybrid)
        # pipeline.search_and_merge returns list of STRINGS (reconstructed docs)
        start_t = time()
        retrieved_docs_text = pipeline.search_and_merge(query, top_k=5)
        elapsed = time() - start_t

        # B. Scoring (LLM Judge)
        doc_scores = []
        if judge_llm and retrieved_docs_text:
            for doc_text in retrieved_docs_text:
                try:
                    # Run Judge
                    chain = eval_prompt | judge_llm
                    res_json = chain.invoke({"question": query, "document": doc_text})
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
                "query": query,
                "retrieved_docs": retrieved_docs_text,  # Save text for review
                "doc_scores": doc_scores,
                "mean_score": mean_score,
                "latency": elapsed,
            }
        )

    # 5. Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_path, index=False)

    avg_score = res_df["mean_score"].mean()
    print(f"\n🏆 Evaluation Complete!")
    print(f"   Overall Mean Score: {avg_score:.4f}")
    print(f"   Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="./99_archive/experiments/retrieval.csv")
    parser.add_argument("--output_path", default="./evaluate_hybrid_results.csv")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of queries for test"
    )

    args = parser.parse_args()
    run_local_evaluation(args.csv_path, args.output_path, args.limit)
