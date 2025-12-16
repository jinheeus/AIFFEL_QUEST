import json
import pandas as pd
import ast
from statistics import mean
from time import time
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import Config


def main():
    input_csv = "retrieval_with_results.csv"
    output_csv = "retrieval_final_score.csv"

    if not os.path.exists(input_csv):
        print(f"File {input_csv} not found. Run retrieval test first.")
        return

    # 1. Load Data
    df = pd.read_csv(input_csv, encoding="utf-8")

    # Parse stringified lists if needed (pandas might read them as strings)
    # The 'documents' column we saved in run_retrieval_test.py
    # If saved as string representation of list, we need eval.
    # But run_retrieval_test saves using to_csv. Lists become strings.
    def parse_docs(x):
        try:
            return ast.literal_eval(x)
        except:
            return []

    df["documents"] = df["documents"].apply(parse_docs)

    print(f"Loaded {len(df)} rows from {input_csv}")

    # 2. Setup LLM
    # Try to get API Key from Config or Env
    api_key = os.getenv("OPENAI_API_KEY") or Config.OPENAI_API_KEY
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        return

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    # 3. Prompt (Same as retrieval_evaluation.ipynb)
    system_prompt = """
당신은 감사문서 기반 질문 문서 유사도 평가 전문가입니다.
아래 다섯 가지 기준을 사용해 질문(question)과 문서(document)가 유사한지 판단하십시오.
판정은 매우 엄격하게 수행하며 True는 기준을 명확히 충족할 때만 선택하십시오.

[평가 기준]
1. 주제 일치(Topic Match): 질문과 문서가 다루는 감사 분야가 완전히 동일한 경우에만 True.
2. 세부쟁점 일치(Sub Issue Match): 질문이 요구하는 핵심 쟁점이 문서에서 다루는 구체적 문제와 직접적으로 일치할 때만 True.
3. 사건 메커니즘 유사(Case Mechanism Match): 문제 발생 과정과 사건 전개 방식이 질문과 문서에서 동일한 경우에만 True.
4. 위반행위 패턴 유사(Violation Pattern Match): 부정행위의 유형이 질문과 문서 양쪽에서 동일할 때 True.
5. 원인 구조 유사(Cause Pattern Match): 문제의 근본 원인이 질문과 문서에서 동일할 때만 True.

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
    user_prompt = """
[Question]
{question}

[Document]
{document}
"""
    evaluation_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", user_prompt)]
    )

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

    # 4. Run Evaluation
    all_doc_sims = []
    all_mean_sims = []

    start_time = time()

    for i, (_, row) in enumerate(df.iterrows()):
        question = (
            row["category"] + " " + row["question"]
        )  # Use Category + Question for context? Original used just "question"
        # Reverting to just Question to match original notebook logic exactly
        question = row["question"]

        documents = row["documents"]  # List of dicts

        doc_scores = []
        for doc in documents:
            # Format doc as text
            # doc is {'content':..., 'file':..., ...}
            doc_text = f"File: {doc.get('file')}\nContent: {doc.get('content')}"

            prompt = evaluation_prompt.format(question=question, document=doc_text)
            try:
                response = llm.invoke(prompt)
                result = json.loads(response.content)
                score = convert_matches_to_score(result)
            except Exception as e:
                print(f"Error evaluating doc: {e}")
                score = 0.0

            doc_scores.append(score)

        all_doc_sims.append(doc_scores)
        mean_score = mean(doc_scores) if doc_scores else 0.0
        all_mean_sims.append(mean_score)

        if (i + 1) % 5 == 0:
            elapsed = time() - start_time
            print(f"{i + 1}/{len(df)} Evaluated ({elapsed:.1f}s)")

    # 5. Save
    df["document_similarity"] = all_doc_sims
    df["mean_document_similarity"] = all_mean_sims
    df.to_csv(output_csv, index=False, encoding="utf-8")

    overall_mean = df["mean_document_similarity"].mean()
    print(f"Evaluation Complete. Overall Mean Score: {overall_mean:.4f}")
    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    main()
