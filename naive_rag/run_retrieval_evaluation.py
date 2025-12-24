import os
import re
import json
import ast
import pandas as pd
import numpy as np
from statistics import mean
from langchain_core.prompts import ChatPromptTemplate
from utils.llm import llm_openai

system_prompt = """
당신은 감사문서 기반 질문-문서 유사도 평가 전문가입니다.
아래 다섯 가지 기준을 사용하여 질문(question)과 문서(document)의 유사 여부를 판단하십시오.
판정은 엄격하게 수행하되, 본 평가는 RAG 시스템 고도화 단계별 성능 비교를 목적으로 하므로
각 기준은 독립적으로 판단하며, 상위 기준을 충족하지 못하더라도 하위 기준을 개별적으로 평가할 수 있습니다.
True는 해당 기준을 명확히 충족하는 경우에만 선택하십시오.

[평가 기준]

1. 주제 일치(Topic Match)
   질문과 문서가 다루는 감사 분야가 세부 감사 분야 수준에서 동일할 경우 topic_match는 true입니다.
   출장비, 계약·수의계약, 인사·복무, 금품수수 등 구체 감사 분야가 동일해야 하며,
   내부통제, 관리 미흡과 같은 포괄적 표현만 공통된 경우는 false입니다.

   질문에 특정 연도, 기간, 시점이 명시된 경우 문서의 date 값이 해당 기간과 일치해야 합니다.
   날짜 조건이 불일치하는 경우 topic_match는 false입니다.
   질문에 날짜나 기간 조건이 없는 경우에는 date를 판단 기준으로 사용하지 마십시오.

2. 세부쟁점 일치(Sub Issue Match)
   질문이 요구하는 핵심 쟁점이 문서에서 다루는 구체적 문제와 직접적으로 대응될 경우
   subtopic_match는 true입니다.
   감사 실무에서 통용되는 동의어, 표현 차이, 서술 방식 차이는 허용할 수 있습니다.
   동일한 감사 분야이더라도 문제의 초점이나 판단 대상이 다르면 false입니다.

3. 사건 메커니즘 유사(Case Mechanism Match)
   문제 발생의 절차적 흐름이나 사건 전개 방식이 질문과 문서에서 본질적으로 유사할 경우
   case_structure_match는 true입니다.
   모든 세부 단계가 완전히 동일할 필요는 없으나,
   주요 절차 위반 구조나 사건 진행 논리가 공통적으로 나타나야 합니다.
   결과만 유사하고 발생 과정의 구조가 다른 경우는 false입니다.

4. 위반행위 패턴 유사(Violation Pattern Match)
   위반 행위의 유형이 질문과 문서에서 동일하거나 감사 실무상 동일한 유형으로 분류될 수 있는 경우
   violation_pattern_match는 true입니다.
   허위 청구, 부당 지급, 규정 미준수 등 실질적으로 동일한 위반 패턴은 일치로 판단할 수 있습니다.
   위반 행위의 성격이 명확히 다른 경우는 false입니다.

5. 원인 구조 유사(Cause Pattern Match)
   문제의 근본 원인이 질문과 문서에서 동일하거나,
   동일한 관리·통제 구조상의 원인으로 설명될 수 있는 경우 cause_pattern_match는 true입니다.
   내부통제 미흡, 관리·감독 소홀, 규정 미비 등 구조적 원인이 공통적으로 나타나면 일치로 판단할 수 있습니다.
   개인의 고의적 비위나 일탈 등 원인 구조가 명확히 다른 경우는 false입니다.

[출력 형식]
반드시 아래 JSON 형식으로만 출력하십시오.
각 항목에 대해 판단 결과(decision)와 그 이유(reason)를 1~2문장으로 간략히 작성하십시오.
이유(reason)는 반드시 질문(question)과 문서(document)에 명시적으로 포함된 정보만을 근거로 작성하십시오.

{{
  "topic_match": {{
      "decision": true/false,
      "reason": "판단 근거 작성"
  }},
  "subtopic_match": {{
      "decision": true/false,
      "reason": "판단 근거 작성"
  }},
  "case_structure_match": {{
      "decision": true/false,
      "reason": "판단 근거 작성"
  }},
  "violation_pattern_match": {{
      "decision": true/false,
      "reason": "판단 근거 작성"
  }},
  "cause_pattern_match": {{
      "decision": true/false,
      "reason": "판단 근거 작성"
  }}
}}

JSON 이외의 서론이나 추임새는 절대 포함하지 마십시오.
"""

user_prompt = """
[Question]
{question}

[Document]
{document}
"""

evaluation_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", user_prompt)
])

def parse_json_response(content):
    try:
        cleaned = re.sub(r"```json\s*|\s*```", "", content).strip()
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]
        return json.loads(cleaned)
    except:
        return {k: {"decision": False, "reason": "Error"} for k in 
                ["topic_match", "subtopic_match", "case_structure_match", "violation_pattern_match", "cause_pattern_match"]}

def calculate_score(result_json):
    if not isinstance(result_json, dict): return 0.0
    count = sum(1 for v in result_json.values() if isinstance(v, dict) and v.get("decision") is True)
    
    if count >= 4: return 1.0
    elif count >= 2: return 0.6
    elif count == 1: return 0.2
    else: return 0.0

def run_evaluation_logic(df, run_id):
    print(f"\n[INFO] Starting Evaluation Run #{run_id}")
    
    all_mean_sims = []
    all_details = []
    
    total_len = len(df)
    
    for i, row in df.iterrows():
        question = row["question"]
        context_texts = row["contexts"]
        scores = []
        row_details = []

        if not context_texts or not isinstance(context_texts, list):
            all_mean_sims.append(0.0)
            all_details.append([])
            continue

        for text in context_texts:
            content_text = str(text)[:2000]

            chain = evaluation_template | llm_openai
            try:
                res = chain.invoke({"question": question, "document": content_text})
                parsed_res = parse_json_response(res.content)
            except Exception:
                parsed_res = {}
            
            score = calculate_score(parsed_res)
            scores.append(score)
            row_details.append(parsed_res)

        avg_score = mean(scores) if scores else 0.0
        all_mean_sims.append(avg_score)
        all_details.append(row_details)
        
        if (i+1) % 10 == 0:
            print(f"   - Progress: {i+1}/{total_len}")

    df[f"score_run_{run_id}"] = all_mean_sims
    df[f"details_run_{run_id}"] = all_details
    
    dataset_avg = mean(all_mean_sims)
    
    return df, dataset_avg

if __name__ == "__main__":
    DATA_FILE = "./results/naive_retrieval_data.csv"
    OUTPUT_FILE = "./results/naive_retrieval_data_score.csv"
    NUM_TRIALS = 3
    
    if not os.path.exists(DATA_FILE):
        print(f"Data file not found: {DATA_FILE}")
        exit()

    df = pd.read_csv(DATA_FILE, encoding="utf-8")
    
    try:
        df["contexts"] = df["contexts"].apply(ast.literal_eval)
    except:
        pass

    run_scores = []
    print("=" * 50)
    print("Starting Evaluation Process (Total 3 Runs)")
    print("=" * 50)

    for n in range(1, NUM_TRIALS + 1):
        df, score = run_evaluation_logic(df, n)
        run_scores.append(score)
        print(f"▶ Run {n} Completed | Mean Score: {score:.4f}")

    score_columns = [f"score_run_{n}" for n in range(1, NUM_TRIALS + 1)]
    df["mean_score"] = df[score_columns].mean(axis=1)

    os.makedirs("./results", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    total_mean = np.mean(run_scores)
    total_std = np.std(run_scores)

    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT")
    print("="*50)
    for i, s in enumerate(run_scores, 1):
        print(f"Run {i} : {s:.4f}")
    print("-" * 50)
    print(f"Final Result: {total_mean:.4f} ± {total_std:.4f}")
    print("="*50)
    print(f"Result saved to: {OUTPUT_FILE}")