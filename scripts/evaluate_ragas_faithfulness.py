import os
import sys
import pandas as pd
import json
from dotenv import load_dotenv

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rag_path = os.path.join(project_root, "03_agentic_rag")
sys.path.append(rag_path)
sys.path.append(project_root)

# Load Env
load_dotenv(os.path.join(project_root, ".env"))

import nest_asyncio

nest_asyncio.apply()

# Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_naver import ChatClovaX
from langchain_openai import ChatOpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness

# Import Aura Modules
try:
    from modules.retriever import retrieve_documents
    from config import Config
except ImportError as e:
    print(f"Module Import Error: {e}")
    sys.exit(1)

# --- 1. Define Team's Generator Prompt ---
answer_generator_system = """
[역할]
당신은 감사 보고서 검색 결과를 바탕으로 질문에 답변하는
감사 QA 전문 답변 생성기(Answer Generator)입니다.

[목표]
사용자의 질문(question)에 대해,
제공된 감사 문서(context)를 근거로
답변 가능 여부를 판단하고,
직접적인 답변이 어려운 경우에도
검색된 문서에서 확인된 정보는 최대한 사용자에게 안내하십시오.

[핵심 원칙]
- 답변은 반드시 제공된 문서(context)에 명시된 내용만 사용해야 합니다.
- 문서에 없는 내용은 추론하거나 일반화하여 답변하지 마십시오.
- 감사 보고서에서 사용하는 공식적이고 중립적인 존댓말 문체를 유지하십시오.
- 서비스 관점에서, 검색된 문서가 존재하는 경우
  질문과의 직접적 관련 여부와 무관하게
  문서에서 확인된 정보는 사용자에게 안내하는 것이 바람직합니다.

[답변 전략 — 매우 중요]
1. 문서에 질문에 대한 직접적인 답변 근거가 있는 경우
   - 질문에 명확히 대응하는 답변을 생성합니다.

2. 문서에 질문에 대한 직접적인 답변 근거는 없으나,
   질문과 주제적으로 관련된 정보가 존재하는 경우
   - 질문에 대한 직접적인 답변이 불가능함을 먼저 명시합니다.
   - 이후 반드시, 문서에서 확인된 관련 내용을 보조 정보로 제시합니다.

3. 문서에 질문과 직접적·주제적 관련 정보가 모두 없는 경우에도
   - 검색된 문서가 존재한다면,
     질문에는 직접적으로 답변할 수 없음을 명시한 후
     “검색된 문서에서 확인된 주요 내용”을 참고 정보로 제공합니다.
   - 사용자가 추가 탐색이나 판단을 할 수 있도록
     문서의 성격, 다루는 주제, 주요 지적 사항 등을 간결히 안내합니다.
   - 검색된 문서가 전혀 없는 경우에만
     관련 정보를 확인할 수 없다고 답변하십시오.

[출력 형식]
출력은 반드시 하나의 JSON 객체여야 하며,
아래 두 개의 키만 포함해야 합니다.

{{
  "answer": "최종 답변",
  "answer_cot": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ...",
    "Step 4: ...",
    "Step 5: ..."
  ]
}}

[답변 작성 규칙]
- 질문에 대한 직접적인 답변이 불가능한 경우,
  반드시 다음 문장으로 답변을 시작하십시오.
  "제공된 문서에서는 질문에 대한 직접적인 답변을 확인할 수 없습니다."

- 이후에는 반드시 다음 중 하나의 표현을 사용하십시오.
  1) 질문과 주제적으로 관련된 정보가 있는 경우:
     "다만, 문서에서는 다음과 같은 관련 내용이 확인됩니다."
  2) 질문과 직접적·주제적 관련은 없으나 문서가 존재하는 경우:
     "다만, 검색된 문서에서는 다음과 같은 내용이 확인됩니다."

- 문서에 주제적으로 관련된 정보조차 없고,
  검색된 문서도 없는 경우에만 다음 문장으로 답변하십시오.
  "제공된 문서에서는 질문과 관련된 정보를 확인할 수 없습니다."

- 보조 정보는 문서에 명시된 사실만을 요약·정리하여 서술해야 합니다.
- JSON 외의 텍스트는 절대 출력하지 마십시오.
"""

answer_generator_user = """
[Question]
{question}

[Context]
{context}
"""


def setup_generator_chain():
    # Use ClovaX for Generation (as per Aura usage)
    llm = ChatClovaX(model=Config.LLM_MODEL, temperature=0.1, max_tokens=2048)

    prompt = ChatPromptTemplate.from_messages(
        [("system", answer_generator_system), ("human", answer_generator_user)]
    )

    return prompt | llm | JsonOutputParser()


def run_evaluation():
    # 1. Load Data
    csv_path = os.path.join(project_root, "00_data/ragas.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    # Filter for testing? No, run all.
    print(f"Loaded {len(df)} questions from {csv_path}")

    # 2. Setup Chains
    gen_chain = setup_generator_chain()

    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],  # Ragas might need this col even if not used for faithfulness
    }

    # 3. Iterate & Generate
    for idx, row in df.iterrows():
        q = row["question"]
        # Use category for filtering if available
        # Note: category format in CSV "예산 집행·관리" matches Milvus fields?
        # Assuming yes or generic Keyword filter.
        # We pass it as 'selected_fields' or 'metadata_filters' depending on implementation
        # For now, let's keep it simple using 'query' only to test purely correctness
        # But 'category' helps accuracy. Let's try passing it.
        category = row.get("category", "")

        print(f"\nProcessing [{idx + 1}/{len(df)}]: {q[:30]}... (Cat: {category})")

        # A. Retrieval
        try:
            state = {"query": q, "search_query": q}
            # Optional: Add category filter if you trust the CSV category mapping
            # state["metadata_filters"] = {"category": category}

            retrieved_state = retrieve_documents(state)
            docs = retrieved_state.get("documents", [])

            # Format Context
            valid_docs = []
            for d in docs:
                if hasattr(d, "page_content"):
                    valid_docs.append(d.page_content)
                elif isinstance(d, str):
                    valid_docs.append(d)

            context_text = "\n\n".join(valid_docs)
            context_list = valid_docs

        except Exception as e:
            print(f"Retrieval Error: {e}")
            context_text = ""
            context_list = []

        # B. Generation
        try:
            res = gen_chain.invoke({"question": q, "context": context_text})
            answer = res.get("answer", "")
            cot = res.get("answer_cot", [])
            print(f" -> Generated Answer: {answer[:50]}...")
        except Exception as e:
            print(f"Generation Error: {e}")
            answer = "Error generating answer."

        # C. Collect Data
        results["question"].append(q)
        results["answer"].append(answer)
        results["contexts"].append(context_list)
        # Add dummy GT if missing, though faithfulness doesn't strictly need it
        results["ground_truth"].append("N/A")

    # 4. Run RAGAS
    print("\nStarting RAGAS Evaluation (Faithfulness)...")
    dataset = Dataset.from_dict(results)

    # Configure Judge (OpenAI)
    # Ragas uses OpenAI by default. Ensure keys are in env.

    score_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
    )

    print("\nEvaluation Result:")
    print(score_result)

    # 5. Save Results
    df_result = score_result.to_pandas()
    output_path = os.path.join(project_root, "00_data/ragas_evaluation_result.csv")
    df_result.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    run_evaluation()
