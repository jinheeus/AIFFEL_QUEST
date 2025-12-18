import os
import sys
import json
import argparse
import pandas as pd
from tqdm import tqdm
from statistics import mean
import concurrent.futures

# --- INSTALL COMMANDS (Run these in Colab first) ---
# !pip install -q pymilvus langchain langchain-openai langchain-community sentence-transformers tqdm pandas


# --- CONFIGURATION ---
class Config:
    # UPDTAE THESE WITH YOUR CREDENTIALS
    MILVUS_URI = os.getenv("MILVUS_URI", "./milvus_demo.db")
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    # Use GPU for Reranker?
    DEVICE = "cuda" if os.getenv("COLAB_GPU") else "cpu"


# --- IMPORTS ---
try:
    from pymilvus import MilvusClient
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from sentence_transformers import CrossEncoder
except ImportError:
    print(
        "❌ Missing dependencies. Please run: pip install pymilvus langchain langchain-openai sentence-transformers tqdm pandas"
    )
    sys.exit(1)


# --- PIPELINE CLASS (Minified) ---
class HighContextRAGPipeline:
    def __init__(self):
        print(f"⚙️ Initializing Pipeline on {Config.DEVICE}...")

        # 1. Reranker (BAAI/bge-reranker-v2-m3)
        try:
            self.reranker = CrossEncoder(
                "BAAI/bge-reranker-v2-m3",
                max_length=512,
                device=Config.DEVICE,
                automodel_args={"torch_dtype": "auto"},
            )
        except Exception as e:
            print(f"⚠️ Failed to load BAAI/bge-reranker-v2-m3: {e}")
            print("Fallback to ms-marco-MiniLM-L-6-v2")
            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", device=Config.DEVICE
            )

        # 2. Milvus
        try:
            self.client = MilvusClient(uri=Config.MILVUS_URI, token=Config.MILVUS_TOKEN)
            self.collection_name = "audit_rag_hybrid_v1"
        except Exception as e:
            print(f"❌ Milvus Connection Failed: {e}")
            raise e

    def search_and_rerank(self, query, top_k=5):
        # 1. Initial Retrieval (Vector)
        # We need embedding for the query.
        # Since we don't have ClovaX here easily, we rely on the vector being present OR
        # we assume simple Keyword Search if we can't embed?
        # WAIT. We need the Embedding Model to search Milvus.
        # If running on Colab, we can use BGE-M3 for embedding if the DB uses BGE-M3.
        # Our DB uses BGE-M3 (based on 'audit_rag_hybrid_v1' ingestion).
        # So we load BGE-M3 as well.
        pass


# RE-THINK: The DB uses BGE-M3. We need BGE-M3 to embed the query.
from langchain_community.embeddings import HuggingFaceEmbeddings


class StandalonePipeline:
    def __init__(self):
        print(f"⚙️ Initializing Models on {Config.DEVICE}...")

        # 1. Embedding (BGE-M3)
        self.embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": Config.DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 2. Reranker
        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3", max_length=512, device=Config.DEVICE
        )

        # 3. Milvus
        self.client = MilvusClient(uri=Config.MILVUS_URI, token=Config.MILVUS_TOKEN)
        self.collection_name = "audit_rag_hybrid_v1"

    def search(self, query, top_k=5):
        # Embed
        q_vec = self.embedding.embed_query(query)

        # Search (Fetch 50)
        res = self.client.search(
            collection_name=self.collection_name,
            data=[q_vec],
            limit=50,
            output_fields=["text", "parent_text", "doc_id"],
        )

        results = res[0]

        # Deduplicate Parents
        seen_parents = set()
        candidates = []  # (text, doc_obj)

        for hit in results:
            entity = hit["entity"]
            parent_text = entity.get("parent_text") or entity.get("text")

            h = hash(parent_text)
            if h not in seen_parents:
                seen_parents.add(h)
                candidates.append(parent_text)

        # Rerank
        if candidates:
            pairs = [[query, doc_text] for doc_text in candidates]
            scores = self.reranker.predict(pairs)

            scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, score in scored[:top_k]]
        else:
            final_docs = []

        return final_docs


# --- EVALUATION LOGIC ---
def convert_matches_to_score(match_dict):
    count = 0
    for val in match_dict.values():
        if isinstance(val, dict):
            if val.get("decision") is True:
                count += 1
        elif val is True:
            count += 1
    if count >= 4:
        return 1.0
    elif count >= 2:
        return 0.6
    elif count == 1:
        return 0.2
    else:
        return 0.0


def evaluate_single(query, pipeline, eval_chain):
    if not query:
        return 0.0, [], "Empty Query"

    try:
        docs = pipeline.search(query)
    except Exception as e:
        return 0.0, [], f"Search Error: {str(e)}"

    scores = []
    reasons = []

    # Evaluate Top 1 or Top 3? Let's eval Top 3 to be safe, or just all docs returned?
    # Usually we eval the top 3-5. efficient eval.
    for i, d in enumerate(docs[:3]):
        try:
            res = eval_chain.invoke({"question": query, "document": d})
            content = res.content.replace("```json", "").replace("```", "").strip()
            # Try to find JSON if there's extra text
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != -1:
                content = content[start:end]

            s_dict = json.loads(content)
            score = convert_matches_to_score(s_dict)
            scores.append(score)
            reasons.append(s_dict)  # Store the full dict for debugging
        except Exception as e:
            scores.append(0.0)
            reasons.append(
                {
                    "error": str(e),
                    "raw_content": res.content if "res" in locals() else "No Response",
                }
            )

    final_score = mean(scores) if scores else 0.0
    return final_score, [d[:200] for d in docs], reasons


def main(csv_path="retrieval.csv"):
    pipeline = StandalonePipeline()

    # Judge
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=Config.OPENAI_API_KEY)

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
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "[Question]\n{question}\n\n[Document]\n{document}"),
        ]
    )
    chain = prompt | llm

    # Data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} queries.")

    # Run
    results = []
    detailed_logs = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        q = row.get("question") or row.get("query")
        # Pass evaluate_chain explicitly
        score, retrieved_snippets, debug_reasons = evaluate_single(q, pipeline, chain)

        results.append(score)

        detailed_logs.append(
            {
                "question": q,
                "score": score,
                "retrieved_docs_snippets": str(retrieved_snippets),
                "debug_reasons": json.dumps(debug_reasons, ensure_ascii=False),
            }
        )

    mean_score = mean(results)
    print(f"Mean Score: {mean_score:.4f}")

    # Save CSV
    output_path = "ablation_results_standalone.csv"
    pd.DataFrame(detailed_logs).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Detailed results saved to {output_path}")


if __name__ == "__main__":
    # Upload 'evaluate_hybrid_results.csv' to Colab and rename/point to it in main()
    # Or pass args
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python run_ablation_standalone.py <csv_path>")
