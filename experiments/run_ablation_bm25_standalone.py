import os
import sys
import json
import pandas as pd
from tqdm import tqdm
from statistics import mean
import concurrent.futures

# --- INSTALL COMMANDS (Run these in Colab first) ---
# !pip install -q pymilvus langchain langchain-openai langchain-community sentence-transformers rank_bm25 kiwipiepy tqdm pandas


# --- CONFIGURATION ---
class Config:
    # UPDATE THESE WITH YOUR CREDENTIALS
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
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from rank_bm25 import BM25Okapi
    from kiwipiepy import Kiwi
    from langchain_core.documents import Document
except ImportError:
    print(
        "❌ Missing dependencies. Please run:\npip install pymilvus langchain langchain-openai langchain-community sentence-transformers rank_bm25 kiwipiepy tqdm pandas"
    )
    sys.exit(1)


# --- HELPER: Load All Docs for BM25 ---
def load_all_docs(client, collection_name, batch_size=500):
    print(f"📥 Loading corpus from Milvus collection: {collection_name}...")

    # 1. Get total count (approximate or just loop until empty)
    # We'll use offset pagination logic
    all_docs = []
    offset = 0

    while True:
        res = client.query(
            collection_name=collection_name,
            filter="id >= 0",  # Match all
            output_fields=["text", "parent_text", "doc_id"],
            limit=batch_size,
            offset=offset,
        )

        if not res:
            break

        all_docs.extend(res)
        offset += len(res)
        print(f" -> Fetched {len(res)} / Total {len(all_docs)}", end="\r")

        if len(res) < batch_size:
            break

    print(f"\n✅ Loaded {len(all_docs)} total documents.")
    return all_docs


# --- PIPELINE CLASS (Hybrid) ---
class HybridPipeline:
    def __init__(self):
        print(f"⚙️ Initializing Hybrid Pipeline on {Config.DEVICE}...")

        # 1. Embedding (BGE-M3)
        print("   -> Loading Embedding Model (BGE-M3)...")
        self.embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": Config.DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 2. Reranker (BGE-Reranker-v2-m3)
        print("   -> Loading Reranker (BGE-Reranker-v2-m3)...")
        try:
            self.reranker = CrossEncoder(
                "BAAI/bge-reranker-v2-m3",
                max_length=512,
                device=Config.DEVICE,
                automodel_args={"torch_dtype": "auto"},
            )
        except Exception:
            print("   ⚠️ Failed to load BGE-Reranker, fallback to MiniLM.")
            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", device=Config.DEVICE
            )

        # 3. Milvus
        print("   -> Connecting to Milvus...")
        self.client = MilvusClient(uri=Config.MILVUS_URI, token=Config.MILVUS_TOKEN)
        self.collection_name = "audit_rag_hybrid_v1"

        # 4. BM25 Index
        print("   -> Building BM25 Index (This may take a minute)...")
        self.tokenizer = Kiwi()
        raw_docs = load_all_docs(self.client, self.collection_name, batch_size=200)

        self.bm25_corpus = []  # Tokenized Clean Text
        self.bm25_docs = []  # Document Objects (1:1 with corpus)
        self.id_to_doc = {}  # Quick Lookup Hash -> Document

        for d in raw_docs:
            text = d.get("text", "")
            parent = d.get("parent_text", "")

            # Tokenize
            tokens = [t.form for t in self.tokenizer.tokenize(text)]
            self.bm25_corpus.append(tokens)

            # Store
            doc_obj = Document(page_content=text, metadata={"parent_text": parent})
            self.bm25_docs.append(doc_obj)
            self.id_to_doc[text] = doc_obj

        self.bm25 = BM25Okapi(self.bm25_corpus)
        print("✅ BM25 Ready")

    def search(self, query, top_k=5):
        # 1. Dense Search
        q_vec = self.embedding.embed_query(query)
        dense_res = self.client.search(
            collection_name=self.collection_name,
            data=[q_vec],
            limit=50,
            output_fields=["text", "parent_text"],
        )
        dense_results = []
        for hit in dense_res[0]:
            entity = hit["entity"]
            d = Document(
                page_content=entity.get("text"),
                metadata={"parent_text": entity.get("parent_text")},
            )
            dense_results.append(d)

        # 2. Sparse Search (BM25)
        q_tokens = [t.form for t in self.tokenizer.tokenize(query)]
        sparse_results = self.bm25.get_top_n(q_tokens, self.bm25_docs, n=50)

        # 3. RRF Fusion
        dense_ranks = {doc.page_content: i for i, doc in enumerate(dense_results)}
        sparse_ranks = {doc.page_content: i for i, doc in enumerate(sparse_results)}

        all_content = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        fused_scores = []
        k_rrf = 60

        for content in all_content:
            rank_d = dense_ranks.get(content, float("inf"))
            rank_s = sparse_ranks.get(content, float("inf"))
            score = 0.0
            if rank_d != float("inf"):
                score += 1.0 / (k_rrf + rank_d)
            if rank_s != float("inf"):
                score += 1.0 / (k_rrf + rank_s)
            fused_scores.append((content, score))

        fused_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = fused_scores[:50]

        # 4. Prepare for Rerank (Parent Mapping)
        seen_parents = set()
        candidates = []

        for content, _ in top_candidates:
            # We assume content is unique enough to lookup metadata
            if content in self.id_to_doc:
                doc_obj = self.id_to_doc[content]
                parent = doc_obj.metadata.get("parent_text") or content

                h = hash(parent)
                if h not in seen_parents:
                    seen_parents.add(h)
                    candidates.append(parent)

        # 5. Rerank
        if candidates:
            pairs = [
                [query, doc_text] for doc_text in candidates[:30]
            ]  # Limit rerank to top 30 unique parents
            scores_list = self.reranker.predict(pairs)
            scored_candidates = sorted(
                zip(candidates[:30], scores_list), key=lambda x: x[1], reverse=True
            )
            final_docs = [doc for doc, score in scored_candidates[:top_k]]
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

    # Evaluate Top 3 for consistency
    for d in docs[:3]:
        try:
            res = eval_chain.invoke({"question": query, "document": d})
            content = res.content.replace("```json", "").replace("```", "").strip()
            # Try to extract JSON if dirty
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != -1:
                content = content[start:end]

            s_dict = json.loads(content)
            scores.append(convert_matches_to_score(s_dict))
            reasons.append(s_dict)
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
    pipeline = HybridPipeline()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=Config.OPENAI_API_KEY)

    # NOTE: User's updated system prompt
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

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} queries.")

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

    print(f"Mean Score: {mean(results):.4f}")

    # Save CSV
    output_path = "ablation_results_bm25_standalone.csv"
    pd.DataFrame(detailed_logs).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Detailed results saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python run_ablation_bm25_standalone.py <csv_path>")
