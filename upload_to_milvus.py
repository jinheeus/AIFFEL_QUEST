"""
audit_v10.json -> Milvus 업로드 스크립트
실행: python upload_to_milvus.py
"""

import sys
import os
import json
import time

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from common.config import Config
from langchain_naver import ClovaXEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from pymilvus import MilvusClient

# ── 설정 ──────────────────────────────────────────────
DATA_PATH = os.path.join(project_root, "audit_v10.json")
COLLECTION_NAME = "audit_v10_collection"
CHUNK_SIZE = 500
BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 2
RESUME_FROM = 0          # 처음부터 시작
# ─────────────────────────────────────────────────────


def build_parent_text(item: dict) -> str:
    """
    v10 구조에 맞게 parent_text 생성
    - contents_summary 안에 outline, problems, opinion, criteria, action이 있음
    - contents, problem, action은 최상위 필드
    """
    summary = item.get("contents_summary") or {}
    
    # contents_summary가 dict인 경우 (감사원 데이터)
    if isinstance(summary, dict):
        outline   = summary.get("outline", "")
        problems  = summary.get("problems", "")
        opinion   = summary.get("opinion", "")
        criteria  = summary.get("criteria", "")
        action    = summary.get("action", "")
    else:
        outline = problems = opinion = criteria = action = ""

    # 최상위 필드 fallback
    if not problems:
        problems = item.get("problem", "")
    if not action:
        action = item.get("action", "")
    if not outline:
        outline = item.get("contents", "")

    parts = [
        f"[Title]: {item.get('title', '')}",
        f"[Outline]: {outline}",
        f"[Problems]: {problems}",
        f"[Opinion]: {opinion}",
        f"[Criteria]: {criteria}",
        f"[Action]: {action}",
    ]
    return "\n".join([p for p in parts if p.split(": ", 1)[1].strip()])


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    if len(text) <= chunk_size:
        return [text]
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def main():
    print("=" * 50)
    print("📦 Milvus 업로드 시작 (audit_v10)")
    print("=" * 50)

    # 1. 데이터 로드
    print(f"\n1️⃣  데이터 로드: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   총 {len(data)}개 항목 로드 완료")

    # 2. 임베딩 모델 초기화
    print(f"\n2️⃣  임베딩 모델 초기화: {Config.EMBEDDING_MODEL}")
    embedding_model = ClovaXEmbeddings(model=Config.EMBEDDING_MODEL)

    # 3. 컬렉션 확인
    print(f"\n3️⃣  기존 컬렉션 확인: {COLLECTION_NAME}")
    client = MilvusClient(uri=Config.MILVUS_URI, token=Config.MILVUS_TOKEN)
    existing = client.list_collections()
    if COLLECTION_NAME in existing:
        if RESUME_FROM > 0:
            print(f"   ▶️  이어서 업로드 (RESUME_FROM={RESUME_FROM})")
        else:
            print(f"   ⚠️  기존 컬렉션 발견 → 삭제 후 재생성")
            client.drop_collection(COLLECTION_NAME)
    else:
        print(f"   ✅ 기존 컬렉션 없음 → 새로 생성")

    # 4. Document 생성
    print(f"\n4️⃣  문서 청킹 및 메타데이터 구성 중...")
    documents = []
    for item in data:
        parent_text = build_parent_text(item)
        if not parent_text.strip():
            continue

        for chunk in chunk_text(parent_text):
            doc = Document(
                page_content=chunk,
                metadata={
                    "doc_text": chunk,
                    "parent_text": parent_text,
                    "source_type": "audit",
                    "source": "audit_v10.json",
                    "idx": str(item.get("idx", "")),
                    "site": item.get("site", ""),
                    "date": item.get("date") or "1900.01.01",
                    "title": item.get("title", ""),
                    "outline": (item.get("contents_summary") or {}).get("outline", "") if isinstance(item.get("contents_summary"), dict) else "",
                    "category": item.get("category", ""),
                    "cat": item.get("cat") or "",
                    "sub_cat": item.get("sub_cat") or "",
                    "download_url": item.get("download_url", ""),
                    "file_path": item.get("file_path", ""),
                    "risk_category": item.get("risk_category", ""),
                    "disposition_level": str(item.get("disposition_level", "")),
                },
            )
            documents.append(doc)

    print(f"   총 {len(documents)}개 청크 생성 완료")
    if RESUME_FROM > 0:
        print(f"   ⏩ {RESUME_FROM}개 건너뛰고 이어서 시작")

    # 5. Milvus 업로드 (배치)
    print(f"\n5️⃣  Milvus 업로드 중 (배치: {BATCH_SIZE}개, 딜레이: {SLEEP_BETWEEN_BATCHES}s)...")
    start = time.time()
    vector_store = None

    for i in range(0, len(documents), BATCH_SIZE):
        if i < RESUME_FROM:
            continue

        batch = documents[i:i + BATCH_SIZE]

        if vector_store is None:
            vector_store = Milvus.from_documents(
                documents=batch,
                embedding=embedding_model,
                connection_args={
                    "uri": Config.MILVUS_URI,
                    "token": Config.MILVUS_TOKEN,
                },
                collection_name=COLLECTION_NAME,
                drop_old=False,
            )
        else:
            vector_store.add_documents(batch)

        elapsed = time.time() - start
        print(f"   [{i + len(batch)}/{len(documents)}] 업로드 완료 ({elapsed:.1f}s)")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"\n✅ 업로드 완료! 총 소요시간: {time.time() - start:.1f}s")

    # 6. 검증
    print(f"\n6️⃣  업로드 검증...")
    final_count = client.get_collection_stats(COLLECTION_NAME)
    print(f"   컬렉션: {COLLECTION_NAME}")
    print(f"   row_count: {final_count.get('row_count', '확인불가')}")
    print("\n🎉 완료! 이제 bm25_cache.pkl 삭제 후 서버를 재시작하세요.")
    print(f"   rm {os.path.join(project_root, 'rag/agentic_rag_v2/modules/bm25_cache.pkl')}")


if __name__ == "__main__":
    main()
