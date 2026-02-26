import os
import json
import time
from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain_core.documents import Document
from utils.embedding import embeddings_hcx

load_dotenv()

def direct_ingest():
    json_path = "./data/contents.json"
    print(f"[ INFO ] {json_path} 분석 및 삽입 시작...")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = []
    for item in data:
        # 선배님들이 만든 모든 필드를 하나의 검색 텍스트로 합칩니다.
        # title, standards, outline, problems, opinion, criteria, action 순서
        fields = ["title", "standards", "outline", "problems", "opinion", "criteria", "action"]
        combined_text = "\n".join([f"[{f.upper()}]: {item.get(f, '')}" for f in fields if item.get(f)])
        
        idx = item.get('idx') or 0
        if combined_text.strip():
            docs.append(Document(page_content=combined_text, metadata={"idx": idx}))

    # Milvus 연결
    vector_db = Milvus(
        embedding_function=embeddings_hcx,
        connection_args={"uri": os.getenv("MILVUS_URI")},
        collection_name="documents",
        auto_id=True
    )

    # 배치 삽입
    batch_size = 20 # 속도 제한을 위해 더 안전하게 20개씩
    total_docs = len(docs)
    print(f"[ INFO ] 총 {total_docs}개의 정제된 문서를 삽입합니다...")

    for i in range(0, total_docs, batch_size):
        batch = docs[i : i + batch_size]
        try:
            vector_db.add_documents(batch)
            print(f"[ PROGRESS ] {min(i + batch_size, total_docs)} / {total_docs} 완료")
            time.sleep(2) # Rate Limit 방지를 위해 2초씩 휴식
        except Exception as e:
            print(f"[ ERROR ] {i}번 배치 오류: {e}. 5초 후 재시도...")
            time.sleep(5)
            vector_db.add_documents(batch)

    print("[ SUCCESS ] 데이터 삽입 완료! 이제 main.py를 실행하세요.")

if __name__ == "__main__":
    direct_ingest()