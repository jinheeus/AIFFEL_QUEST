import os
import json
import time
import ast
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Zilliz

def load_env_variables():
    """
    .env 파일에서 Zilliz 연결 정보를 로드합니다.
    """
    load_dotenv()
    zilliz_uri = os.getenv("ZILLIZ_CLOUD_URI")
    zilliz_token = os.getenv("ZILLIZ_CLOUD_TOKEN")

    if not zilliz_uri:
        raise ValueError("환경 변수 'ZILLIZ_CLOUD_URI'가 .env 파일에 설정되지 않았습니다.")
    if not zilliz_token:
        raise ValueError("환경 변수 'ZILLIZ_CLOUD_TOKEN'이 .env 파일에 설정되지 않았습니다.")
        
    print("✅ Zilliz 연결 정보를 성공적으로 로드했습니다.")
    return zilliz_uri, zilliz_token

def load_and_prepare_docs(filepath="audit_cases.json"):
    """
    [Baseline] 'contents_summary' 필드만을 사용하여 문서를 생성합니다.
    """
    print(f"📄 '{filepath}' 파일에서 'contents_summary' 기반 문서를 생성합니다...")
    with open(filepath, 'r', encoding='utf-8') as f:
        audit_cases = json.load(f)

    docs = []
    for i, case in enumerate(audit_cases):
        site = case.get('site', '알 수 없음')
        category = case.get('category', '알 수 없음')
        date = case.get('date', '알 수 없음')
        original_title = case.get('title', '')
        
        metadata = {"index": i, "title": original_title, "site": site, "category": category, "date": date}

        summary_dict = {}
        summary_str = case.get('contents_summary')
        if summary_str:
            try:
                summary_dict = ast.literal_eval(summary_str)
            except (ValueError, SyntaxError):
                summary_dict = {}
        
        title = summary_dict.get('title_str', original_title)
        keywords = ", ".join(summary_dict.get('keyword_list', []))
        problems = summary_dict.get('problems_str', '')
        action = summary_dict.get('action_str', '')
        standards = summary_dict.get('standards_str', '')

        summary_based_text = (
            f"출처: {site}\n"
            f"분류: {category}\n"
            f"일자: {date}\n"
            f"제목: {title}\n"
            f"핵심 키워드: {keywords}\n"
            f"문제 요약: {problems}\n"
            f"조치 요약: {action}\n"
            f"관련 규정: {standards}"
        )
        docs.append(Document(page_content=summary_based_text, metadata=metadata))

    print(f"  - 총 {len(docs)}개의 요약 기반 문서를 준비했습니다.")
    return docs

def main():
    """
    데이터를 로드하고 Ollama로 임베딩하여 Zilliz Cloud에 업로드하는 메인 함수
    """
    COLLECTION_NAME = "audit_cases_gemma_v1"
    try:
        zilliz_uri, zilliz_token = load_env_variables()
    except ValueError as e:
        print(f"🚨 에러: {e}")
        return

    # 1. 데이터 로드
    documents = load_and_prepare_docs()
    if not documents:
        return

    # 2. 임베딩 모델 초기화 (Ollama)
    print("\n🧠 Ollama 임베딩 모델을 초기화합니다 (nomic-embed-text)...")
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    except Exception as e:
        print(f"🚨 임베딩 모델 초기화 중 에러가 발생했습니다: {e}")
        print("   'ollama'가 실행 중인지, 'nomic-embed-text' 모델이 pull 되었는지 확인해주세요.")
        return

    # 3. Zilliz에 데이터 업로드 (배치 처리)
    print(f"\n☁️ Zilliz Cloud에 데이터를 업로드합니다 (Collection: '{COLLECTION_NAME}')...")
    
    batch_size = 128
    total_batches = (len(documents) -1) // batch_size + 1

    try:
        # 첫 번째 배치로 벡터 저장소 생성
        print(f"  - [1/{total_batches} 배치] 처리 중...")
        vector_store = Zilliz.from_documents(
            documents=documents[:batch_size],
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": zilliz_uri, "token": zilliz_token},
            auto_id=True,
            drop_old=True
        )
        print("  - 첫 번째 배치 완료.")
        time.sleep(1)

        # 나머지 배치 추가
        for i in range(batch_size, len(documents), batch_size):
            batch_num = (i // batch_size) + 1
            batch_docs = documents[i:i+batch_size]
            
            print(f"  - [{batch_num}/{total_batches} 배치] 처리 중...")
            vector_store.add_documents(batch_docs)
            print(f"  - {batch_num}번째 배치 완료.")
            time.sleep(1)
        
        print("\n✨ 모든 문서의 임베딩 및 Zilliz Cloud 업로드를 완료했습니다!")

    except Exception as e:
        print(f"\n🚨 Zilliz 업로드 중 에러가 발생했습니다: {e}")
        print("   - Zilliz Cloud URI와 Token이 올바른지 확인해주세요.")
        print("   - 'pymilvus', 'langchain-community' 패키지가 설치되었는지 확인해주세요.")

if __name__ == "__main__":
    main()
