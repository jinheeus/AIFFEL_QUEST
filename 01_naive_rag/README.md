# 01_naive_rag: Metadata Filtered RAG (실험군)

이 디렉토리는 **메타데이터 필터링(Metadata Filtering)**과 **라우터 에이전트(Router Agent)**를 적용한 RAG 시스템 구현체입니다.

## 목적 (Purpose)
베이스라인(V0) 대비 성능 향상을 검증하기 위한 실험군(Experimental Group)입니다.
- **라우터 에이전트 (Router Agent)**: 사용자의 질문을 분석하여 5가지 감사 카테고리 중 하나로 분류합니다.
- **메타데이터 필터링 (Metadata Filtering)**: 분류된 카테고리에 해당하는 문서 내에서만 검색을 수행하여 정확도를 높입니다.
- **엄격한 프롬프트**: 베이스라인과 동일한 생성 프롬프트를 사용하여 공정한 비교를 수행합니다.

## 구성 요소 (Components)
- `ingest.py`: 데이터를 적재할 때 `cat_L1`(대분류), `cat_L2`(중분류) 메타데이터를 함께 추출하여 Milvus 컬렉션 `audit_reports_v1`에 저장합니다.
- `pipeline.py`:
    1. **Router**: 질문을 카테고리로 분류합니다.
    2. **Retriever**: 해당 카테고리 필터(`expr="cat_L1 == '...'"`)를 적용하여 검색합니다.
    3. **Generator**: 검색된 문서를 바탕으로 답변을 생성합니다.
- `playground.ipynb`: 라우팅 및 전체 파이프라인을 실험할 수 있는 노트북입니다.

## 사용 방법 (Usage)

1. **의존성 설치**:
   ```bash
   pip install -r requirements.txt
   ```

2. **환경 설정**:
   `.env` 파일 확인 (`CLOVANSTUDIO_API_KEY`, `MILVUS_URI` 등).

3. **데이터 적재 (Ingest)**:
   ```bash
   python ingest.py
   ```

4. **쿼리 실행 (Run Query)**:
   - **CLI 실행**:
     ```bash
     python pipeline.py --query "직원 횡령 시 처분 기준은?"
     ```
   - **노트북 실행**:
     `playground.ipynb`에서 라우터 동작과 답변 생성을 테스트해보세요.
