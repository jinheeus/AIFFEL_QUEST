# 00_baseline: Pure Naive RAG (대조군)

이 디렉토리는 **HyperCLOVA X**와 **Milvus**를 사용한 RAG 시스템의 베이스라인 구현체입니다.

## 목적 (Purpose)
성능 비교를 위한 대조군(Control Group) 역할을 합니다. "Pure Naive" 접근 방식을 따릅니다:
- **메타데이터 필터링 없음**: 오직 벡터 유사도(Vector Similarity)에만 의존하여 검색합니다.
- **단순 청킹 (Simple Chunking)**: `problem`(문제점)과 `contents`(상세내용) 필드를 단순 병합하여 청킹합니다.
  - **전략**: `RecursiveCharacterTextSplitter` 사용 (Chunk Size: 800, Overlap: 100).
- **엄격한 프롬프트 (Strict Prompting)**: 환각(Hallucination)을 방지하고 문서 내용에만 충실하도록 설계된 시스템 프롬프트를 사용합니다.

## 구성 요소 (Components)
- `ingest.py`: `data_v10.json`을 로드하고, 텍스트를 청킹한 후, `bge-m3` (ClovaXEmbeddings)로 임베딩하여 Milvus 컬렉션 `audit_reports_v0`에 적재합니다.
- `pipeline.py`: 상위 5개 문서(Top-5)를 검색하고, `HCX-DASH-002`를 사용하여 답변을 생성합니다.
- `../config.py`: (Root 위치) API 키 및 모델 설정 파일입니다. 모든 파이프라인이 이 공통 설정을 참조합니다.
- `playground.ipynb`: 파이프라인을 대화형으로 테스트하고 실험할 수 있는 주피터 노트북입니다.

## 사용 방법 (Usage)

1. **의존성 설치**:
   ```bash
   pip install -r requirements.txt
   ```

2. **환경 설정**:
   `.env` 파일(또는 환경 변수)에 다음 항목이 설정되어 있어야 합니다:
   - `CLOVANSTUDIO_API_KEY`
   - `MILVUS_URI`

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
     `playground.ipynb`를 열어서 대화형으로 실행해볼 수 있습니다.
