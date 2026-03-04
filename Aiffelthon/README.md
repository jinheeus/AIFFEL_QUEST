# 📊 PRISM 감사 리스크 인텔리전스 대시보드
> **공공기관 감사 결과 및 리스크 데이터를 시각화하고, AI 기반 검색 및 보고서 자동 생성을 지원하는 통합 인텔리전스 대시보드입니다.**

본 프로젝트는 [Streamlit](https://streamlit.io/)을 기반으로 개발되었으며, 다차원적인 데이터 필터링, 피어그룹 벤치마킹(Macro), 세부 처분 분석(Micro), 시계열 트렌드 분석 및 RAG(Retrieval-Augmented Generation) 기반 AI 챗봇 기능을 제공합니다.

---

## ✨ 주요 기능 (Key Features)

1. **🏠 Home (대시보드 요약)**
   - 연도별 총 지적 건수, 전년 대비 증감율(YoY), 평균 처분 강도 및 리스크 점수 KPI 제공
   - 주요 감사 지적 사례 타임라인 및 최신 감사 뉴스 피드 제공

2. **📈 감사 트렌드 분석**
   - 시계열 지적 건수 변화 (월별/분기별/연도별)
   - 위반 유형별 시즈널리티 히트맵 및 Top-N 기관 랭킹
   - TF-IDF 기반 감사 핵심 키워드 클라우드 (범용어 자동 필터링)

3. **🏢 리스크 - 기관 벤치마크 (Macro View)**
   - 선택 기관과 자동 추천된 피어그룹 간의 리스크 점수 비교 (Gauge 차트)
   - 3차원 리스크 분포도 (지적건수, 처분강도, 반복비율)
   - 5축 레이더 차트를 활용한 기관별 재정처분 프로파일링

4. **🔍 리스크 - 처분 분석 (Micro View)**
   - 위반 유형 × 처분 수위 교차 히트맵 및 세부 그룹 드릴다운
   - Sankey 다이어그램을 활용한 비위 유형 → 처분 결과 흐름 시각화
   - 재정적 처분 분석 (Treemap, Bubble chart, 금액 구간별 Bar chart)

5. **🤖 감사 정보 AI 검색 및 최신 뉴스**
   - RAG 기반 AI 챗봇 인터페이스 (질의응답 및 참고 문서 출처 제공)
   - 챗봇 대화 맥락을 기반으로 한 **Audit Report(감사 보고서) 자동 작성 및 편집** 패널
   - 블랙엣지뉴스(BlackEdge News) 기반 리스크 카테고리별 뉴스 피드 (현재 기획 가안 적용)

---

## 🛠 시스템 요구사항 (Prerequisites)

- **OS**: Windows, macOS, Linux
- **Python**: 3.9 이상 권장
- **Backend API**: AI 검색 및 보고서 생성을 위해 로컬 환경(`http://localhost:8000`)에 LLM/RAG 백엔드 서버(FastAPI 등) 구동 필요

---

## 🚀 설치 및 실행 방법 (Installation & Usage)

### 1. 레포지토리 클론 및 폴더 이동
### 2. 가상환경 생성 및 활성화 (권장)
    
    # Windows
    python -m venv venv
    venv\Scripts\activate
    
    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate

### 3. 필수 패키지 설치
requirements.txt 파일이 있는 경로에서 아래 명령어를 실행합니다.

    pip install -r requirements.txt


### 4. 필수 데이터 및 리소스 배치
    - 애플리케이션이 정상적으로 작동하려면 루트 디렉토리(app_final.py와 같은 위치)에 다음 파일들이 존재해야 합니다.
    - audit_v10.json (또는 audit_v9.json 등 최신 감사 데이터 세트)
    - PRISM_logo.png (사이드바 상단 로고 이미지)

### 5. 애플리케이션 실행
    streamlit run app_final.py

실행 후 브라우저가 자동 실행되며 http://localhost:8501에서 대시보드를 확인할 수 있습니다.

---

## 📁 디렉토리 구조 (Directory Structure)

      AiffelRepository/
      ├── app_final.py                 # Streamlit 메인 애플리케이션 코드
      ├── requirements.txt             # 파이썬 의존성 패키지 목록
      ├── README.md                    # 프로젝트 설명서
      ├── PRISM_logo.png               # 사이드바 로고 이미지 (필요 시 추가)
      ├── data/                        # (권장) 데이터 폴더
      │   ├── audit_v10.json           # 메인 감사 지적 사례 데이터셋
      │   └── matched_keywords_only.json # 분야별 핵심 키워드 매핑 데이터
      └── .streamlit/
          └── config.toml              # (선택) Streamlit 테마 설정 파일


---

## 🔌 백엔드 API 연동 규격 (Backend API Specification)
- AI 검색 및 보고서 작성 탭이 정상 작동하기 위해서는 포트 8000번에서 다음 API 엔드포인트를 제공하는 백엔드 서버가 필요합니다.

1. POST /chat
  - 기능: 사용자의 질문과 대화 기록을 받아 스트리밍 방식으로 답변, 사고 과정(Thoughts), 참고 문서(References) 반환
  - Payload: {"query": "string", "history": [...], "session_id": "string"}

2. POST /check_report_readiness
  - 기능: 현재 대화 내역을 분석하여 보고서 작성에 필수적인 항목(사건 제목, 대상 기관 등)이 모두 수집되었는지 확인
  - Return: {"status": "missing_info", "missing_fields": ["대상 기관", "문제점"]} 또는 {"status": "ready"}

3. POST /generate_report
  - 기능: 수집된 정보와 대화 맥락, 대시보드의 필터 상태를 종합하여 마크다운 형식의 최종 감사 보고서 초안 생성.

---

## ⚠️ 데이터 스키마 요구사항 (Data Schema)
메인 데이터인 audit_v10.json (List of Dictionaries 형식)은 최소한 다음의 키(Key)를 포함해야 정상적으로 시각화됩니다.

- date: 처분/감사 일자 (Epoch ms 또는 YYYY-MM-DD 문자열)
- title / summary_title: 보고서 또는 사건 제목
- agency_category: 대상 기관명 (구 sub_category, org_name)
- org_category: 기관의 유형 (예: 공기업, 준정부기관 등)
- audit_report_type: 감사 종류 (예: 종합감사, 특정감사)
- risk_category: 위반 분야 (예: 재무/회계/계약, 인사/채용/복무 등)
- disposition_level: 처분 수위 (예: 중징계, 경징계, 시정 등)
- penalty_id / penalty_category / penalty_type / penalty_amount: 재정적 처분 종류 및 금액 (단위: 원)
- doc_code / download_url: 원문 문서 번호 및 다운로드 링크

---

## 📌 향후 개발 계획 (TODO & Future Works)
- 로그인 및 SSO 연동: 기획 가안으로 구현된 로그인 모달(Home 우측 상단)을 정부/공공기관 SSO와 연동하여 소속 기관에 맞는 개인화 대시보드 제공.
- 실시간 뉴스 크롤링: 하드코딩된 '블랙엣지뉴스' 피드를 RSS 연동 또는 BeautifulSoup을 활용한 실시간 크롤링으로 대체.
- 문서 다운로드 및 미리보기: download_url 외에도 대시보드 내에서 PDF 원문을 바로 렌더링하는 기능 추가.

---
- Maintainer: 모두의연구소 데이터사이언티스트 6기 감사합니다팀
- Last Updated: 2026-03-04







