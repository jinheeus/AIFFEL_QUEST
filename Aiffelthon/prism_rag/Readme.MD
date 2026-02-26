# 🔮 PRISM - RAG 챗봇 연결 가이드

> **PRISM** (Public Risk Intelligence & Smart Management)  
> 공공 감사 데이터를 다차원으로 분석하는 AI 감사 분석 시스템

---

## 📁 깃허브에 올려야 하는 파일 목록

```
prism_rag/
│
├── README.MD                                   ✅ 이 파일
├── requirements.txt                            ✅ 패키지 목록
├── upload_to_milvus.py                         ✅ Milvus 업로드 스크립트
├── docker-compose.yml                          ✅ Milvus Docker 설정
│
├── common/
│   ├── config.py                               ✅ 설정 파일
│   ├── model_factory.py                        ✅ 모델 초기화
│   ├── logger_config.py                        ✅ 로거 설정
│   └── memory/
│       └── redis_checkpointer.py               ✅ Redis 체크포인터
│
├── rag/
│   └── agentic_rag_v2/
│       ├── graph.py                            ✅ RAG 그래프
│       ├── state.py                            ✅ 상태 정의
│       └── modules/
│           ├── vector_retriever.py             ✅ Milvus + BM25 검색
│           ├── generator.py                    ✅ 답변 생성 (PRISM 프롬프트)
│           ├── retriever.py                    ✅ 검색 래퍼
│           ├── grader.py                       ✅ 문서 평가
│           ├── rewriter.py                     ✅ 쿼리 재작성
│           ├── field_selector.py               ✅ 필드 선택
│           ├── sop_retriever.py                ✅ SOP 검색
│           ├── memory.py                       ✅ 대화 요약
│           ├── sql_retriever.py                ✅ SQL 검색
│           ├── drafting_agent.py               ✅ 보고서 작성 에이전트
│           └── chat_worker.py                  ✅ 일상 대화 처리
│
└── web_app/
    ├── backend/
    │   └── main.py                             ✅ FastAPI 백엔드
    └── frontend/
        └── app.py                              ✅ Streamlit 대시보드
```

---

## 🚫 깃허브에 올리면 안 되는 파일

```
├── audit_v10.json              ❌ 용량 큼 → 별도 공유 (구글드라이브 등)
├── .env                        ❌ API 키 포함 → 절대 올리면 안 됨
├── venv_aura/                  ❌ 가상환경 폴더
├── volumes/                    ❌ Milvus Docker 데이터
├── __pycache__/                ❌ 파이썬 캐시
└── bm25_cache.pkl              ❌ BM25 캐시 (자동 생성됨)
```

---

## 🐙 깃허브 업로드 방법 (리더가 할 내용)

### Step 1. 팀원 초대

1. [github.com/jinheeus/AIFFEL_QUEST](https://github.com/jinheeus/AIFFEL_QUEST) 접속
2. **Settings** → **Collaborators** 클릭
3. **Add people** 클릭
4. 팀원 깃허브 유저네임 입력 후 초대
5. 팀원은 이메일 또는 깃허브 알림에서 초대 수락

---

### Step 2. .gitignore 만들기

```bash
cd ~/projects/prism_rag

cat > .gitignore << 'EOF'
venv_aura/
.env
audit_v10.json
volumes/
__pycache__/
*.pyc
bm25_cache.pkl
*.log
nohup.out
web_app/frontend/node_modules/
web_app/frontend/.next/
.vscode/
EOF
```

---

### Step 3. 깃허브 계정 설정

```bash
git config --global user.email "wltn021212@gmail.com"
git config --global user.name "JISU"
```

> 깃허브 이메일/유저네임 확인: github.com → 프로필 → Settings → Emails

---

### Step 4. 파일 추가 및 커밋

```bash
git add .

# audit_v10.json 없는지 꼭 확인!
git status

git commit -m "feat: PRISM RAG 챗봇 세팅"
```

---

### Step 5. 깃허브에 업로드

```bash
git push origin main
```

---

### Step 6. 데이터셋 별도 공유

`audit_v10.json`은 용량이 크므로 **구글드라이브**로 팀원에게 별도 공유하세요.

---

## 👥 팀원 연결 방법 (팀원이 따라할 내용)

### Step 1. 초대 수락

- 깃허브 이메일에서 `Accept invitation` 클릭
- 또는 `https://github.com/jinheeus/AIFFEL_QUEST/invitations` 접속해서 수락

### Step 2. 레포 클론

```bash
git clone https://github.com/jinheeus/AIFFEL_QUEST.git
cd AIFFEL_QUEST/Aiffelthon
```

### Step 3. 가상환경 설정

```bash
python -m venv venv_aura

# Mac/Linux
source venv_aura/bin/activate

# Windows
venv_aura\Scripts\activate

pip install -r requirements.txt
```

### Step 4. API 키 설정

팀 리더에게 API 키를 받아서 `common/config.py`에 입력:

```python
CLOVA_API_KEY = "받은 API 키 입력"
```

### Step 5. 데이터셋 받기

팀 리더에게 받은 `audit_v10.json`을 프로젝트 폴더에 넣기:
```
AIFFEL_QUEST/Aiffelthon/audit_v10.json  ← 여기에!
```

### Step 6. Docker로 Milvus 실행

```bash
docker-compose up -d
```

확인:
```bash
docker ps
# milvus-standalone, milvus-etcd, milvus-minio 3개가 Up 상태여야 해요
```

### Step 7. Milvus 데이터 업로드 (최초 1회만)

```bash
python upload_to_milvus.py
```

완료 메시지:
```
✅ 업로드 완료!
row_count: 8000+
```

### Step 8. 백엔드 서버 실행

```bash
python web_app/backend/main.py
```

정상 실행:
```
✅ BM25 Index Ready with XXXXX docs.
🔹 [Startup] VectorRetriever Ready!
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 9. Streamlit 대시보드 실행

새 터미널을 열고:
```bash
source venv_aura/bin/activate
streamlit run web_app/frontend/app.py
```

브라우저에서 `http://localhost:8501` 접속 → **Section 3. AI 분석** 에서 테스트!

---

## 🧪 연결 확인 테스트 질문

```
2024년 성실의무 위반으로 징계받은 사례 알려줘
```

정상 응답이 오면 연결 완료! ✅

---

## ❗ 자주 발생하는 오류

| 오류 | 원인 | 해결 |
|------|------|------|
| `MilvusException: Fail connecting to server` | Docker 꺼짐 | `docker-compose up -d` 재실행 |
| `[Errno 98] address already in use` | 포트 8000 충돌 | `kill $(lsof -t -i:8000)` 후 재시작 |
| `Failed to load corpus` | BM25 캐시 오류 | `rm rag/agentic_rag_v2/modules/bm25_cache.pkl` 후 재시작 |
| `⚠️ Redis connection failed` | Redis 미설치 | 무시 가능 (In-Memory로 자동 전환) |
| `audit_v10.json not found` | 데이터셋 없음 | 팀 리더에게 파일 요청 |

---

## 📌 시스템 정보

| 항목 | 값 |
|------|-----|
| 백엔드 포트 | 8000 |
| 프론트엔드 포트 | 8501 |
| Milvus 포트 | 19530 |
| 컬렉션명 | audit_v10_collection |
| 데이터셋 | audit_v10.json (5043개) |
| RAG 모듈 | agentic_rag_v2 |

---