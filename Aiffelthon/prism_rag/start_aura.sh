#!/bin/bash

echo "--- 1. 인프라 실행 (Milvus) ---"
docker compose up -d

echo "--- 2. 백엔드 실행 (FastAPI) ---"
cd web_app/backend
# 가상환경 이름이 다르면 수정하세요 (예: .venv)
# source venv/bin/activate
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
cd ../..

echo "--- 3. 프론트엔드 실행 (Next.js) ---"
cd web_app/frontend
nohup npm run dev > frontend.log 2>&1 &

echo "--- 모든 서버 구동 시작! ---"
echo "프론트엔드: http://localhost:3000"
echo "백엔드 API: http://localhost:8000/docs"
