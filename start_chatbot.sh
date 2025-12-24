#!/bin/bash

# Function to kill processes on exit
cleanup() {
    echo "🛑 Shutting down..."
    kill $(jobs -p) 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

echo "🚀 Starting Agentic RAG Chatbot System..."

# 1. Start Backend
echo "🔹 [Backend] Starting FastAPI Server on port 8000..."
export PYTHONUNBUFFERED=1
nohup python web_app/backend/main.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "   -> Backend running (PID: $BACKEND_PID). Logs at backend.log"

# 2. Start Frontend
echo "🔹 [Frontend] Starting Next.js App on port 3000..."
cd web_app/frontend
npm run dev > ../../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   -> Frontend running (PID: $FRONTEND_PID). Logs at frontend.log"

echo "✅ System is UP!"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend:  http://localhost:8000"
echo "   (Press Ctrl+C to stop all)"

wait
