#!/bin/bash
# Kill existing process on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null

echo "🚀 Starting Agentic RAG Web Server..."
python 04_web_app/backend/main.py > backend.log 2>&1
