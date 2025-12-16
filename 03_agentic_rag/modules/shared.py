import sys
import os
import importlib

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

# 02_high_context_rag.pipeline을 동적으로 임포트
high_context_rag = importlib.import_module("02_high_context_rag.pipeline")
HighContextRAGPipeline = high_context_rag.HighContextRAGPipeline

# Pipeline 인스턴스 생성 (Singleton처럼 사용)
print("Loading HighContextRAGPipeline...")
rag_pipeline = HighContextRAGPipeline()
