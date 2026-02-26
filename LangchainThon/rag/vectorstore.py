# Chroma init/load (진희님의 기존 코드 기반)

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config import settings
from typing import Optional

def get_embeddings():
    # [확인] 팀에서 합의된 고성능 모델 사용 (Large 모델은 차원이 높아 검색이 정교함)
    return OpenAIEmbeddings(model="text-embedding-3-large")

def get_vectorstore(persist_directory: Optional[str] = None):
    """
    [변경점 설명]
    build_index.py에서 get_vectorstore() 호출 시 인자를 주지 않아도 
    settings.chroma_dir를 기본값으로 사용하도록 설계되어 연결 오류가 없습니다.
    """
    # [추가/확인] 경로가 없을 경우 settings에서 가져오는 안전장치
    persist_directory = persist_directory or settings.chroma_dir
    
    embeddings = get_embeddings()
    
    # [참고] collection_name은 나중에 다른 성격의 데이터(예: 면접 꿀팁 등)와 
    # 구분하기 위해 현재 'accepted_coverletters'로 명확히 지정됨
    return Chroma(
        collection_name="accepted_coverletters",
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )