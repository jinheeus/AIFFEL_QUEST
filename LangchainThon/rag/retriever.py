# retriever(MMR/threshold)

from rag.vectorstore import get_vectorstore
from config import settings
from typing import Optional

def build_retriever(use_mmr: bool, top_k: int, score_threshold: Optional[float]):
    vs = get_vectorstore(settings.chroma_dir)

    search_type = "mmr" if use_mmr else "similarity"
    search_kwargs = {"k": top_k}

    # Chroma retriever는 score_threshold 지원이 버전에 따라 다를 수 있어
    # (안되면 후처리 필터로 해결)
    retriever = vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def retrieve(query: str):
        docs = retriever.invoke(query)
        if score_threshold is None:
            return docs
        # 점수 기반 필터가 필요하면, 여기에서 “문서 내용 기반 간이 필터”로 대체(7일 MVP용)
        # 실제 similarity score를 쓰려면 vectorstore.similarity_search_with_score 사용으로 확장
        return docs[:top_k]

    return retrieve

#
# rag/retriever.py

from typing import List, Optional
from rag.vectorstore import get_vectorstore
from schemas import JDStructured


def retrieve_evidence(
    query_structured_jd: JDStructured,
    user_job: Optional[str] = None,
    user_stack: Optional[str] = None,
    k: int = 4,
) -> List[str]:
    """
    합격 자소서 PDF에서 근거 문장 retrieval
    """

    # ================================
    # 기존 코드 (dict 가정)  ❌ 오류 원인
    # ================================
    # query = " ".join([
    #     query_structured_jd.get("role_summary", ""),
    #     " ".join(query_structured_jd.get("responsibilities", [])),
    #     " ".join(query_structured_jd.get("requirements", [])),
    # ])

    # ================================
    # 수정 코드 (Pydantic model 대응) ⭕ 정상 동작
    # ================================
    # 방법 1) dict로 변환
    jd_dict = query_structured_jd.model_dump()

    role_summary = jd_dict.get("role_summary", "")

    responsibilities = " ".join(jd_dict.get("responsibilities", []) or [])
    requirements = " ".join(jd_dict.get("requirements", []) or [])
    preferred = " ".join(jd_dict.get("preferred", []) or [])

    # 검색 query 구성
    query = " ".join([
        role_summary,
        responsibilities,
        requirements,
        preferred,
        user_job or "",
        user_stack or "",
    ])

    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)

    return [d.page_content for d in docs]
