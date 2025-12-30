# rag/pipeline.py

import json
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from schemas import JDStructured, FullReport
from prompts import SYSTEM_PROMPT, JD_STRUCTURE_PROMPT, REPORT_PROMPT
from rag.retriever import build_retriever
from config import settings

from .retriever import retrieve_evidence


def _llm(streaming: bool = False):
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.temperature,
        streaming=streaming,
    )

def structure_jd(jd_text: str) -> JDStructured:
    """
    JD 원문을 JDStructured 스키마로 변환.
    LLM 출력이 스키마와 살짝 어긋나더라도 정규화하여 안전하게 반환.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", JD_STRUCTURE_PROMPT),
        ]
    )

    llm = _llm(streaming=False).with_structured_output(
        JDStructured,
        method="function_calling",
    )

    chain = prompt | llm

    raw = chain.invoke({"jd_text": jd_text})

      # -------------------------
    # 🔥 안전성 보정 단계 (여기 강화됨)
    # -------------------------
    def _fix_list_str(x):
        if x is None:
            return []

        fixed = []
        for v in x:
            # 이미 문자열이면 그대로
            if isinstance(v, str):
                fixed.append(v)
            # {"불명확":"내용"} -> 값만 꺼내기
            elif isinstance(v, dict):
                # dict의 value를 전부 연결
                fixed.append(" ".join(str(val) for val in v.values()))
            # 리스트가 또 들어온 경우
            elif isinstance(v, list):
                fixed.append(" ".join(str(val) for val in v))
            # 그 외 숫자, bool 다 문자열화
            else:
                fixed.append(str(v))

        return fixed

    # 🔽 JDStructured 내부 필드 값 안전하게 교체
    raw.core_competencies = _fix_list_str(getattr(raw, "core_competencies", []))
    raw.requirements = _fix_list_str(getattr(raw, "requirements", []))
    raw.preferred = _fix_list_str(getattr(raw, "preferred", []))
    raw.responsibilities = _fix_list_str(getattr(raw, "responsibilities", []))
    raw.tech_stack = _fix_list_str(getattr(raw, "tech_stack", []))

    return raw


def generate_report(
    jd_struct: JDStructured, essays: List[Dict[str, Any]], retrieved_summaries: List[str]
) -> FullReport:
    """
    JD 구조 + 자소서 + 유사사례 요약을 넣고 FullReport 스키마에 맞게 리포트 생성.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", REPORT_PROMPT),
        ]
    )

    # FullReport 스키마로 function calling 기반 structured output
    llm = _llm(streaming=False).with_structured_output(
        FullReport,
        method="function_calling",
    )

    chain = prompt | llm

    return chain.invoke(
        {
            "jd_json": jd_struct.model_dump_json(ensure_ascii=False),
            "essays_json": json.dumps(essays, ensure_ascii=False),
            "retrieved_summaries": json.dumps(
                retrieved_summaries,
                ensure_ascii=False,
            ),
        }
    )


def build_all(
    jd_text: str,
    essays: List[Dict[str, Any]],
    user_job: str = "",
    user_stack: str = "",
) -> FullReport:
    """
    한 번 호출로 전체 파이프라인 실행:
    JD 구조화 → 유사 사례 검색 → 최종 리포트 생성
    """
    jd_struct = structure_jd(jd_text)
    evidence = retrieve_evidence(jd_struct, user_job=user_job, user_stack=user_stack)
    report = generate_report(
        jd_struct,
        essays=essays,
        retrieved_summaries=evidence,
    )
    return report
