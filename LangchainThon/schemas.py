# Pydantic 스키마

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator

class Evidence(BaseModel):
    """수정 제안의 근거 자료"""
    source_type: Literal["jd", "user_text", "similar_case"] = Field(
        ...,
        description="근거 유형: JD 내용 | 사용자 작성 문장 | 유사 합격 사례"
    )
    content: str = Field(
        ...,
        description="근거 내용 (구체적 문장이나 키워드)"
    )
    relevance: Optional[float] = Field(
        default=None,
        ge=0, le=1,
        description="관련도 점수 (0.0~1.0, 선택적)"
    )

class JDStructured(BaseModel):
    """채용공고(JD)를 AI가 쓰기 좋게 정리한 구조"""

    # role_summary: str = Field(
    #     ...,
    #     description="이 포지션이 무엇을 하는 자리인지 2~4문장으로 간단 요약"
    # )

    role_summary: Optional[str] = Field(
        default="포지션 요약 정보를 추출할 수 없습니다.",
        description="이 포지션이 무엇을 하는 자리인지 2~4문장으로 간단 요약"
    )    
    responsibilities: List[str] = Field(
        default_factory=list,
        description="주요 업무(한 줄씩)"
    )
    requirements: List[str] = Field(
        default_factory=list,
        description="필수 자격요건(한 줄씩)"
    )
    preferred: List[str] = Field(
        default_factory=list,
        description="우대사항(한 줄씩)"
    )
    core_competencies: List[str] = Field(
        default_factory=list,
        description="핵심 역량 키워드 목록"
    )
    tech_stack: List[str] = Field(
        default_factory=list,
        description="기술 스택(툴/언어 등)"
    )


class SentenceEdit(BaseModel):
    """한 문장 단위로 수정 전/후와 이유를 담는 구조"""

    original: str = Field(..., description="사용자가 처음 작성한 문장")
    revised: str = Field(..., description="AI가 제안하는 수정 문장")
    rationale: str = Field(..., description="왜 이렇게 고쳤는지에 대한 설명")
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="수정에 참고한 근거(JD 문구, 다른 문장, 유사 사례 등)"
    )
    severity: Optional[Literal["high", "medium", "low"]] = Field(
        default="medium",
        description="수정 중요도(필요시 필터링용)"
    )


# class QuestionScores(BaseModel):
#     """문항 하나에 대한 점수 구조 (0~100)"""

#     overall: Optional[int] = Field(default=None, ge=0, le=100, description="종합 점수")
#     jd_match: Optional[int] = Field(default=None, ge=0, le=100, description="JD 적합도")
#     specificity: Optional[int] = Field(default=None, ge=0, le=100, description="구체성")
#     structure: Optional[int] = Field(default=None, ge=0, le=100, description="구조/논리")
#     impact: Optional[int] = Field(default=None, ge=0, le=100, description="임팩트/설득력")


# class OverallScores(BaseModel):
#     """자소서 전체에 대한 점수 구조 (0~100)"""

#     overall: Optional[int] = Field(default=None, ge=0, le=100, description="전체 종합 점수")
#     jd_match: Optional[int] = Field(default=None, ge=0, le=100, description="JD 적합도")
#     consistency: Optional[int] = Field(default=None, ge=0, le=100, description="문항 간 일관성")
#     differentiation: Optional[int] = Field(default=None, ge=0, le=100, description="차별화 정도")
#     clarity: Optional[int] = Field(default=None, ge=0, le=100, description="전반적인 가독성")

class QuestionScores(BaseModel):
    """문항 하나에 대한 점수 구조 (0~100)"""
    overall: int = Field(default=50, ge=0, le=100, description="종합 점수(100점 만점)")
    jd_match: int = Field(default=50, ge=0, le=100, description="JD 적합도(100점 만점)")
    specificity: int = Field(default=50, ge=0, le=100, description="구체성(100점 만점)")
    structure: int = Field(default=50, ge=0, le=100, description="구조/논리(100점 만점)")
    impact: int = Field(default=50, ge=0, le=100, description="임팩트/설득력(100점 만점)")


class OverallScores(BaseModel):
    """자소서 전체에 대한 점수 구조 (0~10)"""
    overall: int = Field(default=50, ge=0, le=100, description="전체 종합 점수(100점 만점)")
    jd_match: int = Field(default=50, ge=0, le=100, description="JD 적합도(100점 만점)")
    consistency: int = Field(default=50, ge=0, le=100, description="문항 간 일관성(100점 만점)")
    differentiation: int = Field(default=50, ge=0, le=100, description="차별화 정도(100점 만점)")
    clarity: int = Field(default=50, ge=0, le=100, description="전반적인 가독성(100점 만점)")

class QuestionReport(BaseModel):
    """문항(질문) 하나에 대한 피드백 리포트"""

    question_title: str = Field(
        ...,
        description="질문 제목 또는 번호. 예) '1번 문항 - 지원 동기'"
    )
    scores: QuestionScores = Field(
        default_factory=QuestionScores,
        description=(
            "overall, jd_match, specificity, structure, impact 를 "
            "0~100 사이 정수 또는 None으로 채우기"
        ),
    )
    top_improvements: List[str] = Field(
        default_factory=list,
        description="이 문항에서 먼저 고치면 좋은 핵심 개선 포인트 3~5개"
    )
    highlights_banned: List[str] = Field(
        default_factory=list,
        description="클리셰/피하는 게 좋은 표현들을 문장 그대로 모아둔 목록"
    )
    paragraph_labels: List[str] = Field(
        default_factory=list,
        description="각 문단 역할 라벨. 예) '도입', '경험 설명', '배운 점', '마무리'"
    )
    edits: List[SentenceEdit] = Field(
        default_factory=list,
        description="문장별 수정 제안 리스트"
    )


class FullReport(BaseModel):
    """자소서 전체에 대한 종합 피드백 리포트"""

    jd_structured: JDStructured = Field(
        ...,
        description="이번 지원에 사용된 채용공고 정리본"
    )
    overall_scores: OverallScores = Field(
        default_factory=OverallScores,
        description=(
            "자소서 전체에 대한 종합 점수. "
            "overall, jd_match, consistency, differentiation, clarity 사용"
        ),
    )
    overall_top3: List[str] = Field(
        default_factory=list,
        description="전체 자소서에서 꼭 먼저 손보면 좋은 핵심 개선 포인트 Top 3"
    )

    # ★ 여기서 per_question을 약간 느슨하게: QuestionReport가 아닌 값이 섞여 들어와도
    #   파싱 에러는 나지 않도록 Union[QuestionReport, str] 사용
    # per_question: List[Union[QuestionReport, str]] = Field(
    #     default_factory=list,
    #     description="문항별 상세 피드백 리포트 목록(가끔 잘못 들어온 문자열은 무시 가능)"
    # )
    per_question: List[QuestionReport] = Field(
        default_factory=list,
        description="문항별 상세 피드백 리포트 목록"
    )

    retrieved_evidence_summary: List[str] = Field(
        default_factory=list,
        description="AI가 참고한 유사 합격 사례/자료를 한 줄 요약으로 정리한 내용"
    )
    disclaimer: str = Field(
        (
            "이 리포트는 '합격/불합격'을 맞추는 도구가 아니라, "
            "합격자들의 자소서 패턴과 비교해서 더 탄탄하게 다듬을 수 있도록 돕는 코칭용 피드백입니다. "
            "점수에 너무 얽매이기보다는, 본인의 이야기를 솔직하고 구체적으로 만드는 데 활용해 주세요 :)"
        ),
        description="사용자에게 반드시 보여줄 안내 문구"
    )

    model_config = {
        "extra": "ignore",  # 혹시 LLM이 이상한 필드를 더 넣어도 무시
    }
####
    @field_validator("per_question", mode="before")
    @classmethod
    def filter_invalid_per_question(cls, v):
        """
        LLM이 per_question 리스트 안에 문자열이나 엉뚱한 값을 섞어 넣는 경우 방어용.

        - dict나 QuestionReport 인스턴스만 남기고 나머지는 버린다.
        - 'retrieved_evidence_summary([])', 'disclaimer":"..."' 같은 문자열은 다 버린다.
        """
        if v is None:
            return []

        # 통째로 문자열로 들어온 경우: JSON이면 파싱 시도, 아니면 버림
        if isinstance(v, str):
            try:
                import json
                parsed = json.loads(v)
                v = parsed
            except Exception:
                return []

        # dict 하나면 리스트로 감싸줌
        if isinstance(v, dict):
            return [v]

        # 리스트인 경우 항목별로 필터링
        if isinstance(v, list):
            cleaned = []
            import json

            for item in v:
                # 이미 dict면 통과
                if isinstance(item, dict):
                    cleaned.append(item)
                    continue

                # QuestionReport 인스턴스면 그대로
                if isinstance(item, QuestionReport):
                    cleaned.append(item)
                    continue

                # 문자열인 경우: 혹시 JSON dict면 파싱
                if isinstance(item, str):
                    try:
                        obj = json.loads(item)
                        if isinstance(obj, dict):
                            cleaned.append(obj)
                    except Exception:
                        # 'retrieved_evidence_summary([])', 'disclaimer":"..."' 이런 건 여기서 걸러짐
                        pass

            return cleaned

        # 그 외 타입은 전부 무시
        return []